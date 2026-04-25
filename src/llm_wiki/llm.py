"""Ollama HTTP client — thin wrapper around the /api/chat endpoint.

Supports:
- Thinking mode toggle (Qwen3's /think, /no_think inline tags)
- JSON mode (format='json') with Pydantic validation
- Streaming for real-time page drafting
- Clear error messages when Ollama isn't running or the model isn't pulled

This module is intentionally stateless — each call is independent. Higher
layers (ingest_llm.py) orchestrate multi-pass pipelines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Generator, Iterator

import httpx

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "gemma3:1b"
DEFAULT_TIMEOUT = 300.0  # 5 minutes — thinking-mode extraction can be slow


class LLMError(Exception):
    """Raised when an LLM call fails in a user-recoverable way."""


class OllamaNotRunning(LLMError):
    """Ollama HTTP API is not reachable."""


class ModelNotFound(LLMError):
    """Requested model isn't pulled."""


@dataclass
class ChatMessage:
    role: str  # 'system' | 'user' | 'assistant'
    content: str


class OllamaClient:
    """Minimal, synchronous Ollama client tuned for the LLM-Wiki use case."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OllamaClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # -----------------------------------------------------------------
    # Health / liveness
    # -----------------------------------------------------------------

    def ping(self) -> bool:
        """True if Ollama is reachable. Does not check model availability."""
        try:
            r = self._client.get(f"{self.host}/api/tags", timeout=5.0)
            return r.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
            return False

    def list_models(self) -> list[str]:
        """List models available in this Ollama instance."""
        try:
            r = self._client.get(f"{self.host}/api/tags", timeout=5.0)
            r.raise_for_status()
            data = r.json()
            return [m.get("name", "") for m in data.get("models", [])]
        except httpx.ConnectError as e:
            raise OllamaNotRunning(
                f"Cannot connect to Ollama at {self.host}. "
                f"Is the Ollama app running?"
            ) from e
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama error: {e}") from e

    def ensure_ready(self) -> None:
        """Verify Ollama is running and the configured model is available.

        Raises OllamaNotRunning or ModelNotFound with user-friendly messages.
        """
        if not self.ping():
            raise OllamaNotRunning(
                f"Ollama isn't reachable at {self.host}.\n"
                f"Start it by opening the Ollama app, or run `ollama serve`."
            )
        models = self.list_models()
        # Match by exact name or prefix (ollama sometimes returns "qwen3:14b" as "qwen3:14b-q4_K_M")
        if not any(m == self.model or m.startswith(self.model) for m in models):
            raise ModelNotFound(
                f"Model '{self.model}' not found in Ollama.\n"
                f"Available: {', '.join(models) if models else '(none)'}\n"
                f"Pull it with: ollama pull {self.model}"
            )

    # -----------------------------------------------------------------
    # Chat (non-streaming)
    # -----------------------------------------------------------------

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        thinking: bool = False,
        json_mode: bool = False,
        temperature: float = 0.3,
    ) -> str:
        """Non-streaming chat. Returns the full assistant message content.

        For Qwen3, thinking mode is controlled via /think and /no_think
        inline tags in the last user message.
        """
        payload_messages = self._prepare_messages(messages, thinking=thinking)
        payload = {
            "model": self.model,
            "messages": payload_messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"

        try:
            r = self._client.post(f"{self.host}/api/chat", json=payload)
            r.raise_for_status()
        except httpx.ConnectError as e:
            raise OllamaNotRunning(
                f"Cannot connect to Ollama at {self.host}."
            ) from e
        except httpx.HTTPStatusError as e:
            body = e.response.text
            if "not found" in body.lower() or e.response.status_code == 404:
                raise ModelNotFound(
                    f"Model '{self.model}' not found. "
                    f"Pull it with: ollama pull {self.model}"
                ) from e
            raise LLMError(f"Ollama error {e.response.status_code}: {body}") from e
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama request failed: {e}") from e

        data = r.json()
        content = data.get("message", {}).get("content", "")
        return self._strip_thinking(content)

    # -----------------------------------------------------------------
    # Chat (streaming)
    # -----------------------------------------------------------------

    def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        thinking: bool = False,
        temperature: float = 0.3,
    ) -> Generator[str, None, str]:
        """Streaming chat. Yields content chunks as they arrive.

        Returns (via generator return value) the full accumulated content
        after the stream closes. Use `yield from` to forward chunks while
        still capturing the final result:

            def consume():
                full = yield from client.chat_stream(messages)
                return full
        """
        payload_messages = self._prepare_messages(messages, thinking=thinking)
        payload = {
            "model": self.model,
            "messages": payload_messages,
            "stream": True,
            "options": {"temperature": temperature},
        }

        full_content: list[str] = []
        in_thinking_block = False

        try:
            with self._client.stream(
                "POST", f"{self.host}/api/chat", json=payload
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msg = data.get("message", {})
                    chunk = msg.get("content", "")
                    if not chunk:
                        if data.get("done"):
                            break
                        continue

                    # Strip thinking blocks on the fly
                    visible = ""
                    i = 0
                    while i < len(chunk):
                        if not in_thinking_block:
                            start = chunk.find("<think>", i)
                            if start == -1:
                                visible += chunk[i:]
                                break
                            visible += chunk[i:start]
                            in_thinking_block = True
                            i = start + len("<think>")
                        else:
                            end = chunk.find("</think>", i)
                            if end == -1:
                                break  # rest of chunk is thinking
                            in_thinking_block = False
                            i = end + len("</think>")

                    if visible:
                        full_content.append(visible)
                        yield visible

                    if data.get("done"):
                        break
        except httpx.ConnectError as e:
            raise OllamaNotRunning(
                f"Cannot connect to Ollama at {self.host}."
            ) from e
        except httpx.HTTPStatusError as e:
            body = e.response.read().decode(errors="replace") if e.response else ""
            raise LLMError(f"Ollama error {e.response.status_code}: {body}") from e
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama streaming failed: {e}") from e

        return "".join(full_content)

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _prepare_messages(
        self, messages: list[ChatMessage], *, thinking: bool
    ) -> list[dict]:
        """Convert to Ollama's wire format and append the Qwen3 thinking tag."""
        result = [{"role": m.role, "content": m.content} for m in messages]
        if result:
            tag = "\n\n/think" if thinking else "\n\n/no_think"
            # Only append to the last user message (Qwen3 convention)
            for i in range(len(result) - 1, -1, -1):
                if result[i]["role"] == "user":
                    result[i]["content"] += tag
                    break
        return result

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove <think>...</think> blocks from a completed response."""
        if "<think>" not in text:
            return text
        import re
        return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
