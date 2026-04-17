"""Intent classification — decide whether a query is wiki-relevant or chitchat.

A separate, fast LLM call before retrieval saves time on small talk like
'hi', 'how are you', 'thanks' — which would otherwise trigger an 8-page
retrieval and synthesis pipeline that wastes ~30 seconds for nothing.

The classifier returns one of two labels:
    - 'wiki'      → proceed with normal retrieval + synthesis
    - 'chitchat'  → respond conversationally without searching

We deliberately bias toward 'wiki' on ambiguous cases. Wasting one retrieval
on an edge case is better than missing a real research question.
"""

from __future__ import annotations

from dataclasses import dataclass

from .llm import ChatMessage, LLMError, OllamaClient


INTENT_SYSTEM_PROMPT = """You are an intent classifier for a personal knowledge \
base. The user can ask questions that fall into two categories:

1. WIKI — questions seeking factual or thematic information that would be \
answered by searching documents. Examples: 'what is RAG', 'how does X relate \
to Y', 'summarize the main themes', 'when is the deadline for X', 'who \
mentioned X', 'compare A and B', any specific factual lookup or research \
question.

2. CHITCHAT — greetings, small talk, meta-questions about the assistant \
itself, or casual conversation. Examples: 'hi', 'hello', 'how are you', \
'thanks', 'good morning', 'what can you do', 'are you there', 'goodbye'.

Bias toward WIKI when uncertain — wasting one search is better than missing \
a real question.

Respond with a single word: WIKI or CHITCHAT. Nothing else.
"""


CHITCHAT_SYSTEM_PROMPT = """You are a friendly assistant attached to a \
personal knowledge wiki. The user just said something conversational rather \
than a research question.

Respond briefly and warmly in 1-2 sentences. If the message is a greeting, \
greet them back and gently invite them to ask a question about their wiki. \
Don't use markdown formatting, headers, or wikilinks — just plain text.
"""


@dataclass
class IntentResult:
    """Outcome of intent classification."""

    intent: str  # 'wiki' or 'chitchat'
    raw_response: str
    chitchat_reply: str | None = None


def classify_intent(
    client: OllamaClient,
    question: str,
    *,
    timeout_seconds: float = 10.0,
) -> IntentResult:
    """Ask Qwen3 (thinking off, low temperature) to classify the intent.

    Returns IntentResult.intent in {'wiki', 'chitchat'}. If classification
    fails for any reason, defaults to 'wiki' so the query still runs.
    """
    messages = [
        ChatMessage(role="system", content=INTENT_SYSTEM_PROMPT),
        ChatMessage(role="user", content=f"Question: {question}\n\nIntent:"),
    ]

    try:
        response = client.chat(messages, thinking=False, temperature=0.0)
    except LLMError:
        return IntentResult(intent="wiki", raw_response="<classification failed>")

    raw = response.strip().upper()

    # Look for the keywords. If neither appears clearly, default to wiki.
    if "CHITCHAT" in raw and "WIKI" not in raw:
        return IntentResult(intent="chitchat", raw_response=response)
    if raw.startswith("CHITCHAT"):
        return IntentResult(intent="chitchat", raw_response=response)

    return IntentResult(intent="wiki", raw_response=response)


def generate_chitchat_reply(
    client: OllamaClient,
    question: str,
) -> str:
    """Generate a brief conversational reply (used when intent='chitchat')."""
    messages = [
        ChatMessage(role="system", content=CHITCHAT_SYSTEM_PROMPT),
        ChatMessage(role="user", content=question),
    ]

    try:
        return client.chat(messages, thinking=False, temperature=0.7).strip()
    except LLMError:
        return "Hi! Ask me a question about your wiki and I'll search for an answer."
