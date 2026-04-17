"""Query pipeline: search → synthesize → (optionally) save back as a page.

Flow:
  1. Call search.query() to get top-K matching wiki pages.
  2. Build a synthesis prompt with the matches as context.
  3. Stream Qwen3's answer via OllamaClient.chat_stream.
  4. Parse the answer for [[wikilinks]] to validate them.
  5. Optionally save the answer as wiki/synthesis/<slug>.md and update
     index + log.

Callbacks let the CLI render progress (search phase, synthesis streaming).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import config as cfg
from . import db
from . import page_writer
from . import prompts
from . import search
from . import slugify
from .llm import (
    LLMError,
    ModelNotFound,
    OllamaClient,
    OllamaNotRunning,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    """The outcome of a single `wiki query` invocation."""

    question: str
    answer: str = ""
    hits: list[search.SearchHit] = field(default_factory=list)
    saved_path: str | None = None   # if --save-as was used
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.answer)


# ---------------------------------------------------------------------------
# Progress callbacks
# ---------------------------------------------------------------------------


class QueryCallbacks:
    """Hooks the CLI provides to render progress during a query."""

    def on_start(self, question: str, mode: str) -> None: ...

    def on_classifying_intent(self) -> None: ...

    def on_intent_classified(self, intent: str) -> None: ...

    def on_chitchat_reply(self, reply: str) -> None: ...

    def on_searching(self) -> None: ...

    def on_search_done(self, results: search.SearchResults) -> None: ...

    def on_no_results(self) -> None: ...

    def on_synthesizing(self) -> None: ...

    def on_stream_chunk(self, chunk: str) -> None: ...

    def on_saved(self, saved_path: str) -> None: ...

    def on_complete(self, result: QueryResult) -> None: ...

    def on_error(self, error: str) -> None: ...


# ---------------------------------------------------------------------------
# Synthesis prompt
# ---------------------------------------------------------------------------


SYNTHESIS_SYSTEM_PROMPT = """You are answering questions by synthesizing content
from an LLM-Wiki knowledge base.

Rules:
1. Base your answer STRICTLY on the provided wiki pages and excerpts below.
   Do not invent facts or speculate beyond what the sources say.
2. When referencing a wiki page, use [[wikilinks]] with the same path format
   shown in the source headers (e.g. [[entities/karpathy]], [[concepts/rag]]).
3. If the sources don't contain enough information to answer confidently,
   say so explicitly. Suggest what additional sources would help.
4. Be concise but substantive. Write in clean markdown (headers, bullets,
   paragraphs as appropriate).
5. Do NOT include YAML frontmatter, preamble like "Based on the sources:",
   or meta-commentary about your process.
"""


def _build_synthesis_user_prompt(
    question: str, results: search.SearchResults
) -> str:
    """Construct the user message for Qwen3 with the search results as context."""
    lines: list[str] = []
    lines.append(f"Question: {question}")
    lines.append("")
    lines.append(f"Here are the {len(results)} most relevant wiki pages:")
    lines.append("")

    for i, hit in enumerate(results.hits, start=1):
        lines.append(f"--- Source {i} ---")
        # Use the full hit path so the LLM sees how to wikilink it.
        # Strip the qmd:// URI prefix and collection name so the link is
        # clean and Obsidian-friendly.
        import re as _re
        raw_path = hit.full_path.removesuffix(".md")
        page_link = _re.sub(r"^/?qmd://[^/]+/", "", raw_path).lstrip("/")
        lines.append(f"Wikilink path: [[{page_link}]]")
        if hit.title:
            lines.append(f"Title: {hit.title}")
        lines.append(f"Relevance score: {hit.score:.2f}")
        lines.append("")
        if hit.full_content:
            lines.append(hit.full_content)
        elif hit.snippet:
            lines.append(hit.snippet)
        else:
            lines.append("(no content available)")
        lines.append("")

    lines.append("--- End of sources ---")
    lines.append("")
    lines.append(
        "Now write a clear, well-structured markdown answer to the question. "
        "Cite each claim with [[wikilinks]] using the paths shown above. "
        "If a claim is not supported by the sources, say so."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Save-back logic
# ---------------------------------------------------------------------------


def _save_synthesis_page(
    paths: cfg.WikiPaths,
    question: str,
    answer: str,
    save_slug: str,
    hits: list[search.SearchHit],
) -> str:
    """Write the answer to wiki/synthesis/<slug>.md, rebuild index, append log.

    Returns the relative path to the saved page.
    """
    slug = slugify.slugify(save_slug)
    if not slug:
        slug = slugify.slugify(question) or "synthesis"

    today = page_writer.today_iso()

    # Derive the list of sources that contributed to this synthesis.
    # Normalize each hit path so we store clean refs like 'sources/foo'
    # instead of '/qmd://llm-wiki-pages/sources/foo'.
    source_refs: list[str] = []
    for hit in hits:
        raw = hit.full_path.removesuffix(".md")
        # Strip qmd:// URI prefix and any collection name prefix
        import re as _re
        cleaned = _re.sub(r"^/?qmd://[^/]+/", "", raw).lstrip("/")
        if cleaned and cleaned not in source_refs:
            source_refs.append(cleaned)

    # Build the final page with frontmatter
    parsed = page_writer.parse_page(answer.strip())
    if not parsed.frontmatter:
        parsed.frontmatter = {
            "title": question if len(question) <= 80 else question[:77] + "...",
            "type": "synthesis",
            "tags": ["synthesis"],
            "created": today,
            "updated": today,
            "question": question,
            "sources_consulted": source_refs,
            "confidence": "medium",
        }
        # Wrap body with a nice H1 and the answer
        title_line = f"# {parsed.frontmatter['title']}"
        parsed.body = f"{title_line}\n\n{answer.strip()}"

    final_page = parsed.to_markdown()
    final_path = paths.wiki / "synthesis" / f"{slug}.md"
    page_writer.write_page(final_path, final_page)

    # Rebuild index & log
    page_writer.rebuild_index(paths, today)
    page_writer.append_log_entry(
        paths,
        today,
        "query",
        question if len(question) <= 80 else question[:77] + "...",
        [
            f"saved: [[synthesis/{slug}]]",
            f"consulted: {len(hits)} page(s)",
        ],
    )

    return f"synthesis/{slug}.md"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_query(
    paths: cfg.WikiPaths,
    client: OllamaClient,
    question: str,
    callbacks: QueryCallbacks,
    *,
    mode: str = "hybrid",
    limit: int = 8,
    min_score: float = 0.0,
    rerank: bool = True,
    save_as: str | None = None,
    temperature: float = 0.3,
    scope: str = "wiki",  # 'wiki' | 'raw' | 'hybrid'
    classify_intent_first: bool = True,
) -> QueryResult:
    """Run a full query → answer pipeline.

    scope determines which QMD collection(s) to search:
        - 'wiki'   → llm-wiki-pages only (LLM-summarized knowledge)
        - 'raw'    → llm-wiki-raw only (original source documents)
        - 'hybrid' → both, results merged

    classify_intent_first runs intent classification before retrieval. If
    the user asked something like 'hi' or 'thanks', we skip retrieval and
    respond conversationally.
    """
    callbacks.on_start(question, mode)

    # 0. Intent classification — skip retrieval for chitchat
    if classify_intent_first:
        from . import intent as intent_module

        callbacks.on_classifying_intent()
        intent_result = intent_module.classify_intent(client, question)
        callbacks.on_intent_classified(intent_result.intent)

        if intent_result.intent == "chitchat":
            reply = intent_module.generate_chitchat_reply(client, question)
            callbacks.on_chitchat_reply(reply)
            result = QueryResult(question=question, answer=reply, hits=[])
            callbacks.on_complete(result)
            return result

    # 1. Search — pick collections based on scope
    callbacks.on_searching()
    if scope == "wiki":
        collections_to_search: list[str] | None = ["llm-wiki-pages"]
    elif scope == "raw":
        collections_to_search = ["llm-wiki-raw"]
    elif scope == "hybrid":
        collections_to_search = ["llm-wiki-pages", "llm-wiki-raw"]
    else:
        collections_to_search = None  # use QMD default (all collections)

    try:
        results = search.query(
            paths,
            question,
            mode=mode,
            limit=limit,
            min_score=min_score,
            collections=collections_to_search,
            hydrate=True,
            rerank=rerank,
        )
    except search.QmdNotInstalled as e:
        result = QueryResult(question=question, error=str(e))
        callbacks.on_error(result.error)
        return result
    except search.SearchBackendError as e:
        result = QueryResult(question=question, error=f"Search failed: {e}")
        callbacks.on_error(result.error)
        return result

    callbacks.on_search_done(results)

    if len(results) == 0:
        callbacks.on_no_results()
        result = QueryResult(
            question=question,
            answer="",
            hits=[],
            error="No matching wiki pages found. Try a different query or "
            "ingest more sources.",
        )
        callbacks.on_error(result.error)
        return result

    # 2. Synthesize
    callbacks.on_synthesizing()
    system_msg = prompts.ChatMessage(role="system", content=SYNTHESIS_SYSTEM_PROMPT)
    user_msg = prompts.ChatMessage(
        role="user", content=_build_synthesis_user_prompt(question, results)
    )
    messages = [system_msg, user_msg]

    answer_parts: list[str] = []
    try:
        gen = client.chat_stream(messages, thinking=False, temperature=temperature)
        try:
            while True:
                chunk = next(gen)
                callbacks.on_stream_chunk(chunk)
                answer_parts.append(chunk)
        except StopIteration as stop:
            if stop.value:
                full = stop.value
                if full and not answer_parts:
                    answer_parts.append(full)
    except (OllamaNotRunning, ModelNotFound) as e:
        result = QueryResult(
            question=question, hits=results.hits, error=str(e)
        )
        callbacks.on_error(result.error)
        return result
    except LLMError as e:
        result = QueryResult(
            question=question, hits=results.hits, error=f"LLM error: {e}"
        )
        callbacks.on_error(result.error)
        return result

    answer = page_writer.strip_llm_noise("".join(answer_parts))

    # 3. Optionally save as a synthesis page
    saved_path: str | None = None
    if save_as:
        try:
            saved_path = _save_synthesis_page(
                paths, question, answer, save_as, results.hits
            )
            callbacks.on_saved(saved_path)
        except OSError as e:
            result = QueryResult(
                question=question,
                answer=answer,
                hits=results.hits,
                error=f"Failed to save synthesis page: {e}",
            )
            callbacks.on_error(result.error)
            return result

    result = QueryResult(
        question=question,
        answer=answer,
        hits=results.hits,
        saved_path=saved_path,
    )
    callbacks.on_complete(result)
    return result
