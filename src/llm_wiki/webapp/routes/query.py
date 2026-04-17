"""Query route — chat-style interface with SSE streaming.

Two endpoints:
  GET  /query              — render the chat shell
  GET  /query/stream?q=... — SSE stream of search hits + answer chunks
  POST /query/save         — save an answer as a synthesis page
"""

from __future__ import annotations

import json
import queue
import threading
from typing import Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from ... import config as cfg
from ... import page_writer
from ... import query as query_module
from ... import search
from ...llm import OllamaClient

router = APIRouter()


@router.get("/query", response_class=HTMLResponse)
async def query_page(request: Request) -> HTMLResponse:
    """Render the chat-style query interface."""
    return request.app.state.templates.TemplateResponse(
        request,
        "query.html",
        {"page": "query"},
    )


def _sse_format(event: str, data: dict | str) -> str:
    """Format an SSE event line. Always JSON-encode the data payload."""
    if isinstance(data, str):
        payload = json.dumps({"text": data})
    else:
        payload = json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


class _SSECallbacks(query_module.QueryCallbacks):
    """QueryCallbacks implementation that pushes events into a queue.

    The query runs on a worker thread; the SSE generator pulls events out
    of the queue as they arrive.
    """

    def __init__(self, q: "queue.Queue[tuple[str, Any]]") -> None:
        self.q = q

    def on_start(self, question: str, mode: str) -> None:
        self.q.put(("start", {"question": question, "mode": mode}))

    def on_classifying_intent(self) -> None:
        self.q.put(("status", {"text": "Understanding your question…"}))

    def on_intent_classified(self, intent: str) -> None:
        self.q.put(("intent", {"intent": intent}))

    def on_chitchat_reply(self, reply: str) -> None:
        self.q.put(("chitchat", {"text": reply}))

    def on_searching(self) -> None:
        self.q.put(("status", {"text": "Searching wiki…"}))

    def on_search_done(self, results: search.SearchResults) -> None:
        hits = []
        for h in results.hits:
            import re as _re
            raw = h.full_path.removesuffix(".md")
            cleaned = _re.sub(r"^/?qmd://[^/]+/", "", raw).lstrip("/")
            hits.append(
                {
                    "path": cleaned,
                    "title": h.title or cleaned.rsplit("/", 1)[-1],
                    "score": round(h.score, 3),
                    "snippet": (h.snippet or "")[:200],
                }
            )
        self.q.put(("hits", {"hits": hits, "count": len(hits)}))

    def on_no_results(self) -> None:
        self.q.put(("status", {"text": "No matching pages found."}))

    def on_synthesizing(self) -> None:
        self.q.put(("status", {"text": "Synthesizing answer…"}))

    def on_stream_chunk(self, chunk: str) -> None:
        self.q.put(("chunk", {"text": chunk}))

    def on_complete(self, result: query_module.QueryResult) -> None:
        self.q.put(("complete", {"answer": result.answer}))

    def on_error(self, error: str) -> None:
        self.q.put(("error", {"text": error}))


@router.get("/query/stream")
async def query_stream(
    request: Request,
    q: str,
    scope: str = "wiki",  # 'wiki' | 'raw' | 'hybrid'
) -> StreamingResponse:
    """SSE stream: run the query in a worker thread, pipe progress events."""
    paths: cfg.WikiPaths = request.app.state.wiki_paths
    config = cfg.load_config(paths)
    llm_cfg = config.get("llm", {})

    # Validate scope
    if scope not in ("wiki", "raw", "hybrid"):
        scope = "wiki"

    event_q: "queue.Queue[tuple[str, Any]]" = queue.Queue()
    done_event = threading.Event()
    result_holder: dict[str, query_module.QueryResult | None] = {"result": None}

    def worker() -> None:
        client = OllamaClient(
            host=llm_cfg.get("host", "http://localhost:11434"),
            model=llm_cfg.get("model", "qwen3:14b"),
        )
        try:
            try:
                client.ensure_ready()
            except Exception as e:
                event_q.put(("error", {"text": f"Ollama not ready: {e}"}))
                return

            callbacks = _SSECallbacks(event_q)
            try:
                result = query_module.run_query(
                    paths,
                    client,
                    question=q,
                    callbacks=callbacks,
                    mode="hybrid",
                    limit=8,
                    min_score=0.0,
                    rerank=True,
                    save_as=None,
                    scope=scope,
                    classify_intent_first=True,
                )
                result_holder["result"] = result
            except Exception as e:
                event_q.put(("error", {"text": f"Query failed: {e}"}))
        finally:
            try:
                client.close()
            except Exception:
                pass
            done_event.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    async def event_generator():
        import asyncio

        # Initial heartbeat so the browser knows the connection is alive
        yield _sse_format("status", {"text": "Connecting…"})

        # Use a short timeout on each get() so we can periodically check
        # whether the worker is done. This gives near-instant chunk delivery
        # (no fixed 50ms polling delay) while still letting the loop exit
        # cleanly if the worker finishes without sending a 'complete' event.
        loop = asyncio.get_event_loop()
        while True:
            try:
                # Run the blocking queue.get() in a thread so we don't
                # block the asyncio event loop. Timeout means we wake up
                # periodically to check done_event.
                event_name, payload = await loop.run_in_executor(
                    None, lambda: event_q.get(timeout=0.5)
                )
                yield _sse_format(event_name, payload)
                if event_name in ("complete", "error"):
                    return
            except queue.Empty:
                # Timeout — check if worker finished
                if done_event.is_set():
                    yield _sse_format("done", {"text": ""})
                    return
                # Otherwise loop and wait for the next chunk
                continue

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/query/save")
async def query_save(
    request: Request,
    question: str = Form(...),
    answer: str = Form(...),
    slug: str = Form(...),
) -> JSONResponse:
    """Save a synthesis answer as a wiki page.

    Reuses page_writer.write_page so the same logic that the CLI uses
    handles validation, frontmatter, and atomic writes.
    """
    paths: cfg.WikiPaths = request.app.state.wiki_paths

    # Sanitize slug — let the slugify module handle it for safety
    from ... import slugify

    safe_slug = slugify.slugify(slug)
    if not safe_slug:
        return JSONResponse(
            {"ok": False, "error": "invalid slug"}, status_code=400
        )

    today = page_writer.today_iso()
    target_path = paths.wiki / "synthesis" / f"{safe_slug}.md"

    if target_path.exists():
        return JSONResponse(
            {"ok": False, "error": f"synthesis/{safe_slug}.md already exists"},
            status_code=409,
        )

    # Build the synthesis page content
    frontmatter = {
        "title": question,
        "type": "synthesis",
        "tags": ["synthesis"],
        "created": today,
        "updated": today,
        "question": question,
        "confidence": "medium",
    }
    body = f"# {question}\n\n{answer}\n"

    parsed = page_writer.ParsedPage(frontmatter=frontmatter, body=body)
    page_writer.write_page(target_path, parsed.to_markdown())

    # Update index + log
    page_writer.rebuild_index(paths, today)
    page_writer.append_log_entry(
        paths,
        today,
        action="query",
        title=question,
        bullets=[f"saved: [[synthesis/{safe_slug}]]"],
    )

    return JSONResponse(
        {"ok": True, "saved_path": f"synthesis/{safe_slug}.md"}
    )
