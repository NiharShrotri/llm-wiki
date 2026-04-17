"""Sources routes — list all tracked sources, show detail with text preview."""

from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from ... import config as cfg
from ... import ingest_raw
from ... import parsers

router = APIRouter(prefix="/sources")


def _format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.2f} GB"


@router.get("", response_class=HTMLResponse)
async def sources_list(
    request: Request, status: str | None = None
) -> HTMLResponse:
    """Render the sources list page, optionally filtered by status."""
    paths: cfg.WikiPaths = request.app.state.wiki_paths
    rows = ingest_raw.list_sources(paths, status_filter=status)

    # Decorate rows with human-friendly formatting
    decorated = []
    for row in rows:
        decorated.append(
            {
                **row,
                "size_human": _format_bytes(row["bytes"]),
                "added_short": row["added_at"][:10] if row.get("added_at") else "",
                "last_ingested_short": (
                    row["last_ingested"][:10]
                    if row.get("last_ingested")
                    else None
                ),
            }
        )

    # Counts by status for the filter tabs
    all_rows = ingest_raw.list_sources(paths)
    status_counts = {"all": len(all_rows), "pending": 0, "ingested": 0, "error": 0}
    for row in all_rows:
        st = row.get("status", "pending")
        if st in status_counts:
            status_counts[st] += 1

    return request.app.state.templates.TemplateResponse(
        request,
        "sources_list.html",
        {
            "sources": decorated,
            "status_filter": status,
            "status_counts": status_counts,
            "page": "sources",
        },
    )


@router.get("/{source_id}", response_class=HTMLResponse)
async def source_detail(request: Request, source_id: int) -> HTMLResponse:
    """Render the detail page for a single source.

    Re-parses the file on the fly to get title, word count, and a text
    preview. This is cheap (local file, fast parsers) and avoids storing
    large parsed text in the DB.
    """
    paths: cfg.WikiPaths = request.app.state.wiki_paths
    row = ingest_raw.get_source(paths, source_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No source with id {source_id}")

    file_path = paths.root / row["relpath"]
    parsed = None
    preview = ""
    error: str | None = None
    if not file_path.exists():
        error = f"Source file missing from disk: {row['relpath']}"
    else:
        try:
            parsed = parsers.parse(file_path)
            preview_chars = 3000
            preview = parsed.text[:preview_chars]
            if len(parsed.text) > preview_chars:
                preview += (
                    f"\n\n[... {len(parsed.text) - preview_chars:,} more characters ...]"
                )
        except parsers.ParserError as e:
            error = f"Parse failed: {e}"

    # Find wiki pages that reference this source (from source_pages table)
    related_pages: list[dict] = []
    from ... import db as _db

    with _db.connect(paths.state_db) as conn:
        page_rows = conn.execute(
            "SELECT wiki_path, operation, at FROM source_pages WHERE source_id = ? ORDER BY wiki_path",
            (source_id,),
        ).fetchall()
        for pr in page_rows:
            related_pages.append(
                {
                    "path": pr["wiki_path"],
                    "operation": pr["operation"],
                    "at": pr["at"],
                }
            )

    metadata_rows: list[tuple[str, str]] = []
    metadata_rows.append(("Path", row["relpath"]))
    metadata_rows.append(("File type", row["file_type"]))
    metadata_rows.append(("Size", _format_bytes(row["bytes"])))
    if parsed:
        metadata_rows.append(("Words", f"{parsed.word_count:,}"))
    metadata_rows.append(("Added", row["added_at"]))
    metadata_rows.append(("Status", row["status"]))
    metadata_rows.append(("Hash", row["content_hash"][:16] + "…"))
    if row.get("last_ingested"):
        metadata_rows.append(("Last ingested", row["last_ingested"]))
    if parsed:
        for k, v in parsed.metadata.items():
            metadata_rows.append((k, str(v)[:120]))

    return request.app.state.templates.TemplateResponse(
        request,
        "source_detail.html",
        {
            "source": row,
            "title": parsed.title if parsed else row["relpath"],
            "metadata_rows": metadata_rows,
            "preview": preview,
            "error": error,
            "related_pages": related_pages,
            "page": "sources",
        },
    )


# ---------------------------------------------------------------------------
# Source mutation endpoints — delete and re-ingest
# ---------------------------------------------------------------------------


@router.post("/{source_id}/delete")
async def source_delete(request: Request, source_id: int) -> RedirectResponse:
    """Delete a source from raw/ and remove its DB row.

    Does NOT cascade to wiki pages — wiki pages created from this source
    stay intact (matches the Karpathy 'wiki accumulates knowledge' pattern).
    Run `wiki lint --fix` afterward to clean up any orphaned references.
    """
    paths: cfg.WikiPaths = request.app.state.wiki_paths
    success, message = ingest_raw.remove_source(paths, source_id, delete_file=True)
    if not success:
        raise HTTPException(status_code=404, detail=message)
    # Redirect to sources list with a flash-style message in the URL
    return RedirectResponse(url="/sources", status_code=303)


@router.post("/{source_id}/reingest")
async def source_reingest(request: Request, source_id: int) -> RedirectResponse:
    """Reset a source to 'pending' status so it can be re-ingested.

    Useful when a previous ingest was interrupted, produced poor extractions,
    or you want to retry with newer prompts. Doesn't touch wiki pages —
    re-ingestion will use the merge path on existing pages.
    """
    paths: cfg.WikiPaths = request.app.state.wiki_paths
    success, message = ingest_raw.mark_source_pending(paths, source_id)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    # Send the user to the ingest page where the source now appears in pending
    return RedirectResponse(url="/ingest", status_code=303)
