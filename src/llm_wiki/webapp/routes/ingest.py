"""Ingest route — persistent-jobs version.

Flow:
  1. GET /ingest — landing page with drag-drop upload + pending list + recent jobs
  2. POST /ingest/upload — accept files, copy into raw/, register sources
  3. POST /ingest/start — create a job for a given source_id, queue it
  4. GET /ingest/jobs/{job_id}/stream — SSE: replay history + live tail

Because progress is persisted in SQLite (via jobs.py), closing the tab and
coming back still shows full history. Multiple tabs can watch the same job.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from ... import config as cfg
from ... import ingest_raw
from ... import jobs as jobs_module

router = APIRouter()


@router.get("/ingest", response_class=HTMLResponse)
async def ingest_page(request: Request) -> HTMLResponse:
    paths: cfg.WikiPaths = request.app.state.wiki_paths
    pending = [s for s in ingest_raw.list_sources(paths) if s["status"] == "pending"]
    recent_jobs = jobs_module.list_jobs(paths, limit=20)
    return request.app.state.templates.TemplateResponse(
        request,
        "ingest.html",
        {"pending": pending, "recent_jobs": recent_jobs, "page": "ingest"},
    )


@router.post("/ingest/upload")
async def ingest_upload(
    request: Request, files: list[UploadFile] = File(...)
) -> JSONResponse:
    paths: cfg.WikiPaths = request.app.state.wiki_paths
    paths.raw.mkdir(parents=True, exist_ok=True)
    results = []
    for up in files:
        if not up.filename:
            continue
        dest = paths.raw / up.filename
        content = await up.read()
        dest.write_bytes(content)
        outcome = ingest_raw.add_file(paths, dest, copy=False)
        results.append(
            {
                "filename": up.filename,
                "result": outcome.result.name,
                "source_id": outcome.source_id,
            }
        )
    return JSONResponse({"ok": True, "files": results})


@router.post("/ingest/start")
async def ingest_start(request: Request) -> JSONResponse:
    paths: cfg.WikiPaths = request.app.state.wiki_paths
    form = await request.form()
    source_id_raw = form.get("source_id")
    if source_id_raw is None:
        raise HTTPException(status_code=400, detail="source_id required")
    try:
        source_id = int(source_id_raw)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="source_id must be int")
    row = ingest_raw.get_source(paths, source_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No source with id {source_id}")
    manager = jobs_module.get_manager(paths)
    job_id = manager.enqueue(source_id)
    return JSONResponse({"ok": True, "job_id": job_id})


@router.get("/ingest/jobs/{job_id}/stream")
async def ingest_job_stream(request: Request, job_id: int) -> StreamingResponse:
    paths: cfg.WikiPaths = request.app.state.wiki_paths
    if jobs_module.get_job(paths, job_id) is None:
        raise HTTPException(status_code=404, detail=f"No job {job_id}")

    async def generator():
        last_seq = -1
        terminal_states = {"done", "failed", "interrupted"}
        job = jobs_module.get_job(paths, job_id)
        if job:
            yield (
                f"event: job\ndata: "
                f"{json.dumps({'state': job.state, 'phase': job.phase, 'progress': job.progress})}\n\n"
            )
        for _ in range(3600):
            if await request.is_disconnected():
                break
            events = jobs_module.get_events_since(paths, job_id, last_seq)
            for ev in events:
                yield f"event: {ev['kind']}\ndata: {json.dumps(ev['data'])}\n\n"
                last_seq = ev["seq"]
            job = jobs_module.get_job(paths, job_id)
            if job is None:
                break
            if job.state in terminal_states:
                yield (
                    f"event: job\ndata: "
                    f"{json.dumps({'state': job.state, 'phase': job.phase, 'progress': job.progress, 'pages_created': job.pages_created, 'pages_updated': job.pages_updated, 'error': job.error})}\n\n"
                )
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/jobs", response_class=HTMLResponse)
async def jobs_page(request: Request) -> HTMLResponse:
    paths: cfg.WikiPaths = request.app.state.wiki_paths
    all_jobs = jobs_module.list_jobs(paths, limit=50)
    counts = {"queued": 0, "running": 0, "done": 0, "failed": 0, "interrupted": 0}
    for j in all_jobs:
        counts[j.state] = counts.get(j.state, 0) + 1
    return request.app.state.templates.TemplateResponse(
        request,
        "jobs.html",
        {"jobs": all_jobs, "counts": counts, "page": "jobs"},
    )


@router.get("/api/jobs")
async def api_jobs(request: Request) -> JSONResponse:
    paths: cfg.WikiPaths = request.app.state.wiki_paths
    all_jobs = jobs_module.list_jobs(paths, limit=50)
    return JSONResponse(
        {
            "jobs": [
                {
                    "id": j.id,
                    "source_id": j.source_id,
                    "source_relpath": j.source_relpath,
                    "source_type": j.source_type,
                    "state": j.state,
                    "phase": j.phase,
                    "progress": j.progress,
                    "pages_created": j.pages_created,
                    "pages_updated": j.pages_updated,
                    "error": j.error,
                    "created_at": j.created_at,
                    "started_at": j.started_at,
                    "finished_at": j.finished_at,
                }
                for j in all_jobs
            ]
        }
    )
