"""Persistent ingest jobs — survive tab close, page reloads, and see multi-job history.

Architecture:
  - A JobManager singleton (per wiki_paths) owns a worker pool.
  - Jobs are persisted in `ingest_jobs` and their progress in `job_events`.
  - The manager accepts new jobs via enqueue(), runs them on worker threads,
    and writes events to SQLite as work progresses.
  - SSE clients read events by job_id + last_seen_seq, so reconnecting always
    catches up from where they left off. The CLI can do the same via polling.

Restart behavior:
  - On JobManager startup, any jobs in state 'running' are marked 'interrupted'
    (the worker died when the process did). User can re-ingest via source detail.

Concurrency:
  - MAX_CONCURRENT defaults to 2 (safe on 18GB with Qwen3-14B). Extra jobs queue.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from queue import Queue
from typing import Any, Callable, Optional

from . import config as cfg
from . import db
from . import ingest_llm
from . import ingest_raw
from .llm import LLMError, OllamaClient


MAX_CONCURRENT_DEFAULT = 2


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# DB CRUD
# ---------------------------------------------------------------------------


@dataclass
class JobRow:
    id: int
    source_id: int
    state: str
    phase: Optional[str]
    progress: float
    pages_created: int
    pages_updated: int
    error: Optional[str]
    created_at: str
    started_at: Optional[str]
    finished_at: Optional[str]
    # Joined columns
    source_relpath: Optional[str] = None
    source_type: Optional[str] = None


def _row_to_job(row: sqlite3.Row) -> JobRow:
    return JobRow(
        id=row["id"],
        source_id=row["source_id"],
        state=row["state"],
        phase=row["phase"],
        progress=row["progress"] or 0.0,
        pages_created=row["pages_created"] or 0,
        pages_updated=row["pages_updated"] or 0,
        error=row["error"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        source_relpath=row["source_relpath"] if "source_relpath" in row.keys() else None,
        source_type=row["source_type"] if "source_type" in row.keys() else None,
    )


def create_job(paths: cfg.WikiPaths, source_id: int) -> int:
    """Insert a new job row in 'queued' state. Returns job id."""
    with db.connect(paths.state_db) as conn:
        cur = conn.execute(
            "INSERT INTO ingest_jobs (source_id, state, created_at) "
            "VALUES (?, 'queued', ?)",
            (source_id, _now()),
        )
        conn.commit()
        return cur.lastrowid


def _update_job(paths: cfg.WikiPaths, job_id: int, **fields: Any) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k} = ?" for k in fields.keys())
    args = list(fields.values()) + [job_id]
    with db.connect(paths.state_db) as conn:
        conn.execute(f"UPDATE ingest_jobs SET {cols} WHERE id = ?", args)
        conn.commit()


def _append_event(
    paths: cfg.WikiPaths,
    job_id: int,
    kind: str,
    data: dict[str, Any],
) -> int:
    """Append an event to job_events. Returns the new seq number."""
    with db.connect(paths.state_db) as conn:
        cur = conn.execute(
            "SELECT COALESCE(MAX(seq), -1) + 1 FROM job_events WHERE job_id = ?",
            (job_id,),
        )
        next_seq = cur.fetchone()[0]
        conn.execute(
            "INSERT INTO job_events (job_id, seq, kind, data, at) "
            "VALUES (?, ?, ?, ?, ?)",
            (job_id, next_seq, kind, json.dumps(data), _now()),
        )
        conn.commit()
        return next_seq


def get_job(paths: cfg.WikiPaths, job_id: int) -> Optional[JobRow]:
    with db.connect(paths.state_db) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT j.*, s.relpath AS source_relpath, s.file_type AS source_type
            FROM ingest_jobs j
            LEFT JOIN sources s ON s.id = j.source_id
            WHERE j.id = ?
            """,
            (job_id,),
        )
        row = cur.fetchone()
        return _row_to_job(row) if row else None


def list_jobs(
    paths: cfg.WikiPaths,
    state: Optional[str] = None,
    limit: int = 50,
) -> list[JobRow]:
    """List jobs, newest first. Filter by state if provided."""
    with db.connect(paths.state_db) as conn:
        conn.row_factory = sqlite3.Row
        sql = """
            SELECT j.*, s.relpath AS source_relpath, s.file_type AS source_type
            FROM ingest_jobs j
            LEFT JOIN sources s ON s.id = j.source_id
        """
        args: list[Any] = []
        if state:
            sql += " WHERE j.state = ?"
            args.append(state)
        sql += " ORDER BY j.created_at DESC LIMIT ?"
        args.append(limit)
        return [_row_to_job(r) for r in conn.execute(sql, args).fetchall()]


def get_events_since(
    paths: cfg.WikiPaths, job_id: int, after_seq: int
) -> list[dict[str, Any]]:
    """Fetch events for job with seq > after_seq, in order."""
    with db.connect(paths.state_db) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT seq, kind, data, at FROM job_events "
            "WHERE job_id = ? AND seq > ? ORDER BY seq",
            (job_id, after_seq),
        ).fetchall()
        return [
            {"seq": r["seq"], "kind": r["kind"], "data": json.loads(r["data"]), "at": r["at"]}
            for r in rows
        ]


def prune_old_jobs(paths: cfg.WikiPaths, keep: int = 50) -> int:
    """Keep only the N most recent terminal (done/failed/interrupted) jobs.

    Running/queued jobs are never pruned. Returns number deleted.
    """
    with db.connect(paths.state_db) as conn:
        cur = conn.execute(
            """
            SELECT id FROM ingest_jobs
            WHERE state IN ('done', 'failed', 'interrupted')
            ORDER BY created_at DESC
            LIMIT -1 OFFSET ?
            """,
            (keep,),
        )
        to_delete = [r[0] for r in cur.fetchall()]
        if not to_delete:
            return 0
        placeholders = ",".join(["?"] * len(to_delete))
        conn.execute(
            f"DELETE FROM job_events WHERE job_id IN ({placeholders})", to_delete
        )
        conn.execute(
            f"DELETE FROM ingest_jobs WHERE id IN ({placeholders})", to_delete
        )
        conn.commit()
        return len(to_delete)


def mark_interrupted_on_startup(paths: cfg.WikiPaths) -> int:
    """On server startup, any 'running' job is actually a zombie (process died).

    Flip them to 'interrupted' so the UI shows them correctly. Returns count.
    """
    with db.connect(paths.state_db) as conn:
        cur = conn.execute(
            "UPDATE ingest_jobs SET state = 'interrupted', "
            "finished_at = ?, error = 'Server restarted during ingest' "
            "WHERE state IN ('running', 'queued')",
            (_now(),),
        )
        conn.commit()
        return cur.rowcount


# ---------------------------------------------------------------------------
# The ingest callbacks that write to job_events
# ---------------------------------------------------------------------------


class _JobCallbacks(ingest_llm.IngestCallbacks):
    """Pipes pipeline progress into the job_events table."""

    def __init__(self, paths: cfg.WikiPaths, job_id: int) -> None:
        self.paths = paths
        self.job_id = job_id

    def _emit(self, kind: str, data: dict[str, Any]) -> None:
        _append_event(self.paths, self.job_id, kind, data)

    def _set(self, **fields: Any) -> None:
        _update_job(self.paths, self.job_id, **fields)

    def on_start(self, source_id: int, source_title: str, file_path: str) -> None:
        self._set(state="running", started_at=_now(), phase="parsing", progress=0.05)
        self._emit(
            "status", {"text": f"Starting ingest of {file_path}", "phase": "parsing"}
        )

    def on_parsing(self) -> None:
        self._set(phase="parsing", progress=0.10)
        self._emit("status", {"text": "Parsing source file…", "phase": "parsing"})

    def on_extracting(self) -> None:
        self._set(phase="extracting", progress=0.20)
        self._emit(
            "status",
            {
                "text": "Extracting entities and concepts (thinking mode)…",
                "phase": "extracting",
            },
        )

    def on_extracted(self, extraction: ingest_llm.Extraction) -> None:
        self._set(phase="drafting", progress=0.40)
        self._emit(
            "extracted",
            {
                "title": extraction.title,
                "summary": extraction.summary,
                "entities": [
                    {"name": e.name, "slug": e.slug, "type": e.type}
                    for e in extraction.entities
                ],
                "concepts": [
                    {"name": c.name, "slug": c.slug} for c in extraction.concepts
                ],
                "tags": extraction.tags,
            },
        )

    def on_extraction_failed(self, error: str) -> None:
        self._emit("error", {"text": f"Extraction failed: {error}"})

    def on_drafting_page(self, kind: str, slug: str, operation: str) -> None:
        self._emit(
            "page_start",
            {"kind": kind, "slug": slug, "operation": operation},
        )

    def on_stream_chunk(self, chunk: str) -> None:
        # Too chatty to log individually — would balloon the DB.
        # We only log the major phase transitions.
        pass

    def on_page_written(self, page: ingest_llm.PageChange) -> None:
        self._emit(
            "page",
            {
                "kind": page.kind,
                "slug": page.slug,
                "path": page.path,
                "operation": page.operation,
            },
        )

    def on_finalizing(self) -> None:
        self._set(phase="finalizing", progress=0.90)
        self._emit("status", {"text": "Finalizing…", "phase": "finalizing"})

    def on_complete(self, result: ingest_llm.IngestResult) -> None:
        self._set(
            state="done",
            phase="done",
            progress=1.0,
            pages_created=result.pages_created,
            pages_updated=result.pages_updated,
            finished_at=_now(),
        )
        self._emit(
            "done",
            {
                "source_title": result.source_title,
                "source_slug": result.source_slug,
                "pages_created": result.pages_created,
                "pages_updated": result.pages_updated,
            },
        )

    def on_error(self, error: str) -> None:
        self._set(
            state="failed",
            phase="failed",
            error=error,
            finished_at=_now(),
        )
        self._emit("error", {"text": error})


# ---------------------------------------------------------------------------
# The JobManager — owns the worker pool
# ---------------------------------------------------------------------------


class JobManager:
    """Singleton per wiki project. Manages a pool of worker threads.

    Start it via get_manager(paths) — first call creates it; subsequent calls
    return the same instance.
    """

    def __init__(
        self, paths: cfg.WikiPaths, max_concurrent: int = MAX_CONCURRENT_DEFAULT
    ) -> None:
        self.paths = paths
        self.max_concurrent = max_concurrent
        self._queue: Queue[int] = Queue()
        self._workers: list[threading.Thread] = []
        self._started = False
        self._stop = threading.Event()

    def start(self) -> None:
        """Spawn worker threads (idempotent)."""
        if self._started:
            return
        self._started = True
        for i in range(self.max_concurrent):
            t = threading.Thread(
                target=self._worker_loop, name=f"ingest-worker-{i}", daemon=True
            )
            t.start()
            self._workers.append(t)

    def enqueue(self, source_id: int) -> int:
        """Create a job row and enqueue it for execution. Returns job_id."""
        self.start()
        job_id = create_job(self.paths, source_id)
        self._queue.put(job_id)
        return job_id

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=1.0)
            except Exception:
                continue
            try:
                self._run_job(job_id)
            except Exception as e:
                # Last-ditch protection — mark failed and continue
                _update_job(
                    self.paths,
                    job_id,
                    state="failed",
                    error=f"Worker exception: {e}",
                    finished_at=_now(),
                )
                _append_event(
                    self.paths, job_id, "error", {"text": f"Worker exception: {e}"}
                )
            finally:
                self._queue.task_done()

    def _run_job(self, job_id: int) -> None:
        job = get_job(self.paths, job_id)
        if job is None:
            return
        config = cfg.load_config(self.paths)
        llm_cfg = config.get("llm", {})
        client = OllamaClient(
            host=llm_cfg.get("host", "http://localhost:11434"),
            model=llm_cfg.get("model", "qwen3:14b"),
        )
        try:
            try:
                client.ensure_ready()
            except LLMError as e:
                _update_job(
                    self.paths,
                    job_id,
                    state="failed",
                    error=f"Ollama not ready: {e}",
                    finished_at=_now(),
                )
                _append_event(
                    self.paths, job_id, "error", {"text": f"Ollama not ready: {e}"}
                )
                return

            # Reload source to make sure it exists + has status='pending'
            source_row = ingest_raw.get_source(self.paths, job.source_id)
            if source_row is None:
                _update_job(
                    self.paths,
                    job_id,
                    state="failed",
                    error="Source no longer exists",
                    finished_at=_now(),
                )
                return

            callbacks = _JobCallbacks(self.paths, job_id)
            ingest_llm.ingest_source(
                self.paths, job.source_id, client, callbacks=callbacks
            )
            # Prune after each successful job to keep the jobs list short
            try:
                prune_old_jobs(self.paths, keep=50)
            except Exception:
                pass
        finally:
            try:
                client.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------


_manager_lock = threading.Lock()
_manager: Optional[JobManager] = None


def get_manager(
    paths: cfg.WikiPaths, max_concurrent: int = MAX_CONCURRENT_DEFAULT
) -> JobManager:
    """Return the singleton JobManager for this process, creating it if needed."""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = JobManager(paths, max_concurrent=max_concurrent)
            # Clean up any zombies from previous runs
            mark_interrupted_on_startup(paths)
            _manager.start()
        return _manager
