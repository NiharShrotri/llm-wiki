"""Raw ingest: copy files into raw/, parse them, register in the state DB.

This is the "plumbing" layer. No LLM calls happen here — that's Stage 3's
`wiki ingest` command. This module just ensures every file in `raw/` has
a corresponding row in the `sources` table with a content hash so dedupe
and provenance work downstream.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable

from . import config as cfg
from . import db
from . import parsers


class AddResult(str, Enum):
    ADDED = "added"
    DEDUPED = "deduped"
    SKIPPED_EMPTY = "skipped_empty"
    SKIPPED_UNSUPPORTED = "skipped_unsupported"
    ERROR = "error"


@dataclass
class AddOutcome:
    """Result of attempting to add a single file."""

    result: AddResult
    source_path: Path              # Where the file ended up (or original on error)
    relpath: str                   # Relative to project root
    title: str | None = None
    file_type: str | None = None
    bytes: int = 0
    word_count: int = 0
    content_hash: str | None = None
    source_id: int | None = None   # Row ID in sources table
    message: str = ""              # Human-friendly explanation

    @property
    def ok(self) -> bool:
        return self.result == AddResult.ADDED

    @property
    def is_warning(self) -> bool:
        return self.result in {
            AddResult.DEDUPED,
            AddResult.SKIPPED_EMPTY,
            AddResult.SKIPPED_UNSUPPORTED,
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _unique_destination(dest_dir: Path, filename: str) -> Path:
    """If `dest_dir/filename` already exists, append `-1`, `-2`, ... before the
    extension until we find an unused path.
    """
    target = dest_dir / filename
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    counter = 1
    while True:
        candidate = dest_dir / f"{stem}-{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _is_inside_raw(path: Path, raw_dir: Path) -> bool:
    """True if `path` is already inside the project's raw/ directory."""
    try:
        path.resolve().relative_to(raw_dir.resolve())
        return True
    except ValueError:
        return False


def add_file(
    paths: cfg.WikiPaths,
    source: Path,
    *,
    copy: bool = True,
) -> AddOutcome:
    """Add a single file to the wiki's raw/ directory and register it.

    Args:
        paths: Resolved wiki project paths.
        source: The file to add (can be anywhere on disk).
        copy: If True, copy the file into raw/. If False (used when the file
              is already inside raw/), skip the copy and just parse + register.

    Returns:
        AddOutcome describing what happened. Check `.result` for the variant.
    """
    source = source.expanduser().resolve()

    # Guard: file must exist
    if not source.exists() or not source.is_file():
        return AddOutcome(
            result=AddResult.ERROR,
            source_path=source,
            relpath=str(source),
            message=f"File not found: {source}",
        )

    # Guard: file type must be supported
    if not parsers.is_supported(source):
        return AddOutcome(
            result=AddResult.SKIPPED_UNSUPPORTED,
            source_path=source,
            relpath=str(source),
            message=f"Unsupported file type: {source.suffix or '(no extension)'}",
        )

    # Determine the final destination in raw/
    if copy and not _is_inside_raw(source, paths.raw):
        paths.raw.mkdir(parents=True, exist_ok=True)
        dest = _unique_destination(paths.raw, source.name)
        try:
            shutil.copy2(source, dest)
        except OSError as e:
            return AddOutcome(
                result=AddResult.ERROR,
                source_path=source,
                relpath=str(source),
                message=f"Failed to copy into raw/: {e}",
            )
        final_path = dest
    else:
        final_path = source

    # Parse
    try:
        parsed = parsers.parse(final_path)
    except parsers.ParserError as e:
        return AddOutcome(
            result=AddResult.ERROR,
            source_path=final_path,
            relpath=str(final_path.relative_to(paths.root)) if _is_inside_raw(final_path, paths.raw) else str(final_path),
            message=f"Parse failed: {e}",
        )

    try:
        relpath = str(final_path.relative_to(paths.root))
    except ValueError:
        relpath = str(final_path)

    # Dedupe by content hash
    with db.connect(paths.state_db) as conn:
        existing = conn.execute(
            "SELECT id, relpath FROM sources WHERE content_hash = ?",
            (parsed.content_hash,),
        ).fetchone()

        if existing is not None:
            # If we just copied this in, remove the duplicate copy.
            if copy and final_path.exists() and final_path != source:
                try:
                    final_path.unlink()
                except OSError:
                    pass
            return AddOutcome(
                result=AddResult.DEDUPED,
                source_path=final_path,
                relpath=relpath,
                title=parsed.title,
                file_type=parsed.file_type,
                bytes=parsed.bytes,
                word_count=parsed.word_count,
                content_hash=parsed.content_hash,
                source_id=existing["id"],
                message=f"Already tracked as #{existing['id']}: {existing['relpath']}",
            )

        # Warn on near-empty extraction (likely scanned PDF)
        if parsed.is_empty:
            # We still register it so the user sees it in `wiki sources list`
            # with SKIPPED_EMPTY status, but flag it loudly.
            status = "error"
            message = (
                f"Extracted only {parsed.word_count} words — likely a scanned "
                f"PDF or empty file. OCR not yet supported."
            )
            result_kind = AddResult.SKIPPED_EMPTY
        else:
            status = "pending"
            message = f"Added as #{'?'}: {parsed.title}"
            result_kind = AddResult.ADDED

        cur = conn.execute(
            """
            INSERT INTO sources (relpath, content_hash, file_type, bytes, added_at, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                relpath,
                parsed.content_hash,
                parsed.file_type,
                parsed.bytes,
                _now_iso(),
                status,
            ),
        )
        source_id = cur.lastrowid
        if "#'?'" in message:
            message = message.replace("#'?'", f"#{source_id}")

    return AddOutcome(
        result=result_kind,
        source_path=final_path,
        relpath=relpath,
        title=parsed.title,
        file_type=parsed.file_type,
        bytes=parsed.bytes,
        word_count=parsed.word_count,
        content_hash=parsed.content_hash,
        source_id=source_id,
        message=message,
    )


def iter_addable_files(root: Path, recursive: bool) -> Iterable[Path]:
    """Yield every supported file at or under `root`.

    If `root` is a file, yield it (if supported). If it's a directory, walk
    it (recursively if requested) and yield every supported file found.
    """
    root = root.expanduser().resolve()
    if not root.exists():
        return
    if root.is_file():
        if parsers.is_supported(root):
            yield root
        return
    if root.is_dir():
        iterator = root.rglob("*") if recursive else root.iterdir()
        for child in iterator:
            if child.is_file() and not child.name.startswith(".") and parsers.is_supported(child):
                yield child


def list_sources(
    paths: cfg.WikiPaths, status_filter: str | None = None
) -> list[dict]:
    """Return all tracked sources as a list of dicts, ordered by id."""
    query = "SELECT * FROM sources"
    params: tuple = ()
    if status_filter:
        query += " WHERE status = ?"
        params = (status_filter,)
    query += " ORDER BY id ASC"

    with db.connect(paths.state_db) as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def get_source(paths: cfg.WikiPaths, source_id: int) -> dict | None:
    """Fetch a single source row by id."""
    with db.connect(paths.state_db) as conn:
        row = conn.execute(
            "SELECT * FROM sources WHERE id = ?", (source_id,)
        ).fetchone()
        return dict(row) if row else None


def mark_source_pending(
    paths: cfg.WikiPaths, source_id: int
) -> tuple[bool, str]:
    """Reset a source's status to 'pending' so it can be re-ingested.

    Doesn't touch any wiki pages that were created from this source —
    re-ingestion will use the merge path on existing pages, or create
    new ones with new content. Useful when:

    - You blocked an earlier ingest mid-flight
    - You want Qwen to re-process the source with updated prompts
    - The previous extraction was poor quality and you want to try again
    """
    row = get_source(paths, source_id)
    if row is None:
        return False, f"No source with id {source_id}"

    file_path = paths.root / row["relpath"]
    if not file_path.exists():
        return False, f"Source file no longer on disk: {row['relpath']}"

    with db.connect(paths.state_db) as conn:
        conn.execute(
            "UPDATE sources SET status = 'pending', last_ingested = NULL WHERE id = ?",
            (source_id,),
        )
        conn.commit()

    return True, f"Marked #{source_id} ({row['relpath']}) as pending"


def remove_source(
    paths: cfg.WikiPaths, source_id: int, delete_file: bool = True
) -> tuple[bool, str]:
    """Remove a source from tracking. Optionally delete the file from raw/.

    Returns (success, message).
    """
    row = get_source(paths, source_id)
    if row is None:
        return False, f"No source with id {source_id}"

    file_path = paths.root / row["relpath"]

    with db.connect(paths.state_db) as conn:
        # Cascade: remove source_pages rows first (no FK CASCADE by default)
        conn.execute("DELETE FROM source_pages WHERE source_id = ?", (source_id,))
        conn.execute("DELETE FROM ingest_runs WHERE source_id = ?", (source_id,))
        conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))

    deleted_file = False
    if delete_file and file_path.exists() and _is_inside_raw(file_path, paths.raw):
        try:
            file_path.unlink()
            deleted_file = True
        except OSError as e:
            return True, f"Removed #{source_id} from DB but failed to delete file: {e}"

    msg = f"Removed #{source_id} ({row['relpath']})"
    if deleted_file:
        msg += " — file deleted from raw/"
    elif delete_file:
        msg += " — file was outside raw/, left in place"
    return True, msg
