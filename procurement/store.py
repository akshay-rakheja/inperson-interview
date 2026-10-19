"""SQLite persistence: cases, immutable source versions, analysis runs, review overlays, drafts.

JSON payloads are stored as text. A connection is opened per operation, so no transaction is
ever held open across a model request.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .models import SourceRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sources (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL UNIQUE,
    case_id TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    metadata TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS analysis_runs (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    case_id TEXT NOT NULL,
    status TEXT NOT NULL,
    input_hash TEXT,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reviews (
    case_id TEXT NOT NULL,
    requirement_key TEXT NOT NULL,
    fields TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (case_id, requirement_key)
);
CREATE TABLE IF NOT EXISTS drafts (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    draft_id TEXT NOT NULL UNIQUE,
    case_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    stale INTEGER NOT NULL DEFAULT 0,
    stale_reason TEXT,
    created_at TEXT NOT NULL
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- sources ------------------------------------------------------------

    def save_sources(self, case_id: str, sources: list[SourceRecord]) -> None:
        """Idempotent: an existing source_id (same case, path and content) is not duplicated."""
        with self._conn() as conn:
            conn.execute("INSERT OR IGNORE INTO cases (case_id, created_at) VALUES (?, ?)", (case_id, now_iso()))
            for src in sources:
                if src.case_id != case_id:
                    raise ValueError(f"Source {src.source_id} belongs to {src.case_id}, not {case_id}")
                conn.execute(
                    "INSERT OR IGNORE INTO sources (source_id, case_id, relative_path, content_sha256, kind, text,"
                    " metadata, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (src.source_id, case_id, src.relative_path, src.content_sha256, src.kind, src.text,
                     json.dumps(src.metadata), now_iso()),
                )

    @staticmethod
    def _row_to_source(row: sqlite3.Row) -> SourceRecord:
        return SourceRecord(
            source_id=row["source_id"], case_id=row["case_id"], relative_path=row["relative_path"],
            content_sha256=row["content_sha256"], text=row["text"], kind=row["kind"],
            metadata=json.loads(row["metadata"]),
        )

    def get_sources(self, case_id: str) -> list[SourceRecord]:
        """Current sources: the latest version for each relative path in this case."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT s.* FROM sources s JOIN (SELECT relative_path, MAX(seq) AS seq FROM sources"
                " WHERE case_id = ? GROUP BY relative_path) latest ON s.seq = latest.seq"
                " ORDER BY s.relative_path",
                (case_id,),
            ).fetchall()
        return [self._row_to_source(r) for r in rows]

    def get_source_versions(self, case_id: str) -> list[SourceRecord]:
        """Every retained version, oldest first (history view)."""
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM sources WHERE case_id = ? ORDER BY seq", (case_id,)).fetchall()
        return [self._row_to_source(r) for r in rows]

    def list_cases(self) -> list[str]:
        with self._conn() as conn:
            return [r["case_id"] for r in conn.execute("SELECT case_id FROM cases ORDER BY case_id")]

    # -- analysis runs ------------------------------------------------------

    def save_run(self, case_id: str, payload: dict) -> str:
        run_id = payload.get("run_id") or f"run-{uuid.uuid4().hex[:12]}"
        payload = {**payload, "run_id": run_id, "case_id": case_id}
        payload.setdefault("created_at", now_iso())
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO analysis_runs (run_id, case_id, status, input_hash, payload, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, case_id, payload["status"], payload.get("input_hash"), json.dumps(payload),
                 payload["created_at"]),
            )
        return run_id

    def latest_run(self, case_id: str) -> dict | None:
        """Latest *successful* run. Failures never replace it."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT payload FROM analysis_runs WHERE case_id = ? AND status = 'success'"
                " ORDER BY seq DESC LIMIT 1",
                (case_id,),
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def list_runs(self, case_id: str) -> list[dict]:
        """All runs (success and failure), newest first."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT payload FROM analysis_runs WHERE case_id = ? ORDER BY seq DESC", (case_id,)
            ).fetchall()
        return [json.loads(r["payload"]) for r in rows]

    def get_run(self, run_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT payload FROM analysis_runs WHERE run_id = ?", (run_id,)).fetchone()
        return json.loads(row["payload"]) if row else None

    def find_successful_run(self, case_id: str, input_hash: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT payload FROM analysis_runs WHERE case_id = ? AND input_hash = ? AND status = 'success'"
                " ORDER BY seq DESC LIMIT 1",
                (case_id, input_hash),
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    # -- review overlays ----------------------------------------------------

    def save_review(self, case_id: str, requirement_key: str, fields: dict) -> None:
        """Merge fields into the overlay; earlier values are kept in `history`."""
        if not requirement_key.startswith(f"{case_id}:"):
            raise ValueError(f"Requirement key {requirement_key!r} is not scoped to case {case_id!r}")
        with self._conn() as conn:
            row = conn.execute(
                "SELECT fields FROM reviews WHERE case_id = ? AND requirement_key = ?", (case_id, requirement_key)
            ).fetchone()
            current = json.loads(row["fields"]) if row else {}
            history = current.pop("history", [])
            if current:
                history.append({**current, "superseded_at": now_iso()})
            merged = {**current, **fields, "history": history}
            conn.execute(
                "INSERT OR REPLACE INTO reviews (case_id, requirement_key, fields, updated_at) VALUES (?, ?, ?, ?)",
                (case_id, requirement_key, json.dumps(merged), now_iso()),
            )

    def get_reviews(self, case_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT requirement_key, fields, updated_at FROM reviews WHERE case_id = ? ORDER BY requirement_key",
                (case_id,),
            ).fetchall()
        return [
            {"case_id": case_id, "requirement_key": r["requirement_key"], "updated_at": r["updated_at"],
             **json.loads(r["fields"])}
            for r in rows
        ]

    # -- drafts -------------------------------------------------------------

    def save_draft(self, case_id: str, run_id: str, payload: dict) -> str:
        draft_id = f"draft-{uuid.uuid4().hex[:12]}"
        payload = {**payload, "draft_id": draft_id, "case_id": case_id, "basis_run_id": run_id}
        payload.setdefault("created_at", now_iso())
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO drafts (draft_id, case_id, run_id, payload, created_at) VALUES (?, ?, ?, ?, ?)",
                (draft_id, case_id, run_id, json.dumps(payload), payload["created_at"]),
            )
        return draft_id

    def update_draft(self, draft_id: str, fields: dict) -> None:
        """Merge human edits / local review into a draft; generated content is never overwritten."""
        with self._conn() as conn:
            row = conn.execute("SELECT payload FROM drafts WHERE draft_id = ?", (draft_id,)).fetchone()
            if row is None:
                raise KeyError(draft_id)
            payload = json.loads(row["payload"])
            if "generated" in fields:
                raise ValueError("Generated draft content is immutable")
            payload.update(fields)
            conn.execute("UPDATE drafts SET payload = ? WHERE draft_id = ?", (json.dumps(payload), draft_id))

    def get_drafts(self, case_id: str) -> list[dict]:
        """Newest first, each with its stale flag and reason."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT payload, stale, stale_reason FROM drafts WHERE case_id = ? ORDER BY seq DESC", (case_id,)
            ).fetchall()
        return [{**json.loads(r["payload"]), "stale": bool(r["stale"]), "stale_reason": r["stale_reason"]}
                for r in rows]

    def invalidate_drafts(self, case_id: str, reason: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE drafts SET stale = 1, stale_reason = ? WHERE case_id = ? AND stale = 0", (reason, case_id)
            )
