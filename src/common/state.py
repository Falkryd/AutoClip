"""SQLite record of which streams were seen and which stages finished."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path


class StreamStatus(StrEnum):
    WAITING = "waiting"  # live, upcoming, or VOD still processing: recheck next run
    QUEUED = "queued"  # VOD ready: the pipeline should process it
    SKIPPED = "skipped"  # not a stream, too old, or gave up waiting
    DONE = "done"  # every stage finished
    FAILED = "failed"  # a stage failed: rerun to resume


class StageStatus(StrEnum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


_SCHEMA = """
PRAGMA journal_mode = WAL;
CREATE TABLE IF NOT EXISTS streams (
    video_id      TEXT PRIMARY KEY,
    title         TEXT NOT NULL DEFAULT '',
    status        TEXT NOT NULL,
    live_status   TEXT,
    ended_at      TEXT,
    note          TEXT NOT NULL DEFAULT '',
    discovered_at TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS stage_runs (
    video_id    TEXT NOT NULL,
    stage       TEXT NOT NULL,
    status      TEXT NOT NULL,
    started_at  TEXT,
    finished_at TEXT,
    error       TEXT,
    PRIMARY KEY (video_id, stage)
);
"""


def utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime | None) -> str | None:
    return value.astimezone(UTC).isoformat(timespec="seconds") if value else None


def _parse(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


@dataclass(frozen=True)
class StreamRecord:
    video_id: str
    title: str
    status: StreamStatus
    live_status: str | None
    ended_at: datetime | None
    note: str
    discovered_at: datetime
    updated_at: datetime


class State:
    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            with conn:  # commit on success, roll back on error
                yield conn
        finally:
            conn.close()

    def get_stream(self, video_id: str) -> StreamRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM streams WHERE video_id = ?", (video_id,)).fetchone()
        return _record(row) if row else None

    def streams(self, status: StreamStatus | None = None) -> list[StreamRecord]:
        query = "SELECT * FROM streams"
        params: tuple[str, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            params = (status.value,)
        with self._connect() as conn:
            rows = conn.execute(query + " ORDER BY discovered_at DESC", params).fetchall()
        return [_record(row) for row in rows]

    def upsert_stream(
        self,
        video_id: str,
        *,
        title: str,
        status: StreamStatus,
        live_status: str | None,
        ended_at: datetime | None,
        note: str,
        now: datetime,
    ) -> None:
        """Insert or update a stream. Keeps discovered_at and any ended_at already known."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO streams
                    (video_id, title, status, live_status, ended_at, note,
                     discovered_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(video_id) DO UPDATE SET
                    title = CASE WHEN excluded.title != ''
                                 THEN excluded.title ELSE streams.title END,
                    status = excluded.status,
                    live_status = COALESCE(excluded.live_status, streams.live_status),
                    ended_at = COALESCE(excluded.ended_at, streams.ended_at),
                    note = excluded.note,
                    updated_at = excluded.updated_at
                """,
                (
                    video_id,
                    title,
                    status.value,
                    live_status,
                    _iso(ended_at),
                    note,
                    _iso(now),
                    _iso(now),
                ),
            )

    def stage_status(self, video_id: str, stage: str) -> StageStatus | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM stage_runs WHERE video_id = ? AND stage = ?", (video_id, stage)
            ).fetchone()
        return StageStatus(row["status"]) if row else None

    def mark_stage(
        self, video_id: str, stage: str, status: StageStatus, error: str | None = None
    ) -> None:
        now = _iso(utcnow())
        started = now if status == StageStatus.RUNNING else None
        finished = None if status == StageStatus.RUNNING else now
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO stage_runs (video_id, stage, status, started_at, finished_at, error)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(video_id, stage) DO UPDATE SET
                    status = excluded.status,
                    started_at = COALESCE(excluded.started_at, stage_runs.started_at),
                    finished_at = excluded.finished_at,
                    error = excluded.error
                """,
                (video_id, stage, status.value, started, finished, error),
            )


def _record(row: sqlite3.Row) -> StreamRecord:
    return StreamRecord(
        video_id=row["video_id"],
        title=row["title"],
        status=StreamStatus(row["status"]),
        live_status=row["live_status"],
        ended_at=_parse(row["ended_at"]),
        note=row["note"],
        discovered_at=datetime.fromisoformat(row["discovered_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )
