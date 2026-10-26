"""Decide which discovered videos are finished streams ready to clip."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from src.common.config import TriggerConfig
from src.common.state import State, StreamRecord, StreamStatus
from src.common.ytdl import MetadataError, VideoMeta, video_url
from src.trigger.discover import Candidate

_log = logging.getLogger("autoclip.trigger")

_WAIT_NOTES = {
    "is_live": "still live",
    "is_upcoming": "upcoming",
    "post_live": "stream ended, VOD still processing",
}

MetaFetcher = Callable[[str], VideoMeta]


class NotReadyError(RuntimeError):
    """A video asked for by hand has no finished VOD yet."""


@dataclass(frozen=True)
class Decision:
    video_id: str
    title: str
    status: StreamStatus
    note: str
    live_status: str | None = None
    ended_at: datetime | None = None
    meta: VideoMeta | None = None


def run_trigger(
    cfg: TriggerConfig,
    state: State,
    candidates: Iterable[Candidate],
    fetch_meta: MetaFetcher,
    now: datetime,
    *,
    dry_run: bool = False,
) -> list[Decision]:
    """Evaluate new and still-waiting videos and save the outcome. Settled ones are left alone."""
    decisions = []
    for candidate, record in _pending(state, candidates):
        decision = evaluate(candidate, record, fetch_meta, cfg, now)
        decisions.append(decision)
        _log.info("%s -> %s: %s", decision.video_id, decision.status, decision.note)
        if not dry_run:
            _save(state, decision, now)
    return decisions


def queue_manually(
    state: State, video_id: str, fetch_meta: MetaFetcher, now: datetime, *, dry_run: bool = False
) -> Decision:
    """Queue one video regardless of the lookback window, as long as its VOD is ready."""
    meta = fetch_meta(video_id)
    if meta.live_status in _WAIT_NOTES:
        raise NotReadyError(f"{video_id} is not ready: {_WAIT_NOTES[meta.live_status]}")
    note = "queued by hand"
    if meta.live_status != "was_live":
        note += f"; not a past live stream ({meta.live_status}), so it has no chat replay"
    decision = Decision(
        video_id, meta.title, StreamStatus.QUEUED, note, meta.live_status, meta.ended_at, meta
    )
    _log.info("%s -> %s: %s", video_id, decision.status, note)
    if not dry_run:
        _save(state, decision, now)
    return decision


def evaluate(
    candidate: Candidate,
    record: StreamRecord | None,
    fetch_meta: MetaFetcher,
    cfg: TriggerConfig,
    now: datetime,
) -> Decision:
    waited = now - record.discovered_at if record else timedelta(0)

    def waiting(
        note: str, live_status: str | None = None, title: str = candidate.title
    ) -> Decision:
        if waited > timedelta(hours=cfg.give_up_after_hours):
            hours = waited.total_seconds() / 3600
            note = f"gave up after waiting {hours:.0f} h (last seen: {note})"
            return Decision(candidate.video_id, title, StreamStatus.SKIPPED, note, live_status)
        return Decision(candidate.video_id, title, StreamStatus.WAITING, note, live_status)

    if candidate.is_stream is False:
        return Decision(
            candidate.video_id, candidate.title, StreamStatus.SKIPPED, "not a live stream"
        )
    if candidate.is_stream and candidate.ended_at is None:
        return waiting("live or upcoming")

    try:
        meta = fetch_meta(candidate.video_id)
    except MetadataError as exc:
        _log.warning("%s: metadata unavailable: %s", candidate.video_id, exc)
        return waiting(f"metadata unavailable: {exc}")

    title = meta.title or candidate.title
    if meta.live_status in _WAIT_NOTES:
        return waiting(_WAIT_NOTES[meta.live_status], meta.live_status, title)
    if meta.live_status != "was_live":
        note = f"not a live stream ({meta.live_status or 'unknown'})"
        return Decision(candidate.video_id, title, StreamStatus.SKIPPED, note, meta.live_status)

    ended_at = candidate.ended_at or meta.ended_at
    # The lookback only applies to streams first seen already finished, so the first run
    # doesn't backfill old streams. Streams we watched go live and end are always queued.
    if record is None:
        reference = ended_at or meta.started_at
        if reference is None or now - reference > timedelta(hours=cfg.lookback_hours):
            note = "finished before the lookback window; queue it by hand with --queue"
            return Decision(
                candidate.video_id, title, StreamStatus.SKIPPED, note, meta.live_status, ended_at
            )
    return Decision(
        candidate.video_id,
        title,
        StreamStatus.QUEUED,
        "VOD ready",
        meta.live_status,
        ended_at,
        meta,
    )


def stream_info(decision: Decision, now: datetime) -> dict[str, Any]:
    """Contents of work/<video_id>/stream.json, read by later stages."""
    meta = decision.meta
    return {
        "video_id": decision.video_id,
        "title": decision.title,
        "url": video_url(decision.video_id),
        "live_status": decision.live_status,
        "started_at": meta.started_at.isoformat() if meta and meta.started_at else None,
        "ended_at": decision.ended_at.isoformat() if decision.ended_at else None,
        "duration_s": meta.duration_s if meta else None,
        "queued_at": now.isoformat(timespec="seconds"),
    }


def _pending(
    state: State, candidates: Iterable[Candidate]
) -> list[tuple[Candidate, StreamRecord | None]]:
    seen: set[str] = set()
    pending = []
    for candidate in candidates:
        if candidate.video_id in seen:
            continue
        seen.add(candidate.video_id)
        record = state.get_stream(candidate.video_id)
        if record is None or record.status == StreamStatus.WAITING:
            pending.append((candidate, record))
    # Waiting streams that dropped out of the feed still need rechecking.
    for record in state.streams(StreamStatus.WAITING):
        if record.video_id not in seen:
            pending.append((Candidate(record.video_id, record.title), record))
    return pending


def _save(state: State, decision: Decision, now: datetime) -> None:
    state.upsert_stream(
        decision.video_id,
        title=decision.title,
        status=decision.status,
        live_status=decision.live_status,
        ended_at=decision.ended_at,
        note=decision.note,
        now=now,
    )
