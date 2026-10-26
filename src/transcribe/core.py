"""Pick the transcript source and write transcript.json and transcript.srt.

auto: YouTube's automatic captions when they exist. While they don't, the stage asks to be
retried, until captions_wait_hours after the download finished; then it uses Whisper.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from src.common.config import AppConfig, TranscribeConfig
from src.common.ffmpeg import probe
from src.common.files import read_json, write_json_atomic, write_text_atomic
from src.common.paths import Artifacts, Target
from src.common.state import utcnow
from src.common.ytdl import ytdl_env
from src.transcribe.segments import (
    remove_hallucinations,
    resegment,
    split_segments,
    to_srt,
)
from src.transcribe.whisper import WhisperError, transcribe_audio
from src.transcribe.youtube import CaptionsError, clip_to_window, fetch_captions, parse_json3

_log = logging.getLogger("autoclip.transcribe")


class TranscribeError(RuntimeError):
    """Nothing to transcribe, or the chosen source failed."""


class CaptionsNotReady(RuntimeError):
    """YouTube hasn't made captions yet and the wait limit hasn't passed: retry later."""


def run_transcribe(
    cfg: AppConfig,
    target: Target,
    *,
    source: str | None = None,
    wait: bool = True,
    now: datetime | None = None,
) -> dict[str, Any]:
    tc = cfg.transcribe
    source = source or tc.source
    a = target.artifacts
    manifest = read_json(a.download_manifest) if a.download_manifest.is_file() else {}

    if source != "whisper":
        document = _from_youtube(cfg, target, manifest)
        if document is not None:
            return _write(a, document)
        if source == "youtube":
            raise TranscribeError("YouTube has no captions in this language for this video")
        deadline = captions_deadline(manifest, tc.captions_wait_hours)
        if wait and deadline is not None and (now or utcnow()) < deadline:
            raise CaptionsNotReady(
                f"No YouTube captions yet; retrying until {deadline:%Y-%m-%d %H:%M} UTC, "
                "then Whisper takes over"
            )
        _log.info("Using Whisper instead of YouTube captions")
    return _write(a, _from_whisper(tc, target, manifest))


def captions_deadline(manifest: dict[str, Any], wait_hours: float) -> datetime | None:
    completed = manifest.get("completed_at")
    return datetime.fromisoformat(completed) + timedelta(hours=wait_hours) if completed else None


def _from_youtube(
    cfg: AppConfig, target: Target, manifest: dict[str, Any]
) -> dict[str, Any] | None:
    a = target.artifacts
    if target.video_id is None:
        _log.warning("No video ID known for %s; can't look up YouTube captions", a.root)
        return None
    try:
        track = fetch_captions(
            target.video_id, a.captions_raw, cfg.transcribe.language, ytdl_env(cfg)
        )
    except CaptionsError as exc:
        _log.warning("%s", exc)
        return None
    if track is None:
        return None
    words, sounds = parse_json3(read_json(a.captions_raw))
    offset, duration = _offset(manifest), _duration(a, manifest)
    words = clip_to_window(words, "start", offset, duration)
    sounds = clip_to_window(sounds, "t", offset, duration)
    block = {"start": 0.0, "end": duration, "text": "", "avg_logprob": None,
             "no_speech_prob": None, "compression_ratio": None, "words": words}  # fmt: skip
    kept, dropped = _split([block], cfg.transcribe)
    return _document(target, "youtube", cfg.transcribe.language, duration, offset,
                     {"captions_track": track}, kept, dropped, sounds)  # fmt: skip


def _from_whisper(tc: TranscribeConfig, target: Target, manifest: dict[str, Any]) -> dict[str, Any]:
    a = target.artifacts
    if not a.audio.is_file():
        raise TranscribeError(f"{a.audio} not found; run the download stage first")
    try:
        result = transcribe_audio(tc, a.audio)
    except WhisperError as exc:
        raise TranscribeError(str(exc)) from exc
    kept, dropped = _split(result.records, tc)
    return _document(target, "whisper", result.language, result.duration_s, _offset(manifest),
                     result.engine, kept, dropped, sounds=[])  # fmt: skip


def _split(
    blocks: list[dict[str, Any]], tc: TranscribeConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pieces: list[dict[str, Any]] = []
    hallucinated: list[dict[str, Any]] = []
    for block in blocks:
        cleaned, removed = remove_hallucinations(block)
        hallucinated.extend(removed)
        pieces.extend(
            resegment(cleaned, max_s=tc.max_segment_seconds, pause_s=tc.split_pause_seconds)
        )
    kept, dropped = split_segments(pieces, tc)
    return kept, sorted(dropped + hallucinated, key=lambda d: d["start"])


def _offset(manifest: dict[str, Any]) -> float:
    """Where vod.mp4 starts in the stream (0 for a full VOD)."""
    if "offset_s" in manifest:
        return float(manifest["offset_s"])
    section = manifest.get("section") or {}
    return float(section.get("start_s", 0.0))


def _duration(a: Artifacts, manifest: dict[str, Any]) -> float:
    if manifest.get("vod"):
        return float(manifest["vod"]["duration_s"])
    source = a.audio if a.audio.is_file() else a.vod
    if not source.is_file():
        raise TranscribeError(f"No vod.mp4 or audio.wav in {a.root}; run the download stage first")
    return probe(source).duration_s


def _document(
    target: Target,
    source: str,
    language: str | None,
    duration_s: float,
    offset_s: float,
    engine: dict[str, Any],
    kept: list[dict[str, Any]],
    dropped: list[dict[str, Any]],
    sounds: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "video_id": target.video_id,
        "source": source,  # youtube or whisper
        "language": language,
        "duration_s": duration_s,
        "offset_s": offset_s,
        "engine": engine,
        "created_at": utcnow().isoformat(timespec="seconds"),
        "segment_count": len(kept),
        "dropped_count": len(dropped),
        "sound_count": len(sounds),
        "segments": kept,
        "dropped": dropped,
        "sounds": sounds,  # [tertawa], [musik] ... from YouTube; empty for Whisper
    }


def _write(a: Artifacts, document: dict[str, Any]) -> dict[str, Any]:
    _log.info(
        "Transcript from %s: %d segments kept, %d dropped, %d sound tags",
        document["source"], document["segment_count"], document["dropped_count"],
        document["sound_count"],
    )  # fmt: skip
    write_text_atomic(a.transcript_srt, to_srt(document["segments"]))
    write_json_atomic(a.transcript, document)  # written last: marks the stage done
    return document
