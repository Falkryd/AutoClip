"""Command line for the transcribe stage: python -m src.transcribe --help"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from src.common.config import ConfigError, load_config
from src.common.files import read_json
from src.common.lock import LockHeldError, file_lock
from src.common.logs import setup_logging
from src.common.paths import Target, lock_path, resolve_target, state_db
from src.common.state import StageStatus, State
from src.common.timecode import format_timecode
from src.transcribe.core import CaptionsNotReady, TranscribeError, run_transcribe

STAGE = "transcribe"
EXIT_WAITING = 3  # captions not ready yet: the scheduled pipeline retries on its next run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.transcribe",
        description="Write transcript.json (sentences with word timings, plus sound tags such "
        "as [tertawa]) and transcript.srt. Uses YouTube's automatic captions when they exist, "
        "otherwise Whisper on the GPU. Skips if transcript.json already exists. Exit code 3 "
        "means YouTube's captions aren't ready yet: run again later.",
    )
    parser.add_argument(
        "target",
        help="video ID or URL (uses work/<id>/), or a folder such as samples/<id>_<start>-<end>",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "youtube", "whisper"],
        help="override transcribe.source from config.yaml",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="if YouTube has no captions yet, use Whisper now instead of waiting",
    )
    parser.add_argument("--force", action="store_true", help="transcribe again")
    parser.add_argument("--config", type=Path, help="config file (default: config.yaml)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    try:
        target = resolve_target(cfg, args.target)
    except ValueError as exc:
        parser.error(str(exc))
    log = setup_logging(cfg, STAGE)
    a = target.artifacts
    if a.transcript.is_file() and not args.force:
        log.info("Already transcribed: %s (use --force to redo)", a.transcript)
        _print_summary(target)
        return 0

    state = State(state_db(cfg)) if target.is_pipeline_dir and target.video_id else None
    video_id = target.video_id or ""
    try:
        with file_lock(lock_path(cfg, f"{STAGE}-{a.root.name}")):
            if state:
                state.mark_stage(video_id, STAGE, StageStatus.RUNNING)
            run_transcribe(cfg, target, source=args.source, wait=not args.no_wait)
    except LockHeldError as exc:
        log.warning("%s; skipping this run", exc)
        return 0
    except CaptionsNotReady as exc:
        log.info("%s", exc)
        if state:
            state.mark_stage(video_id, STAGE, StageStatus.WAITING, error=str(exc))
        return EXIT_WAITING
    except TranscribeError as exc:
        log.error("%s", exc)
        _mark_failed(state, video_id, exc)
        return 1
    except Exception as exc:
        log.exception("Transcription failed")
        _mark_failed(state, video_id, exc)
        return 1
    if state:
        state.mark_stage(video_id, STAGE, StageStatus.DONE)
    _print_summary(target)
    return 0


def _mark_failed(state: State | None, video_id: str, exc: BaseException) -> None:
    if state:
        state.mark_stage(video_id, STAGE, StageStatus.FAILED, error=str(exc)[:500])


def _print_summary(target: Target) -> None:
    a = target.artifacts
    doc = read_json(a.transcript)
    engine = doc["engine"]
    if doc["source"] == "youtube":
        how = f"YouTube captions ({engine['captions_track']})"
    else:
        how = (
            f"Whisper {engine['model']} on {engine['device']}, "
            f"{format_timecode(engine['elapsed_s'])} for {format_timecode(doc['duration_s'])}"
        )
    tags: dict[str, int] = {}
    for sound in doc["sounds"]:
        tags[sound["tag"]] = tags.get(sound["tag"], 0) + 1
    print(f"Transcript in {a.root}")
    print(f"  source: {how}")
    print(f"  {doc['segment_count']} segments kept, {doc['dropped_count']} dropped")
    if tags:
        print(
            "  sound tags: "
            + ", ".join(f"{t} x{n}" for t, n in sorted(tags.items(), key=lambda x: -x[1]))
        )
    print(f"  {a.transcript.name}, {a.transcript_srt.name}")
