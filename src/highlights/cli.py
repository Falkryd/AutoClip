"""Command line for the highlights stage: python -m src.highlights --help"""

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
from src.highlights.core import HighlightsError, run_highlights
from src.highlights.llm import LlmError

STAGE = "highlights"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.highlights",
        description="Pick the best 30-60 s moments: score the stream on laughs ([tertawa] tags), "
        "loudness spikes and chat, let the local LLM (Ollama) judge the shortlist, and write "
        "highlights.json. Skips if highlights.json already exists.",
    )
    parser.add_argument(
        "target",
        help="video ID or URL (uses work/<id>/), or a folder such as samples/<id>_<start>-<end>",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="score with laughs, loudness and chat only (quick, for tuning the weights)",
    )
    parser.add_argument("--force", action="store_true", help="pick again")
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
    if a.highlights.is_file() and not args.force:
        log.info("Already picked: %s (use --force to redo)", a.highlights)
        _print_summary(target)
        return 0

    state = State(state_db(cfg)) if target.is_pipeline_dir and target.video_id else None
    video_id = target.video_id or ""
    try:
        with file_lock(lock_path(cfg, f"{STAGE}-{a.root.name}")):
            if state:
                state.mark_stage(video_id, STAGE, StageStatus.RUNNING)
            run_highlights(cfg, target, use_llm=not args.no_llm)
    except LockHeldError as exc:
        log.warning("%s; skipping this run", exc)
        return 0
    except (HighlightsError, LlmError) as exc:
        log.error("%s", exc)
        _mark_failed(state, video_id, exc)
        return 1
    except Exception as exc:
        log.exception("Picking highlights failed")
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
    doc = read_json(target.artifacts.highlights)
    signals = "; ".join(f"{name}: {status}" for name, status in doc["signals"].items())
    print(f"Highlights in {target.artifacts.highlights}")
    print(f"  signals: {signals}")
    if not doc["clips"]:
        print("  No clip was good enough. See 'candidates' in highlights.json.")
    for clip in doc["clips"]:
        llm = f"LLM {clip['llm_score']}/10, " if clip["llm_score"] is not None else ""
        print(
            f"\n  #{clip['rank']}  {format_timecode(clip['start'])}-{format_timecode(clip['end'])} "
            f"({clip['duration_s']:.0f} s; stream {clip['stream_start']})  "
            f"score {clip['score']:.2f} "
            f"({llm}{clip['laughs']} laughs)"
        )
        print(f"      {clip['title']}")
        print(f"      {clip['reason']}")
        for flag in clip["flags"]:
            print(f"      ! {flag}")
    print(f"\n  {len(doc['candidates'])} candidates judged; all are in highlights.json")
