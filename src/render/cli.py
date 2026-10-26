"""Command line for the render stage: python -m src.render --help"""

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
from src.render.core import RenderError, render_preview, run_render

STAGE = "render"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.render",
        description="Render each clip in highlights.json to clips/NN.mp4: 1080x1920, gameplay "
        "on top, avatar below, captions burned in, audio at YouTube loudness, plus a "
        "thumbnail. Skips clips that are already rendered.",
    )
    parser.add_argument(
        "target",
        help="video ID or URL (uses work/<id>/), or a folder such as samples/<id>_<start>-<end>",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="only write clips/layout_preview.png (one frame) to check the crop boxes",
    )
    parser.add_argument("--force", action="store_true", help="render again")
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

    if args.preview:
        try:
            print(f"Layout preview: {render_preview(cfg, target)}")
        except RenderError as exc:
            log.error("%s", exc)
            return 1
        return 0
    if a.renders.is_file() and not args.force:
        log.info("Already rendered: %s (use --force to redo)", a.renders)
        _print_summary(target)
        return 0

    state = State(state_db(cfg)) if target.is_pipeline_dir and target.video_id else None
    video_id = target.video_id or ""
    try:
        with file_lock(lock_path(cfg, f"{STAGE}-{a.root.name}")):
            if state:
                state.mark_stage(video_id, STAGE, StageStatus.RUNNING)
            run_render(cfg, target, force=args.force)
    except LockHeldError as exc:
        log.warning("%s; skipping this run", exc)
        return 0
    except RenderError as exc:
        log.error("%s", exc)
        _mark_failed(state, video_id, exc)
        return 1
    except Exception as exc:
        log.exception("Rendering failed")
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
    manifest = read_json(target.artifacts.renders)
    print(f"Rendered into {target.artifacts.clips_dir}")
    for clip in manifest["clips"]:
        print(
            f"  {clip['file']}  {clip['width']}x{clip['height']}, {clip['duration_s']:.0f} s, "
            f"{clip['size_mb']:.0f} MB  {clip['title']}"
        )
        for flag in clip["flags"]:
            print(f"      ! {flag}")
