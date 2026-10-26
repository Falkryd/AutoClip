"""Command line for the download stage: python -m src.download --help"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from src.common.config import ConfigError, load_config
from src.common.ffmpeg import FFmpegError
from src.common.files import read_json
from src.common.lock import LockHeldError, file_lock
from src.common.logs import setup_logging
from src.common.paths import Artifacts, artifacts, lock_path, state_db, video_id_from
from src.common.state import StageStatus, State
from src.common.timecode import format_timecode
from src.common.ytdl import MetadataError
from src.download.core import (
    DownloadStageError,
    is_complete,
    parse_section,
    run_download,
    sample_dir,
)

STAGE = "download"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.download",
        description="Download a finished stream's VOD, live chat replay and a 16 kHz audio "
        "track into work/<video_id>/. With --section, also cut just that part into samples/ "
        "for developing later stages. Re-running skips files that already exist.",
    )
    parser.add_argument("video", help="video ID or YouTube URL")
    parser.add_argument(
        "--section",
        type=parse_section,
        metavar="START-END",
        help="only this part of the stream, e.g. 2:33:00-3:33:00 (cut from the full VOD, "
        "which is downloaded into work/<id>/ first and kept)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="output folder (default: work/<id>, or samples/<id>_<start>-<end> with --section)",
    )
    parser.add_argument("--force", action="store_true", help="download again even if files exist")
    parser.add_argument("--config", type=Path, help="config file (default: config.yaml)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        video_id = video_id_from(args.video)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    log = setup_logging(cfg, STAGE)

    if args.out:
        root = args.out.resolve()
    elif args.section:
        root = sample_dir(cfg, video_id, args.section)
    else:
        root = artifacts(cfg, video_id).root
    a = Artifacts(root)
    if is_complete(a) and not args.force:
        log.info("Already downloaded to %s (use --force to download again)", root)
        _print_summary(a)
        return 0

    # Only full downloads into work/ are pipeline runs worth recording in the state database.
    state = State(state_db(cfg)) if args.out is None and args.section is None else None
    try:
        with file_lock(lock_path(cfg, f"{STAGE}-{root.name}")):
            if state:
                state.mark_stage(video_id, STAGE, StageStatus.RUNNING)
            run_download(cfg, video_id, a, args.section, force=args.force)
    except LockHeldError as exc:
        log.warning("%s; skipping this run", exc)
        return 0
    except (DownloadStageError, FFmpegError, MetadataError) as exc:
        log.error("%s", exc)
        _mark_failed(state, video_id, exc)
        return 1
    except Exception as exc:
        log.exception("Download failed")
        _mark_failed(state, video_id, exc)
        return 1
    if state:
        state.mark_stage(video_id, STAGE, StageStatus.DONE)
    _print_summary(a)
    return 0


def _mark_failed(state: State | None, video_id: str, exc: BaseException) -> None:
    if state:
        state.mark_stage(video_id, STAGE, StageStatus.FAILED, error=str(exc)[:500])


def _print_summary(a: Artifacts) -> None:
    manifest = read_json(a.download_manifest)
    vod, chat = manifest["vod"], manifest["chat"]
    section = manifest["section"]
    print(f"{manifest['title']}  ({manifest['url']})")
    print(f"Saved to {a.root}" + (f"  [section {section['label']}]" if section else ""))
    print(
        f"  {a.vod.name:<14} {vod['width']}x{vod['height']} @ {vod['fps'] or 0:.0f} fps, "
        f"{format_timecode(vod['duration_s'])}, {_mb(a.vod)}"
    )
    chat_note = f"{chat['messages']} messages" if chat["available"] else "no chat replay"
    print(f"  {a.chat.name:<14} {chat_note}")
    print(
        f"  {a.audio.name:<14} {manifest['audio']['sample_rate'] // 1000} kHz mono, {_mb(a.audio)}"
    )


def _mb(path: Path) -> str:
    return f"{path.stat().st_size / 1e6:,.0f} MB"
