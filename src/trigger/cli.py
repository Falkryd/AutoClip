"""Command line for the trigger stage: python -m src.trigger --help"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from src.common.config import AppConfig, ConfigError, load_config, load_secrets
from src.common.files import write_json_atomic
from src.common.lock import LockHeldError, file_lock
from src.common.logs import setup_logging
from src.common.paths import artifacts, lock_path, state_db, validate_video_id
from src.common.state import State, StreamRecord, StreamStatus, utcnow
from src.common.ytdl import MetadataError, VideoMeta, YtdlEnv, fetch_metadata, ytdl_env
from src.trigger.core import Decision, NotReadyError, queue_manually, run_trigger, stream_info
from src.trigger.discover import (
    Candidate,
    DiscoveryError,
    discover_api,
    discover_rss,
    discover_ytdlp,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.trigger",
        description="Detect finished live streams on your channel and queue them for clipping. "
        "Safe to run repeatedly: videos already settled are not checked again.",
    )
    parser.add_argument("--config", type=Path, help="config file (default: config.yaml)")
    parser.add_argument("--dry-run", action="store_true", help="show decisions without saving")
    parser.add_argument(
        "--queue",
        nargs="+",
        metavar="VIDEO_ID",
        help="queue these videos by hand, ignoring the lookback window",
    )
    parser.add_argument("--list", action="store_true", help="print the state database and exit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    log = setup_logging(cfg, "trigger")
    state = State(state_db(cfg))
    if args.list:
        _print_records(state.streams())
        return 0
    try:
        with file_lock(lock_path(cfg, "trigger")):
            decisions = _run(cfg, state, args, log)
    except LockHeldError as exc:
        log.warning("%s; skipping this run", exc)
        return 0
    except (ConfigError, DiscoveryError, MetadataError, NotReadyError, ValueError) as exc:
        log.error("%s", exc)
        return 1
    except Exception:
        log.exception("Trigger failed")
        return 1
    _print_decisions(decisions, dry_run=args.dry_run)
    return 0


def _run(
    cfg: AppConfig, state: State, args: argparse.Namespace, log: logging.Logger
) -> list[Decision]:
    now = utcnow()
    env = ytdl_env(cfg)

    def fetch(video_id: str) -> VideoMeta:
        return fetch_metadata(video_id, env)

    if args.queue:
        decisions = [
            queue_manually(state, validate_video_id(v), fetch, now, dry_run=args.dry_run)
            for v in args.queue
        ]
    else:
        candidates = _discover(cfg, env)
        log.info("Discovery (%s) found %d videos", cfg.trigger.method, len(candidates))
        decisions = run_trigger(cfg.trigger, state, candidates, fetch, now, dry_run=args.dry_run)

    if not args.dry_run:
        for decision in decisions:
            if decision.status == StreamStatus.QUEUED:
                path = artifacts(cfg, decision.video_id).stream_json
                write_json_atomic(path, stream_info(decision, now))
                log.info("Wrote %s", path)
    return decisions


def _discover(cfg: AppConfig, env: YtdlEnv) -> list[Candidate]:
    if not cfg.channel.id:
        raise ConfigError("Set channel.id in config.yaml (your channel ID, starting with UC)")
    if cfg.trigger.method == "ytdlp":
        return discover_ytdlp(cfg.channel.id, cfg.trigger.max_results, env)
    if cfg.trigger.method == "rss":
        return discover_rss(cfg.channel.id)
    from src.common.youtube_auth import build_youtube  # only the api method needs OAuth

    return discover_api(build_youtube(load_secrets()), cfg.channel.id, cfg.trigger.max_results)


def _print_decisions(decisions: list[Decision], *, dry_run: bool) -> None:
    if not decisions:
        print("Nothing new to check.")
        return
    print(
        ("DRY RUN, nothing saved\n" if dry_run else "")
        + f"{'VIDEO':<12} {'STATUS':<8} NOTE / TITLE"
    )
    for d in decisions:
        print(f"{d.video_id:<12} {d.status:<8} {d.note}\n{'':<21} {d.title}")


def _print_records(records: list[StreamRecord]) -> None:
    if not records:
        print("State database is empty.")
        return
    print(f"{'VIDEO':<12} {'STATUS':<8} {'ENDED (UTC)':<17} NOTE / TITLE")
    for r in records:
        ended = r.ended_at.strftime("%Y-%m-%d %H:%M") if r.ended_at else "-"
        print(f"{r.video_id:<12} {r.status:<8} {ended:<17} {r.note}\n{'':<39} {r.title}")
