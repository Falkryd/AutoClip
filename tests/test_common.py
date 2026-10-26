from __future__ import annotations

import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from src.common.config import DEFAULT_CONFIG_PATH, AppConfig, Rect, load_config
from src.common.files import read_json, write_json_atomic
from src.common.lock import LockHeldError, file_lock
from src.common.logs import RedactingFormatter, redact
from src.common.paths import artifacts, validate_video_id
from src.common.retry import is_transient
from src.common.state import StageStatus, State, StreamStatus


def test_shipped_config_is_valid() -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert cfg.upload.privacy == "private"
    assert cfg.highlights.clips_per_stream == 3
    assert cfg.trigger.method == "ytdlp"


def test_upload_privacy_cannot_be_public() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"upload": {"privacy": "public"}})


def test_clip_cap_protects_quota() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"highlights": {"clips_per_stream": 10}})


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"render": {"layuot": "stacked"}})


def test_rect_must_stay_inside_frame() -> None:
    with pytest.raises(ValidationError):
        Rect(x=0.8, y=0, w=0.3, h=1)


def test_redact_masks_webhooks_and_tokens() -> None:
    text = (
        "posting to https://discord.com/api/webhooks/123/abcDEF-xyz with ya29.a0AfH6SMB "
        "and 1//0gLONGrefreshTOKENvalue123456"
    )
    assert redact(text).count("[REDACTED]") == 3
    assert "abcDEF" not in redact(text)


def test_redacting_formatter_covers_tracebacks() -> None:
    try:
        raise RuntimeError("bad url https://discordapp.com/api/webhooks/1/secret")
    except RuntimeError:
        record = logging.LogRecord("x", logging.ERROR, __file__, 1, "failed", None, sys.exc_info())
    assert "secret" not in RedactingFormatter().format(record)


def test_validate_video_id_blocks_path_tricks() -> None:
    assert validate_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    for bad in ("../../etc/xx", "short", "dQw4w9WgXcQ/"):
        with pytest.raises(ValueError):
            validate_video_id(bad)


def test_artifacts_layout(cfg: AppConfig) -> None:
    a = artifacts(cfg, "dQw4w9WgXcQ")
    assert a.transcript.parent.name == "dQw4w9WgXcQ"
    assert a.clips_dir.name == "clips"


def test_write_json_atomic_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "out.json"
    write_json_atomic(path, {"title": "Halo semua 👋"})
    assert read_json(path) == {"title": "Halo semua 👋"}
    assert list(path.parent.iterdir()) == [path]  # no temp files left


def test_file_lock_blocks_second_holder(tmp_path: Path) -> None:
    lock = tmp_path / "x.lock"
    with file_lock(lock), pytest.raises(LockHeldError), file_lock(lock):
        pass
    with file_lock(lock):  # released after the first holder exits
        pass


def test_state_upsert_keeps_discovery_time_and_known_end(state: State) -> None:
    first = datetime(2026, 10, 1, tzinfo=UTC)
    ended = datetime(2026, 10, 1, 3, tzinfo=UTC)
    state.upsert_stream(
        "AAAAAAAAAAA",
        title="t",
        status=StreamStatus.WAITING,
        live_status="post_live",
        ended_at=ended,
        note="",
        now=first,
    )
    state.upsert_stream(
        "AAAAAAAAAAA",
        title="",
        status=StreamStatus.QUEUED,
        live_status=None,
        ended_at=None,
        note="VOD ready",
        now=datetime(2026, 10, 2, tzinfo=UTC),
    )
    record = state.get_stream("AAAAAAAAAAA")
    assert record is not None
    assert record.discovered_at == first
    assert record.ended_at == ended
    assert record.title == "t"
    assert record.live_status == "post_live"
    assert record.status == StreamStatus.QUEUED
    assert [r.video_id for r in state.streams(StreamStatus.QUEUED)] == ["AAAAAAAAAAA"]


def test_stage_tracking(state: State) -> None:
    assert state.stage_status("AAAAAAAAAAA", "download") is None
    state.mark_stage("AAAAAAAAAAA", "download", StageStatus.RUNNING)
    state.mark_stage("AAAAAAAAAAA", "download", StageStatus.DONE)
    assert state.stage_status("AAAAAAAAAAA", "download") == StageStatus.DONE


def test_is_transient() -> None:
    request = httpx.Request("GET", "https://example.com")
    assert is_transient(httpx.ConnectTimeout("t", request=request))
    assert is_transient(
        httpx.HTTPStatusError("x", request=request, response=httpx.Response(503, request=request))
    )
    assert not is_transient(
        httpx.HTTPStatusError("x", request=request, response=httpx.Response(404, request=request))
    )
    assert not is_transient(ValueError("bad input"))
