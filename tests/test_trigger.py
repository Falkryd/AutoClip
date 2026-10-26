from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from src.common.config import TriggerConfig
from src.common.state import State, StreamStatus
from src.common.ytdl import MetadataError, VideoMeta
from src.trigger.core import NotReadyError, queue_manually, run_trigger, stream_info
from src.trigger.discover import Candidate, candidate_from_api, parse_rss, uploads_playlist_id
from tests.conftest import FIXTURES

NOW = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)
CFG = TriggerConfig(lookback_hours=48, give_up_after_hours=168)


class FakeMeta:
    """Stands in for yt-dlp; counts calls so tests can check nothing is refetched."""

    def __init__(self, metas: dict[str, VideoMeta | Exception]) -> None:
        self.metas = metas
        self.calls: list[str] = []

    def __call__(self, video_id: str) -> VideoMeta:
        self.calls.append(video_id)
        result = self.metas[video_id]
        if isinstance(result, Exception):
            raise result
        return result


def past_stream(video_id: str, ended_hours_ago: float) -> VideoMeta:
    started = NOW - timedelta(hours=ended_hours_ago + 2)
    return VideoMeta(video_id, f"Stream {video_id}", "was_live", started, 2 * 3600)


def meta(video_id: str, live_status: str | None) -> VideoMeta:
    return VideoMeta(video_id, f"Video {video_id}", live_status, None, None)


def test_parse_rss_reads_ids_and_titles() -> None:
    candidates = parse_rss((FIXTURES / "rss_feed.xml").read_bytes())
    assert candidates == [
        Candidate("AAAAAAAAAAA", "Main bareng chat 🎮 | Live"),
        Candidate("BBBBBBBBBBB", "Short upload"),
    ]


def test_uploads_playlist_id() -> None:
    assert uploads_playlist_id("UCabcdefghijklmnopqrstuv") == "UUabcdefghijklmnopqrstuv"


def test_candidate_from_api() -> None:
    ended = candidate_from_api(
        {
            "id": "AAAAAAAAAAA",
            "snippet": {"title": "t"},
            "liveStreamingDetails": {"actualEndTime": "2026-10-26T10:00:00Z"},
        }
    )
    live = candidate_from_api(
        {"id": "BBBBBBBBBBB", "snippet": {"title": "t"}, "liveStreamingDetails": {}}
    )
    upload = candidate_from_api({"id": "CCCCCCCCCCC", "snippet": {"title": "t"}})
    assert ended.is_stream and ended.ended_at == datetime(2026, 10, 26, 10, tzinfo=UTC)
    assert live.is_stream and live.ended_at is None
    assert upload.is_stream is False


def test_recent_finished_stream_is_queued(state: State) -> None:
    fetch = FakeMeta({"AAAAAAAAAAA": past_stream("AAAAAAAAAAA", ended_hours_ago=3)})
    [decision] = run_trigger(CFG, state, [Candidate("AAAAAAAAAAA", "")], fetch, NOW)
    assert decision.status == StreamStatus.QUEUED
    record = state.get_stream("AAAAAAAAAAA")
    assert record is not None and record.status == StreamStatus.QUEUED
    assert record.title == "Stream AAAAAAAAAAA"
    assert record.ended_at == NOW - timedelta(hours=3)


def test_old_stream_seen_for_first_time_is_skipped(state: State) -> None:
    fetch = FakeMeta({"AAAAAAAAAAA": past_stream("AAAAAAAAAAA", ended_hours_ago=100)})
    [decision] = run_trigger(CFG, state, [Candidate("AAAAAAAAAAA", "")], fetch, NOW)
    assert decision.status == StreamStatus.SKIPPED
    assert "lookback" in decision.note


@pytest.mark.parametrize("live_status", ["is_live", "is_upcoming", "post_live"])
def test_unfinished_streams_wait(state: State, live_status: str) -> None:
    fetch = FakeMeta({"AAAAAAAAAAA": meta("AAAAAAAAAAA", live_status)})
    [decision] = run_trigger(CFG, state, [Candidate("AAAAAAAAAAA", "")], fetch, NOW)
    assert decision.status == StreamStatus.WAITING


@pytest.mark.parametrize("live_status", ["not_live", None])
def test_regular_uploads_are_skipped(state: State, live_status: str | None) -> None:
    fetch = FakeMeta({"AAAAAAAAAAA": meta("AAAAAAAAAAA", live_status)})
    [decision] = run_trigger(CFG, state, [Candidate("AAAAAAAAAAA", "")], fetch, NOW)
    assert decision.status == StreamStatus.SKIPPED


def test_watched_stream_is_queued_even_after_long_processing(state: State) -> None:
    candidate = Candidate("AAAAAAAAAAA", "")
    run_trigger(
        CFG, state, [candidate], FakeMeta({"AAAAAAAAAAA": meta("AAAAAAAAAAA", "post_live")}), NOW
    )
    later = NOW + timedelta(hours=60)
    fetch = FakeMeta({"AAAAAAAAAAA": past_stream("AAAAAAAAAAA", ended_hours_ago=-55)})
    [decision] = run_trigger(CFG, state, [candidate], fetch, later)
    assert decision.status == StreamStatus.QUEUED


def test_settled_videos_are_not_refetched(state: State) -> None:
    candidates = [Candidate("AAAAAAAAAAA", ""), Candidate("BBBBBBBBBBB", "")]
    fetch = FakeMeta(
        {
            "AAAAAAAAAAA": past_stream("AAAAAAAAAAA", ended_hours_ago=1),
            "BBBBBBBBBBB": meta("BBBBBBBBBBB", "not_live"),
        }
    )
    run_trigger(CFG, state, candidates, fetch, NOW)
    assert run_trigger(CFG, state, candidates, fetch, NOW + timedelta(minutes=30)) == []
    assert fetch.calls == ["AAAAAAAAAAA", "BBBBBBBBBBB"]


def test_waiting_stream_missing_from_feed_is_rechecked(state: State) -> None:
    run_trigger(
        CFG,
        state,
        [Candidate("AAAAAAAAAAA", "x")],
        FakeMeta({"AAAAAAAAAAA": meta("AAAAAAAAAAA", "is_live")}),
        NOW,
    )
    fetch = FakeMeta({"AAAAAAAAAAA": past_stream("AAAAAAAAAAA", ended_hours_ago=1)})
    [decision] = run_trigger(CFG, state, [], fetch, NOW + timedelta(hours=2))
    assert decision.status == StreamStatus.QUEUED


def test_metadata_error_keeps_waiting_then_gives_up(state: State) -> None:
    fetch = FakeMeta({"AAAAAAAAAAA": MetadataError("Private video")})
    [first] = run_trigger(CFG, state, [Candidate("AAAAAAAAAAA", "t")], fetch, NOW)
    assert first.status == StreamStatus.WAITING and "Private video" in first.note
    [last] = run_trigger(CFG, state, [], fetch, NOW + timedelta(hours=169))
    assert last.status == StreamStatus.SKIPPED and "gave up" in last.note


def test_api_hints_avoid_metadata_calls(state: State) -> None:
    fetch = FakeMeta({})
    decisions = run_trigger(
        CFG,
        state,
        [
            Candidate("AAAAAAAAAAA", "upload", is_stream=False),
            Candidate("BBBBBBBBBBB", "live now", is_stream=True, ended_at=None),
        ],
        fetch,
        NOW,
    )
    assert [d.status for d in decisions] == [StreamStatus.SKIPPED, StreamStatus.WAITING]
    assert fetch.calls == []


def test_dry_run_saves_nothing(state: State) -> None:
    fetch = FakeMeta({"AAAAAAAAAAA": past_stream("AAAAAAAAAAA", ended_hours_ago=1)})
    run_trigger(CFG, state, [Candidate("AAAAAAAAAAA", "")], fetch, NOW, dry_run=True)
    assert state.get_stream("AAAAAAAAAAA") is None


def test_queue_manually_ignores_lookback(state: State) -> None:
    fetch = FakeMeta({"AAAAAAAAAAA": past_stream("AAAAAAAAAAA", ended_hours_ago=500)})
    decision = queue_manually(state, "AAAAAAAAAAA", fetch, NOW)
    assert decision.status == StreamStatus.QUEUED
    info = stream_info(decision, NOW)
    assert info["url"] == "https://www.youtube.com/watch?v=AAAAAAAAAAA"
    assert info["duration_s"] == 7200


def test_queue_manually_refuses_unfinished_vod(state: State) -> None:
    fetch = FakeMeta({"AAAAAAAAAAA": meta("AAAAAAAAAAA", "post_live")})
    with pytest.raises(NotReadyError):
        queue_manually(state, "AAAAAAAAAAA", fetch, NOW)
    assert state.get_stream("AAAAAAAAAAA") is None


def test_rss_404_is_retried_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.trigger import discover

    feed = (FIXTURES / "rss_feed.xml").read_bytes()
    request = httpx.Request("GET", discover.RSS_URL)
    responses = iter(
        [httpx.Response(404, request=request), httpx.Response(200, content=feed, request=request)]
    )
    monkeypatch.setattr(discover.httpx, "get", lambda *a, **k: next(responses))
    monkeypatch.setattr(discover._get_rss.retry, "sleep", lambda _: None)
    assert len(discover.discover_rss("UC" + "a" * 22)) == 2


def test_rss_persistent_404_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.trigger import discover

    request = httpx.Request("GET", discover.RSS_URL)
    monkeypatch.setattr(discover.httpx, "get", lambda *a, **k: httpx.Response(404, request=request))
    monkeypatch.setattr(discover._get_rss.retry, "sleep", lambda _: None)
    with pytest.raises(discover.FeedUnavailable, match="channel.id"):
        discover.discover_rss("UC" + "a" * 22)


def test_candidates_from_tab_keeps_only_videos() -> None:
    from src.trigger.discover import candidates_from_tab

    info = {
        "entries": [
            {"_type": "url", "id": "8fKUNhHQR_M", "title": "Chill Stream with mates"},
            {"_type": "url", "id": "UCxa-f1esdZktFDntGV1s7Uw", "title": "a channel, not a video"},
            {"_type": "url", "id": None, "title": "no id"},
            "not a dict",
            {"_type": "url", "id": "SQZqOLKKv-E", "title": None},
        ]
    }
    assert candidates_from_tab(info) == [
        Candidate("8fKUNhHQR_M", "Chill Stream with mates"),
        Candidate("SQZqOLKKv-E", ""),
    ]


def test_ytdlp_discovery_channel_without_live_tab(monkeypatch: pytest.MonkeyPatch) -> None:
    from yt_dlp.utils import DownloadError

    from src.trigger import discover

    def no_tab(*_: object) -> dict[str, object]:
        raise DownloadError("ERROR: [youtube:tab] UCx: This channel does not have a streams tab")

    monkeypatch.setattr(discover, "_extract_tab", no_tab)
    assert discover.discover_ytdlp("UC" + "a" * 22, 15) == []


def test_ytdlp_discovery_other_errors_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    from yt_dlp.utils import DownloadError

    from src.trigger import discover

    def blocked(*_: object) -> dict[str, object]:
        raise DownloadError("\x1b[0;31mERROR:\x1b[0m Sign in to confirm you're not a bot")

    monkeypatch.setattr(discover, "_extract_tab", blocked)
    with pytest.raises(discover.DiscoveryError, match="not a bot"):
        discover.discover_ytdlp("UC" + "a" * 22, 15)
