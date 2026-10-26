from __future__ import annotations

from pathlib import Path

import pytest

from src.common.config import AppConfig
from src.common.ffmpeg import last_keyframe, media_info_from_probe
from src.common.paths import Artifacts, video_id_from
from src.common.timecode import compact_timecode, format_timecode, parse_timecode
from src.download.chat import ChatMessage, chat_document, clip_to_window, parse_chat_lines
from src.download.core import Section, is_complete, parse_section, sample_dir
from tests.conftest import FIXTURES


def load_fixture_chat() -> tuple[list[ChatMessage], int]:
    with (FIXTURES / "live_chat.jsonl").open(encoding="utf-8") as fh:
        return parse_chat_lines(fh)


def test_parse_chat_reads_every_message_type() -> None:
    messages, bad_lines = load_fixture_chat()
    assert bad_lines == 1
    assert messages == [
        ChatMessage(-5.0, "Early", "waiting for stream", "text"),
        ChatMessage(9100.0, "Andi", "Welcome to Falkryd Squad", "membership"),
        ChatMessage(9185.5, "Budi", "gg wp :face_with_tears_of_joy:", "text"),
        ChatMessage(9186.0, "Rina", ":_falkLUL:", "text"),
        ChatMessage(9200.0, "Sari", "[Rp20.000] semangat bang", "superchat"),
    ]


def test_clip_to_window_makes_times_relative() -> None:
    messages, _ = load_fixture_chat()
    clipped = clip_to_window(messages, 9180, 12780)
    assert [(m.author, m.t) for m in clipped] == [("Budi", 5.5), ("Rina", 6.0), ("Sari", 20.0)]


def test_clip_to_window_drops_pre_stream_chat() -> None:
    messages, _ = load_fixture_chat()
    assert all(m.t >= 0 for m in clip_to_window(messages, 0, 13683))


def test_chat_document_shape() -> None:
    doc = chat_document(
        "8fKUNhHQR_M", [ChatMessage(1.0, "a", "b", "text")],
        available=True, offset_s=9180, duration_s=3600, skipped_lines=0,
    )  # fmt: skip
    assert doc["message_count"] == 1
    assert doc["messages"] == [{"t": 1.0, "author": "a", "text": "b", "kind": "text"}]


def test_parse_section() -> None:
    assert parse_section("2:33:00-3:33:00") == Section(9180, 12780)
    assert parse_section("10:00-20:00").duration_s == 600
    for bad in ("2:33:00", "3:00:00-2:00:00", "a-b", "1:75-2:00"):
        with pytest.raises(ValueError):
            parse_section(bad)


def test_timecodes() -> None:
    assert parse_timecode("2:33:00") == 9180
    assert parse_timecode("90.5") == 90.5
    assert format_timecode(13683) == "3:48:03"
    assert compact_timecode(9180) == "023300"
    with pytest.raises(ValueError):
        parse_timecode("1::2")


def test_sample_dir_name(cfg: AppConfig) -> None:
    path = sample_dir(cfg, "8fKUNhHQR_M", Section(9180, 12780))
    assert path.name == "8fKUNhHQR_M_023300-033300"


@pytest.mark.parametrize(
    "text",
    [
        "8fKUNhHQR_M",
        "https://youtube.com/live/8fKUNhHQR_M",
        "https://www.youtube.com/live/8fKUNhHQR_M?si=abc",
        "https://www.youtube.com/watch?v=8fKUNhHQR_M&t=9180s",
        "https://youtu.be/8fKUNhHQR_M",
        "https://www.youtube.com/shorts/8fKUNhHQR_M",
    ],
)
def test_video_id_from_urls(text: str) -> None:
    assert video_id_from(text) == "8fKUNhHQR_M"


def test_video_id_from_rejects_other_text() -> None:
    with pytest.raises(ValueError):
        video_id_from("https://www.youtube.com/@falkryd")


def test_media_info_from_probe() -> None:
    info = media_info_from_probe(
        {
            "streams": [
                {"codec_type": "video", "width": 1920, "height": 1080,
                 "avg_frame_rate": "60/1", "start_time": "0.000000"},
                {"codec_type": "audio", "start_time": "0.021000"},
            ],
            "format": {"duration": "3600.5"},
        }
    )  # fmt: skip
    assert (info.width, info.height, info.fps, info.duration_s) == (1920, 1080, 60.0, 3600.5)
    assert info.has_audio and info.audio_start_s == 0.021


def test_media_info_handles_unknown_frame_rate() -> None:
    info = media_info_from_probe(
        {"streams": [{"codec_type": "video", "avg_frame_rate": "0/0"}], "format": {"duration": "1"}}
    )
    assert info.fps is None and not info.has_audio


def test_last_keyframe_picks_the_latest_keyframe_before_t() -> None:
    csv = "9170.000000,K__\n9172.400000,__\n9174.833000,K__\n9180.100000,K__\nN/A,K__\n"
    assert last_keyframe(csv, 9180.0) == 9174.833
    assert last_keyframe("9181.0,K__\n", 9180.0) is None


def test_is_complete_needs_every_file(tmp_path: Path) -> None:
    a = Artifacts(tmp_path)
    for path in (a.vod, a.chat, a.audio):
        path.write_text("x")
    assert not is_complete(a)  # manifest is written last
    a.download_manifest.write_text("{}")
    assert is_complete(a)
