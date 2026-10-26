from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from src.common.config import AppConfig, CaptionConfig, Rect, RenderConfig
from src.common.ffmpeg import probe, run_ffmpeg
from src.common.files import read_json, write_json_atomic
from src.common.paths import resolve_target
from src.render.captions import ass_color, ass_time, build_ass, caption_pages
from src.render.core import clip_words, render_preview, run_render
from src.render.layout import Box, filter_graph, plan_layout

# --- layout -----------------------------------------------------------------------------------


def test_stacked_layout_for_this_channel() -> None:
    layout = plan_layout(RenderConfig(), 1920, 1080)
    # gameplay: the middle 52.8 % of the frame at full width, on top
    assert layout.gameplay_src == Box(454, 0, 1014, 1080)
    assert layout.gameplay_out == Box(0, 0, 1080, 1150)
    # avatar: bottom-right corner of the stream, enlarged at its own shape, centred below
    assert layout.avatar_src == Box(1574, 670, 346, 410)
    assert layout.avatar_out == Box(216, 1152, 648, 768)
    assert layout.gameplay_out.h <= layout.avatar_out.y  # no overlap


def test_tall_gameplay_crop_shrinks_to_fit_above_the_avatar() -> None:
    rc = RenderConfig(gameplay=Rect(x=0.3, y=0, w=0.3, h=1.0))
    layout = plan_layout(rc, 1920, 1080)
    go = layout.gameplay_out
    assert go.h == 1920 - layout.avatar_out.h and go.w < 1080
    assert go.x == (1080 - go.w) // 2 // 2 * 2  # centred


def test_center_layout_is_a_plain_9_16_crop() -> None:
    layout = plan_layout(RenderConfig(layout="center"), 1920, 1080)
    assert layout.gameplay_src == Box(656, 0, 608, 1080)
    assert layout.gameplay_out == Box(0, 0, 1080, 1920)
    assert layout.avatar_src is None


def test_filter_graph() -> None:
    graph = filter_graph(plan_layout(RenderConfig(), 1920, 1080), "01.ass")
    assert graph.startswith("[0:v]split=3[bg][game_in][avatar_in]")
    assert "crop=1014:1080:454:0,scale=1080:1150" in graph
    assert "crop=346:410:1574:670,scale=648:768" in graph
    assert graph.endswith("[with_avatar]subtitles=01.ass,format=yuv420p[vout]")
    no_captions = filter_graph(plan_layout(RenderConfig(layout="center"), 1920, 1080), None)
    assert "subtitles" not in no_captions and "split=2" in no_captions


# --- captions ---------------------------------------------------------------------------------


def w(start: float, end: float, word: str) -> dict[str, Any]:
    return {"start": start, "end": end, "word": word}


def test_caption_pages_break_at_count_pauses_and_punctuation() -> None:
    words = [w(0, 0.3, "Wah,"), w(0.3, 0.6, "ini"), w(0.6, 0.9, "langsung"), w(0.9, 1.2, "gua"),
             w(1.2, 1.5, "report"), w(1.5, 1.8, "juga"), w(3.0, 3.3, "Buci.")]  # fmt: skip
    pages = caption_pages(words, max_words=4)
    assert [[x["word"] for x in p] for p in pages] == [
        ["Wah,"], ["ini", "langsung", "gua", "report"], ["juga"], ["Buci."]
    ]  # fmt: skip


def test_word_highlight_ass() -> None:
    cc = CaptionConfig(highlight_color="#FFD400", font_size=80, y=0.5)
    ass = build_ass([w(1.0, 1.4, "halo"), w(1.4, 1.9, "semua")], cc, 1080, 1920)
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    assert "Style: Default,Arial,80,&H00FFFFFF," in ass and ass.split("\n")[9].endswith(",960,1")
    events = [line for line in ass.splitlines() if line.startswith("Dialogue")]
    assert events == [
        "Dialogue: 0,0:00:01.00,0:00:01.40,Default,,0,0,0,,{\\1c&H00D4FF&}halo{\\r} semua",
        "Dialogue: 0,0:00:01.40,0:00:02.20,Default,,0,0,0,,halo {\\1c&H00D4FF&}semua{\\r}",
    ]


def test_sentence_style_and_escaping() -> None:
    cc = CaptionConfig(style="sentence")
    ass = build_ass([w(0, 0.5, "{\\pos(0,0)}hack"), w(0.5, 1.0, "lol")], cc, 1080, 1920)
    event = next(line for line in ass.splitlines() if line.startswith("Dialogue"))
    assert event.endswith(",,(/pos(0,0))hack lol")  # markup can't sneak in from a transcript


def test_ass_helpers() -> None:
    assert ass_color("#FFD400") == "&H0000D4FF"
    assert ass_time(3725.456) == "1:02:05.46"
    assert ass_time(-1) == "0:00:00.00"


def test_clip_words_are_relative_and_cut_at_the_end() -> None:
    words = [
        w(9.0, 9.5, "before"),
        w(10.2, 10.6, "in"),
        w(19.8, 20.4, "edge"),
        w(20.1, 20.5, "after"),
    ]
    assert clip_words(words, 10.0, 20.0) == [w(0.2, 0.6, "in"), w(9.8, 10.0, "edge")]


def test_parse_loudnorm_takes_the_last_json_block() -> None:
    from src.common.ffmpeg import parse_loudnorm

    measurements = {"input_i": "-21.3", "input_tp": "-4.1", "input_lra": "6.2",
                    "input_thresh": "-31.6", "target_offset": "0.4"}  # fmt: skip
    log = "[Parsed_loudnorm_0 @ 000001] {ignored}\n" + json.dumps(measurements, indent=1) + "\n"
    assert parse_loudnorm(log) == measurements


# --- a real render ----------------------------------------------------------------------------

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg not installed")


def sample_folder(cfg: AppConfig, tmp_path: Path) -> Any:
    root = tmp_path / "AAAAAAAAAAA_000000-000004"
    root.mkdir()
    run_ffmpeg(["-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=30", "-f", "lavfi",
                "-i", "sine=frequency=440:sample_rate=48000", "-t", "4", "-c:v", "libx264",
                "-preset", "ultrafast", "-c:a", "aac", str(root / "vod.mp4")])  # fmt: skip
    write_json_atomic(root / "download.json", {"video_id": "AAAAAAAAAAA"})
    write_json_atomic(root / "transcript.json", {"segments": [
        {"id": 0, "start": 0.5, "end": 2.0, "text": "halo semua",
         "words": [w(0.5, 1.0, "halo"), w(1.0, 2.0, "semua")]},
    ]})  # fmt: skip
    write_json_atomic(root / "highlights.json", {"clips": [
        {"rank": 1, "start": 0.4, "end": 3.4, "duration_s": 3.0, "title": "Halo",
         "reason": "Test.", "flags": [], "stream_start": "0:00:00"},
    ]})  # fmt: skip
    return resolve_target(cfg, str(root))


def cpu_cfg(cfg: AppConfig) -> AppConfig:
    render = cfg.render.model_copy(update={"video_encoder": "libx264"})
    return cfg.model_copy(update={"render": render})


@needs_ffmpeg
def test_render_makes_a_vertical_clip_with_thumbnail(cfg: AppConfig, tmp_path: Path) -> None:
    target = sample_folder(cfg, tmp_path)
    manifest = run_render(cpu_cfg(cfg), target)
    clips = target.artifacts.clips_dir
    info = probe(clips / "01.mp4")
    assert (info.width, info.height) == (1080, 1920)
    assert info.duration_s == pytest.approx(3.0, abs=0.15) and info.has_audio
    assert (clips / "01.jpg").stat().st_size > 0 and (clips / "01.ass").is_file()
    assert manifest["clips"][0]["file"] == "01.mp4"
    assert read_json(target.artifacts.renders)["size"] == [1080, 1920]
    assert not list(clips.glob("*.tmp.*"))


@needs_ffmpeg
def test_render_preview_writes_one_frame(cfg: AppConfig, tmp_path: Path) -> None:
    target = sample_folder(cfg, tmp_path)
    preview = render_preview(cpu_cfg(cfg), target)
    assert preview.name == "layout_preview.png" and preview.stat().st_size > 0
