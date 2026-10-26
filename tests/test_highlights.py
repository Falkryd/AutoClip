from __future__ import annotations

import json
import shutil
import wave
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from src.common.config import AppConfig, OllamaConfig, Rect
from src.common.ffmpeg import run_ffmpeg
from src.common.files import read_json, write_json_atomic
from src.common.paths import Target, resolve_target
from src.highlights.cli import parse_stretch
from src.highlights.core import HighlightsError, run_highlights
from src.highlights.focus import Focus, focus_times, make_focus
from src.highlights.llm import (
    SYSTEM_PROMPT,
    LlmError,
    OllamaJudge,
    Verdict,
    format_excerpt,
    system_prompt,
)
from src.highlights.selection import (
    clip_bounds,
    combine,
    music_flags,
    pick_final,
    segment_ids_between,
    shortlist,
)
from src.highlights.signals import (
    events_per_second,
    loudness_per_second,
    loudness_spikes,
    motion_from_frames,
    motion_per_second,
    scale,
    window_starts,
    window_sum,
    window_top_mean,
)

RATE = 8000


def write_wav(path: Path, levels_db: list[float]) -> None:
    """One second of noise per entry, at that RMS level in dBFS."""
    rng = np.random.default_rng(0)
    chunks = [rng.standard_normal(RATE) * 10 ** (db / 20) for db in levels_db]
    samples = np.clip(np.concatenate(chunks), -1, 1)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(RATE)
        fh.writeframes((samples * 32767).astype("<i2").tobytes())


# --- signals ----------------------------------------------------------------------------------


def test_loudness_per_second_reads_levels(tmp_path: Path) -> None:
    write_wav(tmp_path / "a.wav", [-40, -10, -40])
    db = loudness_per_second(tmp_path / "a.wav")
    assert len(db) == 3
    assert db[1] - db[0] == pytest.approx(30, abs=0.5)


def test_loudness_spikes_are_relative_to_the_surrounding_minute() -> None:
    db = np.full(120, -30.0)
    db[60] = -12.0
    spikes = loudness_spikes(db)
    assert spikes[60] == pytest.approx(18)
    assert spikes[:60].max() == 0 and spikes[61:].max() == 0
    assert not loudness_spikes(np.full(120, -5.0)).any()  # loud all the time is not a spike


def test_window_scores() -> None:
    per_second = events_per_second([1.5, 2.2, 2.9, 50.0, 99.9, 120.0], duration_s=100)
    assert per_second.sum() == 5  # 120 s is past the end
    starts = window_starts(100, window_s=45, step_s=5)
    assert starts[0] == 0 and starts[-1] == 55
    counts = window_sum(per_second, starts, 45)
    assert counts[0] == 3 and counts[list(starts).index(10)] == 1
    loud = np.zeros(100)
    loud[[10, 11, 12]] = [6, 9, 3]
    assert window_top_mean(loud, np.array([0.0]), 45, top=2)[0] == pytest.approx(7.5)


def test_scale_and_missing_signals() -> None:
    assert scale(np.zeros(50)) is None  # nothing to rank: the signal is left out
    one_laugh = np.zeros(200)
    one_laugh[10] = 1
    assert scale(one_laugh)[10] == 1  # sparse: falls back to the maximum
    values = np.arange(101.0)
    assert scale(values)[-1] == 1 and scale(values)[50] == pytest.approx(50 / 99)


# --- selection --------------------------------------------------------------------------------


def test_combine_rescales_over_present_signals() -> None:
    weights = {"chat": 0.1, "laughter": 0.3, "audio": 0.2, "llm": 0.4}
    assert combine({"laughter": 1.0, "audio": 0.0}, weights) == pytest.approx(0.6)
    both = combine({"laughter": np.array([1.0, 0.0]), "audio": np.array([0.0, 1.0])}, weights)
    assert both == pytest.approx([0.6, 0.4])
    with pytest.raises(ValueError):
        combine({}, weights)


def test_shortlist_skips_overlapping_windows() -> None:
    starts = np.arange(0, 100, 5.0)
    scores = np.zeros(len(starts))
    scores[[4, 5, 15]] = [0.9, 0.8, 0.5]  # windows at 20 s and 25 s overlap
    assert shortlist(starts, scores, window_s=45, size=3)[:2] == [4, 15]


def segs(*spans: tuple[float, float]) -> list[dict[str, Any]]:
    return [{"id": i, "start": s, "end": e, "text": f"line {i}", "words": []}
            for i, (s, e) in enumerate(spans)]  # fmt: skip


def test_clip_bounds_trims_long_choices() -> None:
    segments = segs(*[(i * 10.0, i * 10.0 + 8) for i in range(10)])
    start, end = clip_bounds(segments, 0, 9, [], min_s=30, max_s=60, duration_s=100)
    assert start == 0 and end <= 60 and end == pytest.approx(58.7)


def test_clip_bounds_grows_short_choices_with_whole_lines() -> None:
    segments = segs(*[(i * 10.0, i * 10.0 + 8) for i in range(10)])
    start, end = clip_bounds(segments, 4, 4, [], min_s=30, max_s=60, duration_s=100)
    assert (start, end) == (39.7, 78.7)  # lines 4-7: grown forward first


def test_clip_bounds_keeps_a_laugh_right_after_the_punchline() -> None:
    segments = segs((0, 20), (21, 35), (80, 90))
    start, end = clip_bounds(segments, 0, 1, [37.0], min_s=30, max_s=60, duration_s=100)
    assert end == pytest.approx(38.5)


def test_clip_bounds_pads_in_plain_time_when_lines_run_out() -> None:
    segments = segs((50, 55))
    assert clip_bounds(segments, 0, 0, [], min_s=30, max_s=60, duration_s=70) == (40.0, 70.0)


def test_pick_final_keeps_clips_apart() -> None:
    spans = [(0, 40, 0.9), (70, 110, 0.8), (120, 160, 0.7), (200, 240, 0.6)]
    cands = [{"start": s, "end": e, "score": score} for s, e, score in spans]
    chosen = pick_final(cands, count=3, min_gap_s=60)
    assert [c["start"] for c in chosen] == [0, 120]


def test_music_flags() -> None:
    db = np.full(100, -30.0)
    db[10:40] = -10.0
    speech = np.zeros(100, dtype=bool)
    flags = music_flags(0, 60, [{"t": 12.0, "tag": "musik"}], db, speech, flag_after_s=20)
    assert len(flags) == 2 and "0:00:12" in flags[0] and flags[1].startswith("30 s")
    speech[10:40] = True  # loud because someone's talking: not music
    assert music_flags(0, 60, [], db, speech, flag_after_s=20) == []


# --- LLM ---------------------------------------------------------------------------------------


def test_format_excerpt_puts_sound_tags_on_the_line_they_follow() -> None:
    segments = segs((0, 2), (5, 7), (9, 10))
    sounds = [{"t": 3.0, "tag": "tertawa"}, {"t": 7.5, "tag": "musik"}]
    assert format_excerpt(segments, sounds, [0, 1]).splitlines() == [
        "[0] 0:00:00 line 0 [tertawa]",
        "[1] 0:00:05 line 1 [musik]",
    ]


class FakeOllama:
    """Answers /api/chat from a list of replies and records every request."""

    def __init__(self, replies: list[dict[str, Any]], models: tuple[str, ...] = ("qwen2.5:7b",)):
        self.replies = list(replies)
        self.models = models
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        self.requests.append((request.url.path, body))
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": m} for m in self.models]})
        if request.url.path == "/api/chat":
            return httpx.Response(
                200, json={"message": {"content": json.dumps(self.replies.pop(0))}}
            )
        return httpx.Response(200, json={})

    def judge(self) -> OllamaJudge:
        client = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(self))
        return OllamaJudge(OllamaConfig(), client)


GOOD = {"score": 8, "start_id": 2, "end_id": 4, "title": "Kena ulti lagi", "reason": "Rage."}


def test_judge_returns_a_valid_verdict() -> None:
    fake = FakeOllama([GOOD])
    verdict = fake.judge().judge("[2] ...", "3 laughs", [2, 3, 4, 5], min_s=30, max_s=60)
    assert verdict == Verdict(**GOOD)
    path, body = fake.requests[0]
    assert path == "/api/chat" and body["format"]["required"][0] == "score"
    assert body["options"]["num_ctx"] == 8192  # set explicitly so nothing is silently cut


def test_judge_asks_again_after_a_bad_answer() -> None:
    fake = FakeOllama([{**GOOD, "end_id": 99}, GOOD])
    assert fake.judge().judge("x", "y", [2, 3, 4], min_s=30, max_s=60) == Verdict(**GOOD)
    retry = fake.requests[1][1]["messages"]
    assert "invalid" in retry[-1]["content"] and retry[-2]["role"] == "assistant"


def test_judge_gives_up_after_two_bad_answers() -> None:
    fake = FakeOllama([{**GOOD, "score": 42}, {"nonsense": True}])
    assert fake.judge().judge("x", "y", [2, 3, 4], min_s=30, max_s=60) is None


def test_judge_checks_ollama_and_the_model() -> None:
    FakeOllama([]).judge().check_ready()
    with pytest.raises(LlmError, match="ollama pull"):
        FakeOllama([], models=("llama3.1:8b",)).judge().check_ready()

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(down))
    with pytest.raises(LlmError, match="--no-llm"):
        OllamaJudge(OllamaConfig(), client).check_ready()


def test_judge_refuses_prompts_that_would_be_truncated() -> None:
    with pytest.raises(LlmError, match="num_ctx"):
        FakeOllama([GOOD]).judge().judge("x" * 30_000, "y", [1, 2], min_s=30, max_s=60)


def test_unload_frees_the_model() -> None:
    fake = FakeOllama([])
    fake.judge().unload()
    assert fake.requests == [("/api/generate", {"model": "qwen2.5:7b", "keep_alive": 0})]


# --- the whole stage ---------------------------------------------------------------------------


class ScriptedJudge:
    """Likes the moment around 100 s, dislikes everything else. Given a focus, it likes the
    on-theme moment around 200 s best, as the real prompt asks."""

    def __init__(self) -> None:
        self.calls: list[list[int]] = []
        self.hints: list[str] = []
        self.focus: str | None = None
        self.unloaded = False

    def check_ready(self) -> None:
        pass

    def judge(
        self,
        excerpt: str,
        hint: str,
        ids: list[int],
        *,
        min_s: float,
        max_s: float,
        focus: str | None = None,
    ) -> Verdict:
        self.calls.append(ids)
        self.hints.append(hint)
        self.focus = focus
        if focus and 50 in ids:  # the on-theme line at 200 s
            return Verdict(score=9, start_id=48, end_id=53, title="Gak tidur", reason="Theme.")
        if 25 in ids:  # the line at 100 s
            score = 6 if focus else 9  # funny but off-theme
            return Verdict(score=score, start_id=23, end_id=28, title="Ngakak", reason="Laughs.")
        return Verdict(score=2, start_id=ids[0], end_id=ids[-1], title="Biasa", reason="Dull.")

    def unload(self) -> None:
        self.unloaded = True


def stream_folder(cfg: AppConfig, tmp_path: Path, source: str = "youtube") -> Target:
    """240 s stream: a line every 4 s, laughter and a shout around 100 s."""
    root = tmp_path / "AAAAAAAAAAA"
    root.mkdir()
    levels = [-40.0] * 240
    levels[100:104] = [-12.0] * 4
    write_wav(root / "audio.wav", levels)
    segments = [{"id": i, "start": i * 4.0, "end": i * 4.0 + 3, "text": f"kalimat {i}",
                 "words": [{"start": i * 4.0, "end": i * 4.0 + 3, "word": "x", "prob": None}]}
                for i in range(60)]  # fmt: skip
    sounds = [{"t": t, "tag": "tertawa"} for t in (101.0, 103.5, 105.0, 107.0)] + [
        {"t": 200.0, "tag": "tertawa"}, {"t": 30.0, "tag": "mendengus"}
    ]  # fmt: skip
    write_json_atomic(root / "transcript.json", {
        "source": source, "duration_s": 240.0, "offset_s": 9000.0, "segments": segments,
        "sounds": sounds if source == "youtube" else [],
    })  # fmt: skip
    write_json_atomic(root / "chat.json", {"available": False, "messages": []})
    return resolve_target(cfg, str(root))


def small_cfg(cfg: AppConfig, **highlights: Any) -> AppConfig:
    update = {"shortlist_size": 3, "clips_per_stream": 2} | highlights
    return cfg.model_copy(update={"highlights": cfg.highlights.model_copy(update=update)})


def test_stage_picks_the_laughing_moment(cfg: AppConfig, tmp_path: Path) -> None:
    judge = ScriptedJudge()
    target = stream_folder(cfg, tmp_path)
    doc = run_highlights(small_cfg(cfg), target, judge=judge)  # type: ignore[arg-type]

    assert judge.unloaded and len(judge.calls) == 3
    assert [c["title"] for c in doc["clips"]] == ["Ngakak"]  # the dull ones scored below 4
    clip = doc["clips"][0]
    assert clip["rank"] == 1 and clip["llm_score"] == 9 and clip["laughs"] == 4
    assert 30 <= clip["duration_s"] <= 60 and clip["start"] <= 101 <= clip["end"]
    assert clip["stream_start"] == "2:31:31"  # offset 9000 s + 91.7 s
    assert doc["signals"] == {"audio": "used", "laughter": "used",
                              "chat": "not available: no chat replay",
                              "still_screen": "not checked: no vod.mp4"}  # fmt: skip
    assert read_json(target.artifacts.highlights)["clips"][0]["title"] == "Ngakak"


def test_stage_without_llm_or_laugh_tags(cfg: AppConfig, tmp_path: Path) -> None:
    target = stream_folder(cfg, tmp_path, source="whisper")
    doc = run_highlights(small_cfg(cfg), target, use_llm=False)
    assert doc["signals"]["laughter"].startswith("not available")
    assert doc["signals"]["llm"] == "skipped (--no-llm)"
    best = doc["clips"][0]
    assert best["start"] <= 101 <= best["end"]  # loudness alone still finds the shout
    assert best["reason"] == "LLM not used" and best["llm_score"] is None


def test_stage_needs_a_transcript(cfg: AppConfig, tmp_path: Path) -> None:
    target = stream_folder(cfg, tmp_path)
    target.artifacts.transcript.unlink()
    with pytest.raises(HighlightsError, match="transcribe"):
        run_highlights(cfg, target, use_llm=False)


def test_segment_ids_between() -> None:
    assert segment_ids_between(segs((0, 2), (3, 5), (6, 8)), 4, 6.5) == [1, 2]


# --- glossary and still screens ------------------------------------------------------------


def test_glossary_goes_into_the_llm_instructions() -> None:
    fake = FakeOllama([GOOD])
    client = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(fake))
    judge = OllamaJudge(OllamaConfig(), client, glossary={"kotol": "Keeper of the Light"})
    judge.judge("x", "y", [2, 3, 4], min_s=30, max_s=60)
    system = fake.requests[0][1]["messages"][0]["content"]
    assert "- Nicknames and slang on this stream: kotol = Keeper of the Light." in system
    assert system.index("kotol") < system.index("Speech recognition makes mistakes")
    assert system_prompt({}) == SYSTEM_PROMPT


def test_motion_from_frames_takes_the_bigger_change_per_second() -> None:
    frames = np.array([[0.0], [0.0], [10.0], [10.0], [10.0], [30.0]])
    # differences per half second: 0, 10, 0, 0, 20 -> seconds: max(0, 10), max(0, 0)
    assert motion_from_frames(frames).tolist() == [10.0, 0.0]
    assert len(motion_from_frames(frames[:1])) == 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg not installed")
def test_motion_per_second_tells_moving_from_still(tmp_path: Path) -> None:
    vod = tmp_path / "vod.mp4"
    run_ffmpeg(["-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=3",
                "-f", "lavfi", "-i", "color=c=gray:size=320x180:rate=30:duration=3",
                "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]",
                "-c:v", "libx264", "-preset", "ultrafast", str(vod)])  # fmt: skip
    motion = motion_per_second(vod, Rect(x=0, y=0, w=1, h=1))
    assert len(motion) >= 5
    assert motion[:2].min() > 4 and motion[-2:].max() < 1


def test_still_screen_lowers_the_score_and_is_flagged(cfg: AppConfig, tmp_path: Path) -> None:
    target = stream_folder(cfg, tmp_path)
    motion = [20.0] * 240
    motion[80:140] = [1.0] * 60  # the game is paused around the laughing moment
    write_json_atomic(target.artifacts.motion, {"fps": 2, "per_second": motion})

    paused = run_highlights(small_cfg(cfg), target, judge=ScriptedJudge())  # type: ignore[arg-type]
    clip = paused["clips"][0]
    assert clip["still_screen"] >= 0.9
    assert any("still screen" in flag for flag in clip["flags"])
    assert paused["signals"]["still_screen"] == "used (penalty up to 50%)"

    plain = run_highlights(small_cfg(cfg, still_screen_penalty=0), target, judge=ScriptedJudge())  # type: ignore[arg-type]
    assert plain["signals"]["still_screen"].startswith("off")
    assert plain["clips"][0]["score"] > clip["score"] * 1.5


# --- focus ---------------------------------------------------------------------------------


def test_make_focus_takes_the_meaningful_words() -> None:
    assert make_focus("ga tidur sebelum menang") == Focus(
        "ga tidur sebelum menang", ("tidur", "menang")
    )
    assert make_focus("x", "Tidur, menang ,ngantuk").words == ("tidur", "menang", "ngantuk")
    with pytest.raises(ValueError, match="--focus-words"):
        make_focus("ga yang aja")


def test_focus_times_match_inside_longer_words() -> None:
    said = [(1.0, "Kemenangan!"), (2.0, "tidurnya"), (3.0, "farming")]
    segments = [{"words": [{"start": t, "word": w} for t, w in said]}]
    assert focus_times(segments, make_focus("ga tidur sebelum menang")) == [1.0, 2.0]


def test_focus_is_put_in_the_llm_prompt() -> None:
    fake = FakeOllama([GOOD])
    fake.judge().judge("x", "y", [2, 3, 4], min_s=30, max_s=60, focus="ga tidur sebelum menang")
    prompt = fake.requests[0][1]["messages"][1]["content"]
    assert prompt.startswith('The streamer especially wants clips about this theme: "ga tidur')
    assert "at most 6" in prompt


def test_focus_shifts_the_pick_to_the_theme(cfg: AppConfig, tmp_path: Path) -> None:
    target = stream_folder(cfg, tmp_path)
    transcript = read_json(target.artifacts.transcript)
    transcript["segments"][50]["words"] = [
        {"start": 200.0, "end": 200.5, "word": "gak"},
        {"start": 200.5, "end": 201.0, "word": "tidur"},
        {"start": 201.0, "end": 201.5, "word": "sampai"},
        {"start": 201.5, "end": 202.0, "word": "menang"},
    ]
    write_json_atomic(target.artifacts.transcript, transcript)
    one_clip = small_cfg(cfg, clips_per_stream=1)

    plain = run_highlights(one_clip, target, judge=ScriptedJudge())  # type: ignore[arg-type]
    assert plain["clips"][0]["title"] == "Ngakak" and plain["focus"] is None

    judge = ScriptedJudge()
    focus = make_focus("ga tidur sebelum menang")
    themed = run_highlights(one_clip, target, judge=judge, focus=focus)  # type: ignore[arg-type]
    clip = themed["clips"][0]
    assert clip["title"] == "Gak tidur" and clip["theme_words"] == 2
    assert themed["focus"] == {"theme": "ga tidur sebelum menang", "words": ["tidur", "menang"]}
    assert themed["signals"]["focus"] == "used"
    assert judge.focus == "ga tidur sebelum menang"
    assert any("theme words (tidur, menang)" in hint for hint in judge.hints)


# --- --clips and --include -------------------------------------------------------------------


def test_pick_final_starts_with_the_asked_for_clips() -> None:
    asked = {"start": 100, "end": 140, "score": 0.1}
    cands = [{"start": 110, "end": 150, "score": 0.9}, {"start": 300, "end": 340, "score": 0.8}]
    chosen = pick_final(cands, count=2, min_gap_s=60, start_with=[asked])
    assert chosen == [asked, cands[1]]  # the 0.9 one overlaps the asked-for clip
    assert pick_final(cands, count=1, min_gap_s=60, start_with=[asked, asked]) == [asked]


def test_parse_stretch() -> None:
    assert parse_stretch("3:34:00-3:35:24") == (12840.0, 12924.0)
    for bad in ("3:34:00", "3:35:00-3:34:00"):
        with pytest.raises(ValueError):
            parse_stretch(bad)


def test_include_keeps_a_stretch_the_llm_dislikes(cfg: AppConfig, tmp_path: Path) -> None:
    target = stream_folder(cfg, tmp_path)
    doc = run_highlights(
        small_cfg(cfg, min_gap_seconds=10), target, judge=ScriptedJudge(),  # type: ignore[arg-type]
        include=[(195.0, 240.0)], clips=2,
    )  # fmt: skip
    asked = [c for c in doc["clips"] if c["asked_for"]]
    assert len(doc["clips"]) == 2 and len(asked) == 1
    assert asked[0]["llm_score"] == 2  # below min_llm_score, kept anyway
    assert asked[0]["start"] >= 170 and asked[0]["end"] <= 240
    assert {c["title"] for c in doc["clips"]} == {"Ngakak", "Biasa"}
    assert doc["included"] == [[195.0, 240.0]]
    assert [c["rank"] for c in doc["clips"]] == [1, 2]


def test_clips_option_overrides_the_config(cfg: AppConfig, tmp_path: Path) -> None:
    target = stream_folder(cfg, tmp_path)
    doc = run_highlights(
        small_cfg(cfg), target, use_llm=False, clips=1,
    )  # fmt: skip
    assert len(doc["clips"]) == 1
