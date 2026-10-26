from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from src.common.config import AppConfig, OllamaConfig
from src.common.files import read_json, write_json_atomic
from src.common.paths import Target, resolve_target
from src.highlights.core import HighlightsError, run_highlights
from src.highlights.llm import LlmError, OllamaJudge, Verdict, format_excerpt
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
    """Likes the moment around 100 s, dislikes everything else."""

    def __init__(self) -> None:
        self.calls: list[list[int]] = []
        self.unloaded = False

    def check_ready(self) -> None:
        pass

    def judge(
        self, excerpt: str, hint: str, ids: list[int], *, min_s: float, max_s: float
    ) -> Verdict:
        self.calls.append(ids)
        if 25 in ids:  # the line at 100 s
            return Verdict(score=9, start_id=23, end_id=28, title="Ngakak", reason="Laughs.")
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
                              "chat": "not available: no chat replay"}  # fmt: skip
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
