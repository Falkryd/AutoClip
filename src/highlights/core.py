"""Score the stream, let the LLM judge the best windows, and write highlights.json."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.common.config import AppConfig, HighlightsConfig
from src.common.files import read_json, write_json_atomic
from src.common.paths import Target
from src.common.state import utcnow
from src.common.timecode import format_timecode
from src.highlights.llm import OllamaJudge, Verdict, format_excerpt
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
    speech_per_second,
    window_starts,
    window_sum,
    window_top_mean,
)

_log = logging.getLogger("autoclip.highlights")


class HighlightsError(RuntimeError):
    """An input is missing or unusable."""


class _Inputs:
    """Everything read from the folder, plus the per-window signal scores."""

    def __init__(self, hc: HighlightsConfig, target: Target) -> None:
        a = target.artifacts
        for path, stage in ((a.transcript, "transcribe"), (a.audio, "download")):
            if not path.is_file():
                raise HighlightsError(f"{path} not found; run the {stage} stage first")
        transcript = read_json(a.transcript)
        self.segments: list[dict[str, Any]] = transcript["segments"]
        if any(s["id"] != i for i, s in enumerate(self.segments)):
            raise HighlightsError(f"Segment ids in {a.transcript} aren't sequential")
        self.source: str = transcript["source"]
        self.duration: float = float(transcript["duration_s"])
        self.offset: float = float(transcript.get("offset_s", 0.0))
        self.sounds: list[dict[str, Any]] = transcript["sounds"]
        laugh_tags = {t.lower() for t in hc.laughter_tags}
        self.laugh_times = [s["t"] for s in self.sounds if s["tag"].lower() in laugh_tags]
        chat = read_json(a.chat) if a.chat.is_file() else {"available": False, "messages": []}
        self.chat_available: bool = bool(chat.get("available"))
        self.chat_times = [m["t"] for m in chat.get("messages", [])]
        self.loudness = loudness_per_second(a.audio)

        self.starts = window_starts(self.duration, hc.window_seconds, hc.step_seconds)
        self.raw: dict[str, np.ndarray] = {
            "audio": window_top_mean(loudness_spikes(self.loudness), self.starts, hc.window_seconds)
        }
        self.notes: dict[str, str] = {}
        if self.source == "youtube":
            self.raw["laughter"] = self._window_counts(self.laugh_times, hc)
        else:
            self.notes["laughter"] = "not available: Whisper transcripts have no laugh tags"
        if self.chat_available:
            self.raw["chat"] = self._window_counts(self.chat_times, hc)
        else:
            self.notes["chat"] = "not available: no chat replay"
        self.scaled: dict[str, np.ndarray] = {}
        for name, values in self.raw.items():
            scaled = scale(values)
            if scaled is None:
                self.notes[name] = "not available: nothing in this stream"
            else:
                self.scaled[name] = scaled

    def _window_counts(self, times: list[float], hc: HighlightsConfig) -> np.ndarray:
        return window_sum(events_per_second(times, self.duration), self.starts, hc.window_seconds)

    def hint(self, i: int) -> str:
        """The cheap signals in words, for the LLM prompt."""
        parts = []
        if "laughter" in self.raw:
            parts.append(f"{int(self.raw['laughter'][i])} laughs")
        parts.append(f"loudest moments {self.raw['audio'][i]:.0f} dB above the surrounding minute")
        if "chat" in self.raw:
            parts.append(f"{int(self.raw['chat'][i])} chat messages")
        return ", ".join(parts)

    def count_between(self, times: list[float], start: float, end: float) -> int:
        return sum(1 for t in times if start <= t <= end)


def run_highlights(
    cfg: AppConfig, target: Target, *, use_llm: bool = True, judge: OllamaJudge | None = None
) -> dict[str, Any]:
    hc = cfg.highlights
    inputs = _Inputs(hc, target)
    weights = hc.weights.model_dump()
    signal_weights = {name: w for name, w in weights.items() if name != "llm"}
    first_pass = combine(inputs.scaled, signal_weights)
    picked = shortlist(inputs.starts, first_pass, hc.window_seconds, hc.shortlist_size)
    _log.info(
        "Signals: %s. Shortlisted %d windows for the LLM",
        ", ".join(inputs.scaled) or "none", len(picked),
    )  # fmt: skip

    owns_judge = use_llm and judge is None
    if owns_judge:
        judge = OllamaJudge(hc.ollama)
    if not use_llm:
        judge = None
        inputs.notes["llm"] = "skipped (--no-llm)"
    candidates = []
    try:
        if judge:
            judge.check_ready()
        for n, i in enumerate(picked, start=1):
            candidate = _evaluate(i, inputs, hc, weights, judge)
            _log.info(
                "Candidate %d/%d at %s: score %.2f%s",
                n, len(picked), format_timecode(candidate["start"]), candidate["score"],
                f", LLM {candidate['llm_score']}/10" if candidate["llm_score"] is not None else "",
            )  # fmt: skip
            candidates.append(candidate)
    finally:
        if judge:
            judge.unload()
            if owns_judge:
                judge.close()

    eligible = [
        c for c in candidates if c["llm_score"] is None or c["llm_score"] >= hc.min_llm_score
    ]
    chosen = pick_final(eligible, hc.clips_per_stream, hc.min_gap_seconds)
    speech = speech_per_second(inputs.segments, inputs.duration)
    for rank, clip in enumerate(chosen, start=1):
        clip["rank"] = rank
        clip["flags"] = music_flags(
            clip["start"], clip["end"], inputs.sounds, inputs.loudness, speech,
            hc.music_flag_seconds,
        )  # fmt: skip

    document = {
        "video_id": target.video_id,
        "transcript_source": inputs.source,
        "duration_s": inputs.duration,
        "offset_s": inputs.offset,
        "created_at": utcnow().isoformat(timespec="seconds"),
        "llm_model": hc.ollama.model if judge else None,
        "signals": {name: "used" for name in inputs.scaled} | inputs.notes,
        "weights": weights,
        "clips": chosen,
        "candidates": sorted(candidates, key=lambda c: -c["score"]),
    }
    write_json_atomic(target.artifacts.highlights, document)  # marks the stage done
    return document


def _evaluate(
    i: int,
    inputs: _Inputs,
    hc: HighlightsConfig,
    weights: dict[str, float],
    judge: OllamaJudge | None,
) -> dict[str, Any]:
    segments = inputs.segments
    window_start = float(inputs.starts[i])
    window_end = window_start + hc.window_seconds
    context_ids = segment_ids_between(
        segments, window_start - hc.context_seconds, window_end + hc.context_seconds
    )
    verdict: Verdict | None = None
    if judge and context_ids:
        excerpt = format_excerpt(segments, inputs.sounds, context_ids)
        verdict = judge.judge(
            excerpt, inputs.hint(i), context_ids,
            min_s=hc.min_clip_seconds, max_s=hc.max_clip_seconds,
        )  # fmt: skip

    window_ids = segment_ids_between(segments, window_start, window_end)
    if verdict:
        first_id, last_id = verdict.start_id, verdict.end_id
    elif window_ids:
        first_id, last_id = window_ids[0], window_ids[-1]
    else:
        first_id = last_id = None
    if first_id is not None and last_id is not None:
        start, end = clip_bounds(
            segments, first_id, last_id, inputs.laugh_times,
            min_s=hc.min_clip_seconds, max_s=hc.max_clip_seconds, duration_s=inputs.duration,
        )  # fmt: skip
    else:  # nobody talks here: the window itself
        start, end = window_start, min(window_end, inputs.duration)

    scores = {name: round(float(values[i]), 3) for name, values in inputs.scaled.items()}
    if verdict:
        scores["llm"] = verdict.score / 10
    covered = segment_ids_between(segments, start, end)
    if verdict:
        title, reason = verdict.title, verdict.reason
    else:
        title = " ".join(segments[covered[0]]["text"].split()[:8]) if covered else "(no speech)"
        reason = "LLM not used" if judge is None else "LLM gave no usable answer"
    return {
        "start": start,
        "end": end,
        "duration_s": round(end - start, 2),
        "stream_start": format_timecode(inputs.offset + start),
        "stream_end": format_timecode(inputs.offset + end),
        "score": round(float(combine(scores, weights)), 3),
        "scores": scores,
        "llm_score": verdict.score if verdict else None,
        "title": title,
        "reason": reason,
        "laughs": inputs.count_between(inputs.laugh_times, start, end),
        "chat_messages": inputs.count_between(inputs.chat_times, start, end),
        "loudness_spike_db": round(float(inputs.raw["audio"][i]), 1),
        "segment_ids": [covered[0], covered[-1]] if covered else None,
        "text": " ".join(segments[j]["text"] for j in covered),
    }
