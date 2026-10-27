"""Score the stream, let the LLM judge the best windows, and write highlights.json."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from src.common.config import AppConfig, HighlightsConfig
from src.common.files import read_json, write_json_atomic
from src.common.paths import Artifacts, Target
from src.common.state import utcnow
from src.common.timecode import format_timecode
from src.highlights.focus import Focus, focus_times
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
    MOTION_FPS,
    STILL_MOTION,
    events_per_second,
    loudness_per_second,
    loudness_spikes,
    motion_per_second,
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

    def __init__(self, cfg: AppConfig, target: Target, focus: Focus | None = None) -> None:
        hc = cfg.highlights
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
        manifest = read_json(a.download_manifest) if a.download_manifest.is_file() else {}
        self.title: str | None = manifest.get("title")  # context for the LLM
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
        self.focus = focus
        self.focus_times = focus_times(self.segments, focus) if focus else []
        if focus:
            self.raw["focus"] = self._window_counts(self.focus_times, hc)
        self.scaled: dict[str, np.ndarray] = {}
        for name, values in self.raw.items():
            scaled = scale(values)
            if scaled is None:
                self.notes[name] = "not available: nothing in this stream"
            else:
                self.scaled[name] = scaled

        self.penalty = hc.still_screen_penalty
        self.still = self._still_seconds(cfg, a)
        self.window_still = (
            window_sum(self.still.astype(float), self.starts, hc.window_seconds) / hc.window_seconds
            if self.still is not None
            else np.zeros(len(self.starts))
        )

    def _still_seconds(self, cfg: AppConfig, a: Artifacts) -> np.ndarray | None:
        """True for seconds where the gameplay area barely moves (paused game, menus)."""
        if self.penalty <= 0:
            self.notes["still_screen"] = "off (still_screen_penalty: 0)"
            return None
        if a.motion.is_file():
            motion = np.array(read_json(a.motion)["per_second"], dtype=float)
        elif a.vod.is_file():
            _log.info("Measuring how much the gameplay moves (once; cached in %s)", a.motion.name)
            motion = motion_per_second(a.vod, cfg.render.gameplay)
            cache = {
                "fps": MOTION_FPS,
                "area": cfg.render.gameplay.model_dump(),
                "per_second": [round(float(v), 2) for v in motion],
            }
            write_json_atomic(a.motion, cache)
        else:
            self.notes["still_screen"] = "not checked: no vod.mp4"
            return None
        self.notes["still_screen"] = f"used (penalty up to {self.penalty:.0%})"
        return motion < STILL_MOTION

    def still_fraction(self, start: float, end: float) -> float:
        if self.still is None:
            return 0.0
        part = self.still[int(start) : max(int(end), int(start) + 1)]
        return float(part.mean()) if len(part) else 0.0

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
        if self.focus and "focus" in self.raw:
            words = ", ".join(self.focus.words)
            parts.append(f"{int(self.raw['focus'][i])} theme words ({words})")
        return ", ".join(parts)

    def count_between(self, times: list[float], start: float, end: float) -> int:
        return sum(1 for t in times if start <= t <= end)


def run_highlights(
    cfg: AppConfig,
    target: Target,
    *,
    use_llm: bool = True,
    judge: OllamaJudge | None = None,
    focus: Focus | None = None,
    include: Sequence[tuple[float, float]] = (),
    clips: int | None = None,
) -> dict[str, Any]:
    """focus: a theme to favour (--focus); its words become a signal and the LLM is told.
    include: (start, end) stretches of vod.mp4 that must become clips (--include).
    clips: how many clips this time, instead of highlights.clips_per_stream (--clips)."""
    hc = cfg.highlights
    count = clips or hc.clips_per_stream
    inputs = _Inputs(cfg, target, focus)
    if focus:
        _log.info("Focus: %r; %d theme words in the transcript (%s)",
                  focus.theme, len(inputs.focus_times), ", ".join(focus.words))  # fmt: skip
    weights = hc.weights.model_dump()
    signal_weights = {name: w for name, w in weights.items() if name != "llm"}
    # A still screen is penalised already here, so live moments get the LLM's attention.
    first_pass = combine(inputs.scaled, signal_weights) * (1 - inputs.penalty * inputs.window_still)
    # At least three candidates per clip, so the LLM has a real choice even for 7+ clips.
    size = max(hc.shortlist_size, 3 * count)
    picked = shortlist(inputs.starts, first_pass, hc.window_seconds, size)
    _log.info(
        "Signals: %s. Shortlisted %d windows for the LLM",
        ", ".join(inputs.scaled) or "none", len(picked),
    )  # fmt: skip

    owns_judge = use_llm and judge is None
    if owns_judge:
        judge = OllamaJudge(hc.ollama, glossary=hc.glossary)
    if not use_llm:
        judge = None
        inputs.notes["llm"] = "skipped (--no-llm)"
    # Asked-for stretches are judged like any window (the LLM still picks clean lines and
    # writes the title) but are always kept. Their signal scores come from the nearest window.
    windows: list[tuple[int, tuple[float, float] | None]] = [(i, None) for i in picked]
    for start, end in include:
        start, end = max(0.0, start), min(end, inputs.duration)
        nearest = int(np.argmin(np.abs(inputs.starts - start)))
        windows.append((nearest, (start, end)))
    candidates, forced = [], []
    try:
        if judge:
            judge.check_ready()
        for n, (i, window) in enumerate(windows, start=1):
            candidate = _evaluate(i, inputs, hc, weights, judge, window)
            _log.info(
                "Candidate %d/%d at %s: score %.2f%s%s",
                n, len(windows), format_timecode(candidate["start"]), candidate["score"],
                f", LLM {candidate['llm_score']}/10" if candidate["llm_score"] is not None else "",
                " (asked for)" if window else "",
            )  # fmt: skip
            candidate["asked_for"] = window is not None
            (forced if window else candidates).append(candidate)
    finally:
        if judge:
            judge.unload()
            if owns_judge:
                judge.close()

    eligible = [
        c for c in candidates if c["llm_score"] is None or c["llm_score"] >= hc.min_llm_score
    ]
    rank = theme_rank if focus else None
    chosen = pick_final(eligible, count, hc.min_gap_seconds, start_with=forced, rank=rank)
    chosen.sort(key=rank or (lambda c: c["score"]), reverse=True)
    speech = speech_per_second(inputs.segments, inputs.duration)
    for rank, clip in enumerate(chosen, start=1):
        clip["rank"] = rank
        clip["flags"] = music_flags(
            clip["start"], clip["end"], inputs.sounds, inputs.loudness, speech,
            hc.music_flag_seconds,
        )  # fmt: skip
        if clip["still_screen"] >= 0.5:
            clip["flags"].append(
                f"{clip['still_screen']:.0%} still screen (paused game, menu or scoreboard)"
            )

    document = {
        "video_id": target.video_id,
        "transcript_source": inputs.source,
        "duration_s": inputs.duration,
        "offset_s": inputs.offset,
        "created_at": utcnow().isoformat(timespec="seconds"),
        "llm_model": hc.ollama.model if judge else None,
        "focus": {"theme": focus.theme, "words": list(focus.words)} if focus else None,
        "ranking": "LLM theme score, then score (--focus)" if focus else "score",
        "signals": {name: "used" for name in inputs.scaled} | inputs.notes,
        "weights": weights,
        "included": [[round(s, 2), round(e, 2)] for s, e in include],
        "clips": chosen,
        "candidates": sorted(candidates + forced, key=lambda c: -c["score"]),
    }
    write_json_atomic(target.artifacts.highlights, document)  # marks the stage done
    return document


def theme_rank(candidate: dict[str, Any]) -> tuple[int, float]:
    """With --focus, the LLM is the only part that understands the theme, so its score
    decides; laughs, loudness and chat only break ties."""
    llm = candidate["llm_score"]
    return (llm if llm is not None else -1, candidate["score"])


def _evaluate(
    i: int,
    inputs: _Inputs,
    hc: HighlightsConfig,
    weights: dict[str, float],
    judge: OllamaJudge | None,
    window: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Window i of the grid, or `window` (start, end) when a stretch was asked for by hand."""
    segments = inputs.segments
    grid = (float(inputs.starts[i]), float(inputs.starts[i]) + hc.window_seconds)
    window_start, window_end = window or grid
    # A stretch you name yourself is used as is: the LLM only picks lines inside it.
    context = 0.0 if window else hc.context_seconds
    context_ids = segment_ids_between(segments, window_start - context, window_end + context)
    verdict: Verdict | None = None
    if judge and context_ids:
        excerpt = format_excerpt(segments, inputs.sounds, context_ids)
        verdict = judge.judge(
            excerpt, inputs.hint(i), context_ids,
            min_s=hc.min_clip_seconds, max_s=hc.max_clip_seconds,
            focus=inputs.focus.theme if inputs.focus else None,
            stream_title=inputs.title,
        )  # fmt: skip

    window_ids = segment_ids_between(segments, window_start, window_end)
    if window and window_ids:  # a stretch you asked for is the clip; the LLM only titles it
        first_id, last_id = window_ids[0], window_ids[-1]
    elif verdict:
        first_id, last_id = verdict.start_id, verdict.end_id
    elif window_ids:
        first_id, last_id = window_ids[0], window_ids[-1]
    else:
        first_id = last_id = None
    if first_id is not None and last_id is not None:
        start, end = clip_bounds(
            segments, first_id, last_id, inputs.laugh_times,
            min_s=hc.min_clip_seconds, max_s=hc.max_clip_seconds, duration_s=inputs.duration,
            theme_times=inputs.focus_times,
        )  # fmt: skip
    else:  # nobody talks here: the window itself
        start, end = window_start, min(window_end, inputs.duration)

    scores = {name: round(float(values[i]), 3) for name, values in inputs.scaled.items()}
    if verdict:
        scores["llm"] = verdict.score / 10
    covered = segment_ids_between(segments, start, end)
    still = inputs.still_fraction(start, end)
    # Theme words only build the shortlist. Literal matches can't tell "going to bed" from
    # "putting an enemy to sleep"; the LLM, told the theme, judges the fit instead.
    final = {name: value for name, value in scores.items() if name != "focus"}
    score = float(combine(final, weights)) * (1 - inputs.penalty * still)
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
        "score": round(score, 3),
        "scores": scores,
        "still_screen": round(still, 2),  # share of seconds where the gameplay barely moves
        "llm_score": verdict.score if verdict else None,
        "title": title,
        "hook": verdict.hook if verdict else "",
        "reason": reason,
        "laughs": inputs.count_between(inputs.laugh_times, start, end),
        "theme_words": inputs.count_between(inputs.focus_times, start, end),
        "chat_messages": inputs.count_between(inputs.chat_times, start, end),
        "loudness_spike_db": round(float(inputs.raw["audio"][i]), 1),
        "segment_ids": [covered[0], covered[-1]] if covered else None,
        "text": " ".join(segments[j]["text"] for j in covered),
    }
