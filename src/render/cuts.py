"""Jump cuts: drop the pauses between lines so a clip reaches its payoff sooner."""

from __future__ import annotations

from typing import Any

PAD_S = 0.25  # keep this much around every word
LAUGH_BEFORE_S = 0.5  # and around every laugh tag (laughs run on after their tag)
LAUGH_AFTER_S = 1.5


def keep_intervals(
    words: list[dict[str, Any]],
    laugh_times: list[float],
    duration_s: float,
    max_pause_s: float,
) -> list[tuple[float, float]]:
    """Clip-relative stretches to keep: speech and laughter, merged across pauses no longer
    than max_pause_s. Everything else is cut. A clip with no speech is kept whole."""
    spans = [(w["start"] - PAD_S, w["end"] + PAD_S) for w in words]
    spans += [(t - LAUGH_BEFORE_S, t + LAUGH_AFTER_S) for t in laugh_times]
    spans = sorted((max(0.0, a), min(duration_s, b)) for a, b in spans if b > 0 and a < duration_s)
    if not spans:
        return [(0.0, duration_s)]
    merged: list[list[float]] = []
    for start, end in spans:
        if merged and start - merged[-1][1] <= max_pause_s:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(round(a, 3), round(b, 3)) for a, b in merged]


def cut_time(t: float, intervals: list[tuple[float, float]]) -> float:
    """Where clip time t lands once the gaps are removed."""
    return round(sum(max(0.0, min(end, t) - start) for start, end in intervals), 3)


def kept_duration(intervals: list[tuple[float, float]]) -> float:
    return round(sum(end - start for start, end in intervals), 3)


def select_expression(intervals: list[tuple[float, float]]) -> str:
    """FFmpeg select/aselect expression keeping those stretches (t is clip time)."""
    terms = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in intervals)
    return f"'{terms}'"  # quoted: the commas inside must not split the filter graph


def remap_words(
    words: list[dict[str, Any]], intervals: list[tuple[float, float]]
) -> list[dict[str, Any]]:
    """Caption words moved onto the cut timeline."""
    return [
        {**w, "start": cut_time(w["start"], intervals), "end": cut_time(w["end"], intervals)}
        for w in words
    ]
