"""From window scores to candidates, and from candidates to the final clips."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

import numpy as np

from src.common.timecode import format_timecode

LEAD_S = 0.3  # start a little before the first word
TAIL_S = 0.7  # and let the last word ring out
LAUGH_REACH_S = 3.0  # a laugh this soon after the last line is the payoff: keep it
LAUGH_HOLD_S = 1.5
LOUD_DB = 6.0  # "loud" for the music check: this far above the stream's median loudness

T = TypeVar("T", float, np.ndarray)


def combine(scores: Mapping[str, T], weights: Mapping[str, float]) -> T:
    """Weighted mean over the signals present. A missing signal (no chat replay, no laugh
    tags) is left out and the remaining weights are rescaled, instead of counting as zero."""
    active = {name: w for name, w in weights.items() if name in scores and w > 0}
    total = sum(active.values())
    if total <= 0:
        raise ValueError("No signal available to score with")
    return sum(scores[name] * w for name, w in active.items()) / total  # type: ignore[return-value]


def shortlist(starts: np.ndarray, scores: np.ndarray, window_s: float, size: int) -> list[int]:
    """Indexes of the best windows, highest first, none overlapping another."""
    picked: list[int] = []
    for i in np.argsort(-scores, kind="stable"):
        if all(abs(starts[i] - starts[j]) >= window_s for j in picked):
            picked.append(int(i))
            if len(picked) == size:
                break
    return picked


def segment_ids_between(segments: list[dict[str, Any]], start_s: float, end_s: float) -> list[int]:
    return [s["id"] for s in segments if s["end"] > start_s and s["start"] < end_s]


def clip_bounds(
    segments: list[dict[str, Any]],
    first_id: int,
    last_id: int,
    laugh_times: list[float],
    *,
    min_s: float,
    max_s: float,
    duration_s: float,
) -> tuple[float, float]:
    """Whole sentences from first_id to last_id, fitted to min_s..max_s."""

    def span(first: int, last: int) -> float:
        return segments[last]["end"] + TAIL_S - (segments[first]["start"] - LEAD_S)

    first, last = first_id, last_id
    while last > first and span(first, last) > max_s:
        last -= 1
    while span(first, last) < min_s and last + 1 < len(segments) and span(first, last + 1) <= max_s:
        last += 1
    while span(first, last) < min_s and first > 0 and span(first - 1, last) <= max_s:
        first -= 1
    start = max(0.0, segments[first]["start"] - LEAD_S)
    end = min(duration_s, segments[last]["end"] + TAIL_S)

    trailing = [t for t in laugh_times if end < t <= end + LAUGH_REACH_S]
    if trailing:
        end = min(max(trailing) + LAUGH_HOLD_S, start + max_s, duration_s)
    # Still outside the range (few or very long sentences): pad or cut in plain time.
    if end - start < min_s:
        end = min(duration_s, start + min_s)
        start = max(0.0, end - min_s)
    end = min(end, start + max_s)
    return round(start, 2), round(end, 2)


def pick_final(
    candidates: list[dict[str, Any]],
    count: int,
    min_gap_s: float,
    start_with: Sequence[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """start_with (clips asked for by hand) first, then the best-scoring candidates, skipping
    any within min_gap_s of one already picked."""
    chosen: list[dict[str, Any]] = list(start_with)[:count]
    if len(chosen) == count:
        return chosen
    for candidate in sorted(candidates, key=lambda c: -c["score"]):
        if all(
            candidate["start"] >= other["end"] + min_gap_s
            or candidate["end"] <= other["start"] - min_gap_s
            for other in chosen
        ):
            chosen.append(candidate)
            if len(chosen) == count:
                break
    return chosen


def music_flags(
    start: float,
    end: float,
    sounds: list[dict[str, Any]],
    loudness_db: np.ndarray,
    speech: np.ndarray,
    flag_after_s: float,
) -> list[str]:
    """Warnings that a clip may contain music (copyright claims), for the review message."""
    flags = []
    tagged = [
        s["t"] for s in sounds if s["tag"].lower() in ("musik", "music") and start <= s["t"] <= end
    ]
    if tagged:
        times = ", ".join(format_timecode(t) for t in tagged[:3])
        flags.append(f"YouTube tagged music at {times}: check for copyright claims")
    if flag_after_s > 0 and len(loudness_db):
        threshold = float(np.median(loudness_db)) + LOUD_DB
        seconds = range(int(start), min(int(end), len(loudness_db)))
        loud_without_speech = sum(
            1 for t in seconds if loudness_db[t] > threshold and not (t < len(speech) and speech[t])
        )
        if loud_without_speech >= flag_after_s:
            flags.append(
                f"{loud_without_speech} s of loud audio without speech (music?): "
                "check for copyright claims"
            )
    return flags
