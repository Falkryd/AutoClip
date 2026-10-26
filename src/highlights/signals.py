"""Per-second signals over vod.mp4, and their scores per sliding window."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SPIKE_BASELINE_S = 60  # loudness is compared with the median of the surrounding minute
TOP_SECONDS = 5  # a window's loudness score: mean of its 5 biggest spikes


def loudness_per_second(wav_path: Path) -> np.ndarray:
    """RMS loudness in dBFS for each second of a 16-bit mono WAV, read a second at a time."""
    values = []
    with wave.open(str(wav_path), "rb") as fh:
        if fh.getsampwidth() != 2 or fh.getnchannels() != 1:
            raise ValueError(f"{wav_path} must be 16-bit mono (the download stage writes that)")
        rate = fh.getframerate()
        while frames := fh.readframes(rate):
            samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768
            rms = float(np.sqrt(np.mean(samples**2)))
            values.append(20 * np.log10(max(rms, 1e-6)))
    return np.array(values)


def loudness_spikes(db: np.ndarray, baseline_s: int = SPIKE_BASELINE_S) -> np.ndarray:
    """How far each second rises above the median of the surrounding minute, in dB (>= 0).
    Relative, so a loud teamfight with no reaction scores less than you suddenly shouting."""
    if len(db) == 0:
        return db
    half = baseline_s // 2
    padded = np.pad(db, half, mode="edge")
    baseline = np.median(np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1), axis=1)
    return np.clip(db - baseline, 0, None)


def events_per_second(times: list[float], duration_s: float) -> np.ndarray:
    counts = np.zeros(int(np.ceil(duration_s)))
    for t in times:
        if 0 <= t < len(counts):
            counts[int(t)] += 1
    return counts


def window_starts(duration_s: float, window_s: float, step_s: float) -> np.ndarray:
    return np.arange(0, max(duration_s - window_s, 0) + 1e-9, step_s)


def window_sum(per_second: np.ndarray, starts: np.ndarray, window_s: float) -> np.ndarray:
    cumulative = np.concatenate([[0.0], np.cumsum(per_second)])
    first = np.clip(starts.astype(int), 0, len(per_second))
    last = np.clip((starts + window_s).astype(int), 0, len(per_second))
    return cumulative[last] - cumulative[first]


def window_top_mean(
    per_second: np.ndarray, starts: np.ndarray, window_s: float, top: int = TOP_SECONDS
) -> np.ndarray:
    scores = []
    for start in starts:
        part = per_second[int(start) : int(start + window_s)]
        scores.append(float(np.mean(np.sort(part)[-top:])) if len(part) else 0.0)
    return np.array(scores)


def scale(values: np.ndarray) -> np.ndarray | None:
    """0-1 relative to the 99th percentile (the maximum if that's 0). None if there's nothing
    to rank, e.g. no laughs at all: the signal is then left out instead of counting as zero."""
    if len(values) == 0:
        return None
    reference = float(np.percentile(values, 99)) or float(np.max(values))
    if reference <= 0:
        return None
    return np.clip(values / reference, 0, 1)


def speech_per_second(segments: list[dict], duration_s: float) -> np.ndarray:
    """True for seconds covered by a transcribed word."""
    speech = np.zeros(int(np.ceil(duration_s)), dtype=bool)
    for segment in segments:
        for word in segment["words"]:
            speech[int(word["start"]) : int(np.ceil(word["end"]))] = True
    return speech
