"""Full-screen shots: on a clip's big moments, cut from the stacked layout to the game or to
the avatar. The hero dying and bursts of action go to the game; laughs and shouts go to the
avatar, so a clip reads as cause, then reaction."""

from __future__ import annotations

import wave
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Literal

import numpy as np

from src.common.ffmpeg import raw_frames

View = Literal["game", "face"]

SAMPLE_FPS = 5  # the screen and the voice are looked at this often
SCREEN_SIZE = (36, 64)  # tiny 9:16 frames are enough to see colour and movement

# How long a shot runs around its moment (seconds before, after). Earlier kinds win overlaps.
AROUND: dict[str, tuple[float, float]] = {
    "death": (1.5, 1.0),
    "laugh": (0.3, 1.7),
    "shout": (0.2, 1.5),
    "action": (1.0, 1.0),
}
VIEW_OF: dict[str, View] = {"death": "game", "laugh": "face", "shout": "face", "action": "game"}
LEAD_IN_S = 1.0  # the clip opens on the whole layout
MIN_SHOT_S = 0.8
MAX_SHOT_S = 4.0
MERGE_GAP_S = 1.0  # two shots of the same view this close become one
BRIDGE_GAP_S = 0.6  # a flash of the stacked view this short between two shots is skipped
# Dota greys the world out while your hero is dead: colour drops, brightness holds. Opening
# the shop or the scoreboard and end-of-game fades also drain colour, but darken the screen.
DEATH_SATURATION = 0.72  # colour falls to this share of the 2 s before...
DEATH_BRIGHTNESS = 0.95  # ...while brightness stays at least this share
DEATH_MIN_BRIGHTNESS = 40.0  # and the screen before was the game, not a black loading screen
ACTION_FACTOR = 2.0  # a fight: movement this many times the clip's typical...
ACTION_MIN = 12.0  # ...and at least this much (mean pixel change, 0-255)


@dataclass(frozen=True)
class Moment:
    at: float  # when it happens, seconds into the clip
    kind: str  # a key of AROUND
    start: float | None = None  # the stretch worth showing, when it isn't just around `at`
    end: float | None = None


@dataclass(frozen=True)
class Shot:
    start: float
    end: float
    view: View
    reason: str


def plan_shots(
    moments: list[Moment], duration_s: float, max_fraction: float, views: set[str]
) -> list[Shot]:
    """Pick shots for the moments, most important kind first, until max_fraction of the clip
    is used. A shot never covers another view's shot; it takes the free part next to it."""
    budget = max_fraction * duration_s
    shots: list[Shot] = []
    for kind in AROUND:
        for m in sorted((m for m in moments if m.kind == kind), key=lambda m: m.at):
            view = VIEW_OF[kind]
            if view not in views:
                continue
            before, after = AROUND[kind]
            start = m.start if m.start is not None else m.at - before
            end = m.end if m.end is not None else m.at + after
            span = _free_part(
                (max(start, LEAD_IN_S), min(end, duration_s)),
                [(s.start, s.end) for s in shots if s.view != view],
                m.at,
            )
            if span is None:
                continue
            same = [s for s in shots if s.view == view and _near(span, s)]
            merged = (
                min([span[0], *(s.start for s in same)]),
                max([span[1], *(s.end for s in same)]),
            )
            others = [(s.start, s.end) for s in shots if s.view != view]
            if merged[1] - merged[0] > MAX_SHOT_S or any(_overlaps(merged, o) for o in others):
                same, merged = [], span
                if any(_overlaps(span, (s.start, s.end)) for s in shots):
                    continue
            added = (merged[1] - merged[0]) - sum(s.end - s.start for s in same)
            if _used(shots) + added > budget:
                continue
            reason = "+".join(dict.fromkeys([*(s.reason for s in same), kind]))
            shots = [s for s in shots if s not in same]
            shots.append(Shot(round(merged[0], 3), round(merged[1], 3), view, reason))
    return _bridge(sorted(shots, key=lambda s: s.start))


def manual_shots(spec: list[dict[str, object]], duration_s: float) -> list[Shot]:
    """Shots written by hand: [{"start": 3.2, "end": 5.0, "view": "face"}, ...], times as seen
    in the rendered clip. An empty list means no shots for that clip."""
    shots = []
    for item in spec:
        view = item["view"]
        if view not in VIEW_OF.values():
            raise ValueError(f'view must be "game" or "face", not {view!r}')
        start, end = float(item["start"]), min(float(item["end"]), duration_s)  # type: ignore[arg-type]
        if not 0 <= start < end:
            raise ValueError(f"{item['start']}-{item['end']} is not a stretch of the clip")
        shots.append(Shot(start, end, view, str(item.get("reason", "manual"))))  # type: ignore[arg-type]
    shots.sort(key=lambda s: s.start)
    for view in VIEW_OF.values():
        same = [s for s in shots if s.view == view]
        for a, b in pairwise(same):
            if b.start < a.end:
                raise ValueError(f"two {view} shots overlap at {b.start} s")
    return shots


def _free_part(
    span: tuple[float, float], taken: list[tuple[float, float]], at: float
) -> tuple[float, float] | None:
    """The part of span outside `taken`: the piece holding `at`, else the longest piece."""
    pieces = [span]
    for a, b in taken:
        pieces = [p for x, y in pieces for p in ((x, min(y, a)), (max(x, b), y)) if p[1] > p[0]]
    pieces = [p for p in pieces if p[1] - p[0] >= MIN_SHOT_S]
    if not pieces:
        return None
    holding = [p for p in pieces if p[0] <= at <= p[1]]
    return holding[0] if holding else max(pieces, key=lambda p: p[1] - p[0])


def _near(span: tuple[float, float], shot: Shot) -> bool:
    return span[0] - MERGE_GAP_S <= shot.end and shot.start - MERGE_GAP_S <= span[1]


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _used(shots: list[Shot]) -> float:
    return sum(s.end - s.start for s in shots)


def _bridge(shots: list[Shot]) -> list[Shot]:
    """Close short gaps between shots of different views: cause runs straight into reaction."""
    out = list(shots)
    for i in range(len(out) - 1):
        gap = out[i + 1].start - out[i].end
        if 0 < gap < BRIDGE_GAP_S and out[i].view != out[i + 1].view:
            s = out[i]
            out[i] = Shot(s.start, out[i + 1].start, s.view, s.reason)
    return out


# --- what happens on screen and in the voice ------------------------------------------------


def measure_screen(
    vod: Path, start_s: float, duration_s: float, crop: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per sample: movement, colour and brightness of the `crop` (an FFmpeg crop=w:h:x:y)."""
    w, h = SCREEN_SIZE
    data = raw_frames(vod, f"fps={SAMPLE_FPS},crop={crop},scale={w}:{h},format=rgb24",
                      start_s=start_s, duration_s=duration_s)  # fmt: skip
    frames = np.frombuffer(data, np.uint8).reshape(-1, h, w, 3).astype(np.float32)
    if len(frames) == 0:
        empty = np.zeros(0)
        return empty, empty, empty
    gray = frames.mean(axis=3)
    motion = np.r_[0.0, np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2))]
    saturation = (frames.max(axis=3) - frames.min(axis=3)).mean(axis=(1, 2))
    return motion, saturation, gray.mean(axis=(1, 2))


def measure_voice(wav_path: Path, start_s: float, duration_s: float) -> np.ndarray:
    """Loudness in dBFS per sample of a 16-bit mono WAV, over one stretch."""
    with wave.open(str(wav_path), "rb") as fh:
        if fh.getsampwidth() != 2 or fh.getnchannels() != 1:
            raise ValueError(f"{wav_path} must be 16-bit mono (the download stage writes that)")
        rate = fh.getframerate()
        fh.setpos(min(int(start_s * rate), fh.getnframes()))
        pcm = np.frombuffer(fh.readframes(int(duration_s * rate)), dtype="<i2")
    step = rate // SAMPLE_FPS
    n = len(pcm) // step
    blocks = pcm[: n * step].astype(np.float32).reshape(n, step) / 32768
    return 10 * np.log10(np.maximum((blocks**2).mean(axis=1), 1e-12))


def deaths(saturation: np.ndarray, brightness: np.ndarray) -> list[float]:
    """When the screen turns grey but not dark: the hero just died."""
    before, after = 2 * SAMPLE_FPS, 8
    found: list[float] = []
    i = before
    while i <= len(saturation) - after:
        sat_before = float(np.median(saturation[i - before : i]))
        grey = saturation[i : i + after]
        bright_before = float(np.median(brightness[i - before : i]))
        bright_after = float(np.median(brightness[i : i + after]))
        if (
            np.median(grey) <= DEATH_SATURATION * sat_before
            and bright_before >= DEATH_MIN_BRIGHTNESS
            and bright_after >= DEATH_BRIGHTNESS * bright_before
        ):
            onset = i + int(np.argmax(grey <= DEATH_SATURATION * sat_before))
            found.append(round(onset / SAMPLE_FPS, 2))
            i = onset + 5 * SAMPLE_FPS  # one death at a time
        else:
            i += 1
    return found


def action(motion: np.ndarray) -> list[Moment]:
    """The busiest stretches on screen (fights, spells going off), at least a second long."""
    if len(motion) < SAMPLE_FPS:
        return []
    smooth = np.convolve(motion, np.ones(SAMPLE_FPS) / SAMPLE_FPS, mode="same")
    busy = smooth >= max(ACTION_FACTOR * float(np.median(smooth)), ACTION_MIN)
    moments = []
    for a, b in _runs(busy):
        if b - a >= SAMPLE_FPS:
            peak = a + int(np.argmax(smooth[a:b]))
            moments.append(Moment(peak / SAMPLE_FPS, "action", a / SAMPLE_FPS, b / SAMPLE_FPS))
    return moments


def shouts(db: np.ndarray, threshold_db: float) -> list[float]:
    """When the voice track jumps threshold_db over the clip's median for 0.4 s."""
    if len(db) < 2:
        return []
    loud = np.convolve(db - np.median(db), np.ones(2) / 2, mode="valid") >= threshold_db
    found: list[float] = []
    for a, _ in _runs(loud):
        t = a / SAMPLE_FPS
        if not found or t - found[-1] >= 2.0:
            found.append(round(t, 2))
    return found


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of the True stretches."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))
