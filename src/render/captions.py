"""Burned-in captions as an ASS subtitle file: a few words at a time, current word lit up."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import Any

from src.common.config import CaptionConfig, HookConfig

PAGE_GAP_S = 0.6  # a pause this long starts a new caption
LINGER_S = 0.3  # keep a caption up a moment after its last word
_BREAK_AFTER = (".", "?", "!", ",")


def caption_pages(words: list[dict[str, Any]], max_words: int) -> list[list[dict[str, Any]]]:
    """Group words into what's on screen together: at most max_words, broken at pauses and
    punctuation so a caption doesn't straddle two sentences."""
    pages: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for word in words:
        if current and (
            len(current) >= max_words
            or word["start"] - current[-1]["end"] > PAGE_GAP_S
            or current[-1]["word"].endswith(_BREAK_AFTER)
        ):
            pages.append(current)
            current = []
        current.append(word)
    if current:
        pages.append(current)
    return pages


def build_ass(
    words: list[dict[str, Any]],
    cc: CaptionConfig,
    width: int,
    height: int,
    *,
    hook: str | None = None,
    hook_style: HookConfig | None = None,
    duration_s: float = 0.0,
    low_spans: Sequence[tuple[float, float]] = (),
    low_y: float = 0.8,
) -> str:
    """words: {start, end, word} with times relative to the clip. hook: a line shown at the
    top for the whole clip (duration_s), in a box. low_spans: while these run (full-screen
    shots), captions sit at low_y instead, clear of the face or the action."""
    margin_v = round(height * (1 - cc.y))  # bottom-aligned: distance from the bottom edge
    low_margin_v = round(height * (1 - low_y))
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{cc.font},{cc.font_size},{ass_color(cc.text_color)},&H000000FF,"
        f"&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,{cc.outline},2,2,60,60,{margin_v},1",
        *([_hook_style(hook_style, cc.font, height)] if hook and hook_style else []),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    if hook and hook_style:
        lines.append(_dialogue(0.0, duration_s, _escape(hook), style="Hook"))

    def add(start: float, end: float, text: str) -> None:
        for a, b, low in split_at_spans(start, end, low_spans):
            lines.append(_dialogue(a, b, text, margin_v=low_margin_v if low else 0))

    highlight = _bgr(cc.highlight_color)
    pages = caption_pages(words, cc.max_words_per_line)
    for n, page in enumerate(pages):
        next_start = pages[n + 1][0]["start"] if n + 1 < len(pages) else float("inf")
        page_end = min(page[-1]["end"] + LINGER_S, next_start)
        texts = [_escape(w["word"]) for w in page]
        if cc.style == "sentence":
            add(page[0]["start"], page_end, " ".join(texts))
            continue
        for i, word in enumerate(page):
            start = word["start"]
            end = page[i + 1]["start"] if i + 1 < len(page) else page_end
            lit = [
                f"{{\\1c&H{highlight}&}}{t}{{\\r}}" if j == i else t for j, t in enumerate(texts)
            ]
            add(start, max(end, start + 0.05), " ".join(lit))
    return "\n".join(lines) + "\n"


def split_at_spans(
    start: float, end: float, spans: Sequence[tuple[float, float]]
) -> list[tuple[float, float, bool]]:
    """Cut [start, end) where spans begin and end: (from, to, inside a span) pieces."""
    cuts = sorted({start, end, *(t for s in spans for t in s if start < t < end)})
    return [(a, b, any(s0 <= a and b <= s1 for s0, s1 in spans)) for a, b in pairwise(cuts)]


def ass_color(hex_rgb: str) -> str:
    """'#RRGGBB' to ASS's style form '&H00BBGGRR' (blue first, alpha 00 = opaque)."""
    return f"&H00{_bgr(hex_rgb)}"


def _bgr(hex_rgb: str) -> str:
    rgb = hex_rgb.lstrip("#").upper()
    return f"{rgb[4:6]}{rgb[2:4]}{rgb[0:2]}"


def ass_time(seconds: float) -> str:
    centis = max(round(seconds * 100), 0)
    hours, centis = divmod(centis, 360_000)
    minutes, centis = divmod(centis, 6000)
    secs, centis = divmod(centis, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _dialogue(
    start: float, end: float, text: str, style: str = "Default", margin_v: int = 0
) -> str:
    """margin_v 0 keeps the style's height; anything else overrides it for this line."""
    return f"Dialogue: 0,{ass_time(start)},{ass_time(end)},{style},,0,0,{margin_v},,{text}"


def _hook_style(hs: HookConfig, font: str, height: int) -> str:
    """Top-centred text in an opaque box (BorderStyle 3: the box takes OutlineColour)."""
    alpha = f"{round((1 - hs.box_opacity) * 255):02X}"
    return (
        f"Style: Hook,{font},{hs.font_size},{ass_color(hs.text_color)},&H000000FF,"
        f"&H{alpha}{_bgr(hs.box_color)},&H00000000,-1,0,0,0,100,100,0,0,3,16,0,8,70,70,"
        f"{round(height * hs.y)},1"
    )


def _escape(text: str) -> str:
    """Braces and backslashes are ASS markup; transcripts shouldn't be able to inject it."""
    return text.replace("\\", "/").replace("{", "(").replace("}", ")")
