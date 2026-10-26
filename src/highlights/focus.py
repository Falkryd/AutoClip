"""--focus: a theme to favour, such as "ga tidur sebelum menang"."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Words too common to say anything about a theme (Indonesian and English).
_STOPWORDS = {
    "yang", "dan", "atau", "sebelum", "sesudah", "setelah", "sampai", "sama", "dari", "untuk",
    "dengan", "tidak", "nggak", "enggak", "kagak", "ini", "itu", "aja", "juga", "lagi",
    "the", "and", "before", "after", "until", "with", "about", "this", "that",
}  # fmt: skip
_NOT_WORD = re.compile(r"[^\w]")


@dataclass(frozen=True)
class Focus:
    theme: str  # told to the LLM as is
    words: tuple[str, ...]  # searched in the transcript


def make_focus(theme: str, words: str | None = None) -> Focus:
    """Theme words come from `words` (comma-separated) or else from the theme itself."""
    if words:
        found = [w.strip().lower() for w in words.split(",") if w.strip()]
    else:
        found = [
            w for w in re.findall(r"\w+", theme.lower()) if len(w) >= 4 and w not in _STOPWORDS
        ]
    if not found:
        raise ValueError(f"No words to search for in {theme!r}; name some with --focus-words")
    return Focus(theme.strip(), tuple(dict.fromkeys(found)))


def focus_times(segments: list[dict[str, Any]], focus: Focus) -> list[float]:
    """Start times of transcript words containing a theme word: Indonesian adds prefixes and
    suffixes, so "menang" should also find "kemenangan" and "tidur" "tidurnya"."""
    return [
        word["start"]
        for segment in segments
        for word in segment["words"]
        if any(theme_word in _NOT_WORD.sub("", word["word"].lower()) for theme_word in focus.words)
    ]
