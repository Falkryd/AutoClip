"""Convert between H:MM:SS timecodes and seconds."""

from __future__ import annotations


def parse_timecode(text: str) -> float:
    """'2:33:00', '33:00', '90' or '90.5' -> seconds."""
    parts = text.strip().split(":")
    if not 1 <= len(parts) <= 3 or any(part.strip() == "" for part in parts):
        raise ValueError(f"Not a timecode: {text!r}")
    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        raise ValueError(f"Not a timecode: {text!r}") from None
    if any(n < 0 for n in numbers) or any(n >= 60 for n in numbers[1:]):
        raise ValueError(f"Not a timecode: {text!r}")
    seconds = 0.0
    for number in numbers:
        seconds = seconds * 60 + number
    return seconds


def format_timecode(seconds: float) -> str:
    """Seconds -> 'H:MM:SS' (whole seconds)."""
    total = int(seconds)
    return f"{total // 3600}:{total % 3600 // 60:02d}:{total % 60:02d}"


def compact_timecode(seconds: float) -> str:
    """Seconds -> 'HHMMSS', safe for folder names."""
    total = int(seconds)
    return f"{total // 3600:02d}{total % 3600 // 60:02d}{total % 60:02d}"
