"""YouTube's automatic captions: download (json3) and turn into words and sound tags."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import yt_dlp
from yt_dlp.utils import DownloadError

from src.common.retry import with_retries
from src.common.ytdl import YtdlEnv, video_url, ytdl_base_options, ytdl_error_message

_log = logging.getLogger("autoclip.transcribe")

_TAG = re.compile(r"\[([^\]]+)\]")  # [tertawa], [musik], [mendengus] ...
_CENSORED = "[ __ ]"  # YouTube's mask for some swear words
MAX_WORD_S = 1.0  # the json3 format has start times only; cap how long a word lasts


class CaptionsError(RuntimeError):
    """YouTube's captions could not be downloaded."""


def caption_key(automatic_captions: dict[str, Any], language: str | None) -> str | None:
    """The track YouTube's speech recognition produced, if it's in our language.

    '<lang>-orig' marks the original recognition; the other keys are machine translations of it.
    A different original language means YouTube misheard the stream's language: unusable."""
    originals = [key for key in automatic_captions if key.endswith("-orig")]
    if language is None:
        return originals[0] if originals else None
    if f"{language}-orig" in automatic_captions:
        return f"{language}-orig"
    if not originals and language in automatic_captions:
        return language
    return None


def fetch_captions(video_id: str, dest: Path, language: str | None, env: YtdlEnv) -> str | None:
    """Save the captions as json3 to dest. Returns the track used, or None if there's none (yet)."""
    try:
        info = _extract(video_id, ytdl_base_options(env) | {"skip_download": True}, False)
        key = caption_key(info.get("automatic_captions") or {}, language)
        if key is None:
            found = sorted(k for k in info.get("automatic_captions") or {} if k.endswith("-orig"))
            _log.info("No YouTube captions in %s for %s (original tracks: %s)",
                      language, video_id, found or "none")  # fmt: skip
            return None
        options = ytdl_base_options(env) | {
            "skip_download": True,
            "writeautomaticsub": True,
            "subtitleslangs": [key],
            "subtitlesformat": "json3",
            "outtmpl": {"default": str(dest.parent / "yt_captions_download.%(ext)s")},
        }
        info = _extract(video_id, options, True)
    except DownloadError as exc:
        raise CaptionsError(f"Caption download failed: {ytdl_error_message(exc)}") from exc
    track = (info.get("requested_subtitles") or {}).get(key) or {}
    path = Path(track.get("filepath") or dest.parent / f"yt_captions_download.{key}.json3")
    if not path.is_file():
        raise CaptionsError(f"yt-dlp reported captions at {path} but the file is missing")
    os.replace(path, dest)
    return key


@with_retries(attempts=4)
def _extract(video_id: str, options: dict[str, Any], download: bool) -> dict[str, Any]:
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(video_url(video_id), download=download, process=download)
    return info if isinstance(info, dict) else {}


def parse_json3(data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Words ({start, end, word, prob}) and sound tags ({t, tag}), in stream time, sorted."""
    tokens: list[tuple[float, float, str]] = []  # start, end of its caption line, text
    for event in data.get("events") or []:
        base_ms = event.get("tStartMs", 0)
        line_end = (base_ms + event.get("dDurationMs", 0)) / 1000
        for seg in event.get("segs") or []:
            text = seg.get("utf8", "").replace("\xa0", " ").strip()
            if text:
                tokens.append(((base_ms + seg.get("tOffsetMs", 0)) / 1000, line_end, text))
    tokens.sort(key=lambda token: token[0])

    words: list[dict[str, Any]] = []
    sounds: list[dict[str, Any]] = []
    for i, (start, line_end, text) in enumerate(tokens):
        if text == _CENSORED:
            text = "***"
        elif match := _TAG.fullmatch(text):
            sounds.append({"t": round(start, 3), "tag": match.group(1).strip()})
            continue
        next_start = tokens[i + 1][0] if i + 1 < len(tokens) else line_end
        end = max(min(next_start, line_end, start + MAX_WORD_S), start + 0.05)
        words.append({"start": round(start, 3), "end": round(end, 3), "word": text, "prob": None})
    return words, sounds


def clip_to_window(
    items: list[dict[str, Any]], key: str, offset_s: float, duration_s: float
) -> list[dict[str, Any]]:
    """Keep items whose `key` time falls inside vod.mp4 and make times relative to it."""
    clipped = []
    for item in items:
        if offset_s <= item[key] < offset_s + duration_s:
            moved = dict(item)
            for field in ("start", "end", "t"):
                if field in moved:
                    moved[field] = round(moved[field] - offset_s, 3)
            clipped.append(moved)
    return clipped
