"""List the channel's recent videos.

ytdlp: the channel's Live tab (free, no login, streams only). rss: the public feed (free,
but often fails). api: the Data API (needs OAuth, ~2 quota units).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx
import yt_dlp
from defusedxml import ElementTree
from yt_dlp.utils import DownloadError

from src.common.retry import with_retries
from src.common.ytdl import YtdlEnv, ytdl_base_options, ytdl_error_message

RSS_URL = "https://www.youtube.com/feeds/videos.xml"
_ATOM = "{http://www.w3.org/2005/Atom}"
_YT = "{http://www.youtube.com/xml/schemas/2015}"
_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")


class DiscoveryError(RuntimeError):
    """The channel's videos could not be listed."""


class FeedUnavailable(DiscoveryError):
    """The RSS feed kept answering 404."""


@dataclass(frozen=True)
class Candidate:
    video_id: str
    title: str
    is_stream: bool | None = None  # only the API knows this up front
    ended_at: datetime | None = None  # only the API knows this up front


def live_tab_url(channel_id: str) -> str:
    return f"https://www.youtube.com/channel/{channel_id}/streams"


def discover_ytdlp(
    channel_id: str, max_results: int, env: YtdlEnv | None = None
) -> list[Candidate]:
    """The channel's Live tab, newest first: past, live and upcoming streams."""
    try:
        info = _extract_tab(live_tab_url(channel_id), max_results, env)
    except DownloadError as exc:
        message = ytdl_error_message(exc)
        if "does not have a streams tab" in message.lower():
            return []  # the channel has never streamed
        raise DiscoveryError(f"Could not list the Live tab of {channel_id}: {message}") from exc
    return candidates_from_tab(info)


def candidates_from_tab(info: dict[str, Any]) -> list[Candidate]:
    candidates = []
    for entry in info.get("entries") or []:
        video_id = entry.get("id") if isinstance(entry, dict) else None
        if video_id and _VIDEO_ID.fullmatch(video_id):
            candidates.append(Candidate(video_id, entry.get("title") or ""))
    return candidates


@with_retries(attempts=4)
def _extract_tab(url: str, max_results: int, env: YtdlEnv | None) -> dict[str, Any]:
    # Flat listing: IDs and titles only, one request. Details come from fetch_metadata later.
    options = ytdl_base_options(env) | {"extract_flat": True, "playlistend": max_results}
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)
    if not isinstance(info, dict):
        raise DiscoveryError(f"yt-dlp returned nothing for {url}")
    return info


def parse_rss(xml: bytes) -> list[Candidate]:
    root = ElementTree.fromstring(xml)
    candidates = []
    for entry in root.iter(f"{_ATOM}entry"):
        video_id = entry.findtext(f"{_YT}videoId")
        if video_id:
            title = (entry.findtext(f"{_ATOM}title") or "").strip()
            candidates.append(Candidate(video_id, title))
    return candidates


def discover_rss(channel_id: str) -> list[Candidate]:
    """The feed lists the latest ~15 uploads, streams included."""
    return parse_rss(_get_rss(channel_id))


@with_retries(attempts=6, also_retry=(FeedUnavailable,))
def _get_rss(channel_id: str) -> bytes:
    response = httpx.get(
        RSS_URL, params={"channel_id": channel_id}, timeout=30, follow_redirects=True
    )
    if response.status_code == 404:
        # YouTube's feed answers 404 at random for valid channels, so retry it.
        # A wrong channel ID ends up here too, once the retries run out.
        raise FeedUnavailable(
            f"RSS feed kept returning 404 for {channel_id}. Check channel.id in config.yaml; "
            "if the ID is right, set trigger.method to ytdlp or api."
        )
    response.raise_for_status()
    return response.content


def uploads_playlist_id(channel_id: str) -> str:
    """Uploads playlist ID = channel ID with UC swapped for UU (saves a channels.list call)."""
    return "UU" + channel_id[2:]


def discover_api(youtube: Any, channel_id: str, max_results: int) -> list[Candidate]:
    """playlistItems.list + videos.list: 1 quota unit each."""
    playlist = _execute(
        youtube.playlistItems().list(
            part="contentDetails",
            playlistId=uploads_playlist_id(channel_id),
            maxResults=max_results,
        )
    )
    video_ids = [item["contentDetails"]["videoId"] for item in playlist.get("items", [])]
    if not video_ids:
        return []
    videos = _execute(
        youtube.videos().list(part="snippet,liveStreamingDetails", id=",".join(video_ids))
    )
    return [candidate_from_api(item) for item in videos.get("items", [])]


def candidate_from_api(item: dict[str, Any]) -> Candidate:
    details = item.get("liveStreamingDetails")
    ended = details.get("actualEndTime") if details else None
    return Candidate(
        video_id=item["id"],
        title=item["snippet"]["title"],
        is_stream=details is not None,
        ended_at=datetime.fromisoformat(ended) if ended else None,
    )


@with_retries()
def _execute(request: Any) -> dict[str, Any]:
    return request.execute(num_retries=0)
