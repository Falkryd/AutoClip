"""List the channel's recent videos: public RSS feed (free) or Data API (~2 quota units)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx
from defusedxml import ElementTree

from src.common.retry import with_retries

RSS_URL = "https://www.youtube.com/feeds/videos.xml"
_ATOM = "{http://www.w3.org/2005/Atom}"
_YT = "{http://www.youtube.com/xml/schemas/2015}"


@dataclass(frozen=True)
class Candidate:
    video_id: str
    title: str
    is_stream: bool | None = None  # only the API knows this up front
    ended_at: datetime | None = None  # only the API knows this up front


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


class FeedUnavailable(RuntimeError):
    """The RSS feed kept answering 404."""


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
            "if the ID is right and this keeps happening, set trigger.method to api."
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
