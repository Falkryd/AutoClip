"""Turn yt-dlp's raw live chat replay (JSON lines) into a flat list of messages."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from typing import Any

# Chat item types worth counting. System notices, placeholders and polls are ignored.
_RENDERERS = {
    "liveChatTextMessageRenderer": "text",
    "liveChatPaidMessageRenderer": "superchat",
    "liveChatPaidStickerRenderer": "sticker",
    "liveChatMembershipItemRenderer": "membership",
}


@dataclass(frozen=True)
class ChatMessage:
    t: float  # seconds from the start of vod.mp4
    author: str
    text: str
    kind: str  # text, superchat, sticker or membership


def parse_chat_lines(lines: Iterable[str]) -> tuple[list[ChatMessage], int]:
    """Messages sorted by time, plus how many lines could not be read."""
    messages: list[ChatMessage] = []
    bad_lines = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            replay = json.loads(line).get("replayChatItemAction") or {}
            offset_ms = replay.get("videoOffsetTimeMsec")
            if offset_ms is None:
                continue
            t = int(offset_ms) / 1000
        except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
            bad_lines += 1
            continue
        for action in replay.get("actions") or []:
            message = _message(action, t)
            if message is not None:
                messages.append(message)
    messages.sort(key=lambda m: m.t)
    return messages, bad_lines


def clip_to_window(messages: list[ChatMessage], start_s: float, end_s: float) -> list[ChatMessage]:
    """Keep messages in [start, end) and make their times relative to start."""
    return [replace(m, t=round(m.t - start_s, 3)) for m in messages if start_s <= m.t < end_s]


def chat_document(
    video_id: str,
    messages: list[ChatMessage],
    *,
    available: bool,
    offset_s: float,
    duration_s: float,
    skipped_lines: int,
) -> dict[str, Any]:
    """Contents of chat.json, read by the highlights stage."""
    return {
        "video_id": video_id,
        "available": available,
        "offset_s": offset_s,  # where vod.mp4 starts in the full stream
        "duration_s": duration_s,
        "message_count": len(messages),
        "skipped_lines": skipped_lines,
        "messages": [asdict(m) for m in messages],
    }


def _message(action: dict[str, Any], t: float) -> ChatMessage | None:
    item = (action.get("addChatItemAction") or {}).get("item") or {}
    for key, kind in _RENDERERS.items():
        if key in item:
            renderer = item[key]
            return ChatMessage(t, _text(renderer.get("authorName")), _body(renderer), kind)
    return None


def _body(renderer: dict[str, Any]) -> str:
    text = _runs((renderer.get("message") or {}).get("runs"))
    if not text:  # memberships put their text in a header instead
        text = _text(renderer.get("headerSubtext"))
    amount = _text(renderer.get("purchaseAmountText"))
    return f"[{amount}] {text}".strip() if amount else text


def _text(obj: dict[str, Any] | None) -> str:
    if not obj:
        return ""
    return obj.get("simpleText") or _runs(obj.get("runs"))


def _runs(runs: list[dict[str, Any]] | None) -> str:
    parts = []
    for run in runs or []:
        if "text" in run:
            parts.append(run["text"])
        elif "emoji" in run:
            emoji = run["emoji"]
            shortcuts = emoji.get("shortcuts") or []
            parts.append(shortcuts[0] if shortcuts else emoji.get("emojiId", ""))
    return "".join(parts).strip()
