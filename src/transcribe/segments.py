"""Whisper segments to plain records, clean-up rules, and SRT output."""

from __future__ import annotations

import re
import zlib
from typing import Any, Protocol

from src.common.config import TranscribeConfig

# Phrases Whisper invents over music or noise, learned from subtitled videos. Matched against
# the whole segment, so a streamer saying them mid-sentence is kept.
_HALLUCINATIONS = re.compile(
    r"terima kasih (telah|sudah|karena) menonton( video ini)?"
    r"|thanks? (you )?for watching"
    r"|sampai jumpa (lagi )?di video (selanjutnya|berikutnya)"
    r"|(subtitles?|sub indo|terjemahan|teks) (by|oleh) .*"
)
_PUNCTUATION = re.compile(r"[^\w\s]")


class _Word(Protocol):
    start: float
    end: float
    word: str
    probability: float


class _Segment(Protocol):
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float
    words: list[_Word] | None


def segment_record(segment: _Segment) -> dict[str, Any]:
    return {
        "start": round(segment.start, 3),
        "end": round(segment.end, 3),
        "text": segment.text.strip(),
        "avg_logprob": round(segment.avg_logprob, 3),
        "no_speech_prob": round(segment.no_speech_prob, 3),
        "compression_ratio": round(segment.compression_ratio, 3),
        "words": [
            {
                "start": round(w.start, 3),
                "end": round(w.end, 3),
                "word": w.word.strip(),
                "prob": round(w.probability, 3),
            }
            for w in segment.words or []
        ],
    }


def remove_hallucinations(
    record: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Take invented sentences ("Terima kasih telah menonton!") out of a block. Runs before
    pause-splitting, which can cut such a sentence in halves that no longer match."""
    words = record["words"]
    if not words:
        return record, []
    kept_words: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    sentence: list[dict[str, Any]] = []
    for i, word in enumerate(words):
        sentence.append(word)
        if word["word"].endswith((".", "?", "!")) or i == len(words) - 1:
            text = " ".join(w["word"] for w in sentence)
            if _HALLUCINATIONS.fullmatch(_normalize(text)):
                dropped.append({"start": sentence[0]["start"], "end": sentence[-1]["end"],
                                "text": text, "reason": "known Whisper hallucination"})  # fmt: skip
            else:
                kept_words.extend(sentence)
            sentence = []
    return {**record, "words": kept_words}, dropped


def resegment(record: dict[str, Any], *, max_s: float, pause_s: float) -> list[dict[str, Any]]:
    """Batched Whisper returns blocks up to a few minutes long. Split them into sentence-sized
    pieces at sentence-ending punctuation, at pauses between words, and at max_s."""
    words = record["words"]
    if not words:
        return [record]
    pieces: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for word in words:
        if current and (
            word["start"] - current[-1]["end"] > pause_s
            or word["end"] - current[0]["start"] > max_s
        ):
            pieces.append(current)
            current = []
        current.append(word)
        if word["word"].endswith((".", "?", "!")):
            pieces.append(current)
            current = []
    if current:
        pieces.append(current)
    return [_piece(record, piece) for piece in pieces]


def _piece(record: dict[str, Any], words: list[dict[str, Any]]) -> dict[str, Any]:
    text = " ".join(w["word"] for w in words)
    return {
        **record,  # confidence scores are only known per block
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "text": text,
        "compression_ratio": round(compression_ratio(text), 3),
        "words": words,
    }


def compression_ratio(text: str) -> float:
    """How well the text compresses, as Whisper measures it: loops compress very well."""
    data = text.encode("utf-8")
    return len(data) / len(zlib.compress(data)) if data else 0.0


def drop_reason(record: dict[str, Any], cfg: TranscribeConfig) -> str | None:
    """Why a segment should be left out of the transcript, or None to keep it."""
    normalized = _normalize(record["text"])
    if not normalized and "***" not in record["text"]:  # *** is a censored word: real speech
        return "empty"
    if _HALLUCINATIONS.fullmatch(normalized):
        return "known Whisper hallucination"
    no_speech, logprob = record.get("no_speech_prob"), record.get("avg_logprob")
    # YouTube captions carry no confidence scores; only Whisper segments get this check.
    if (
        no_speech is not None
        and logprob is not None
        and no_speech > cfg.drop_no_speech_prob
        and logprob < cfg.drop_avg_logprob
    ):
        return "probably not speech"
    if (record.get("compression_ratio") or 0) > cfg.drop_compression_ratio:
        return "repetitive text (decoding loop)"
    return None


def split_segments(
    records: list[dict[str, Any]], cfg: TranscribeConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Kept segments get sequential ids, which later stages use to refer to them."""
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for record in records:
        reason = drop_reason(record, cfg)
        if reason is None and kept and _is_repeat(record, kept[-1]):
            reason = "repeated line (decoding loop)"
        if reason is None:
            kept.append({"id": len(kept), **record})
        else:
            dropped.append({"start": record["start"], "end": record["end"],
                            "text": record["text"], "reason": reason})  # fmt: skip
    return kept, dropped


def _is_repeat(record: dict[str, Any], previous: dict[str, Any]) -> bool:
    """Whisper loops can repeat a whole sentence back to back. Short repeats ("Aduh. Aduh.",
    "Let's go. Let's go.") are real, and YouTube's captions (no avg_logprob) don't loop."""
    if record.get("avg_logprob") is None or len(record["text"].split()) < 3:
        return False
    return _normalize(record["text"]) == _normalize(previous["text"])


def _normalize(text: str) -> str:
    return " ".join(_PUNCTUATION.sub(" ", text.lower()).split())


def to_srt(segments: list[dict[str, Any]]) -> str:
    blocks = [
        f"{i}\n{_srt_time(s['start'])} --> {_srt_time(s['end'])}\n{s['text']}\n"
        for i, s in enumerate(segments, start=1)
    ]
    return "\n".join(blocks)


def _srt_time(seconds: float) -> str:
    ms = round(seconds * 1000)
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
