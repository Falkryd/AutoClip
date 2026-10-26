from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.common.config import AppConfig, TranscribeConfig
from src.common.files import read_json, write_json_atomic
from src.common.paths import Target, artifacts, resolve_target
from src.transcribe import core
from src.transcribe.segments import (
    compression_ratio,
    drop_reason,
    remove_hallucinations,
    resegment,
    segment_record,
    split_segments,
    to_srt,
)
from src.transcribe.whisper import WhisperResult
from src.transcribe.youtube import caption_key, clip_to_window, parse_json3
from tests.conftest import FIXTURES

CFG = TranscribeConfig()


def record(text: str, *, lp: float = -0.5, ns: float = 0.1, cr: float = 1.5) -> dict[str, Any]:
    return {"start": 1.0, "end": 2.0, "text": text, "avg_logprob": lp, "no_speech_prob": ns,
            "compression_ratio": cr, "words": []}  # fmt: skip


def test_segment_record_rounds_and_strips() -> None:
    word = SimpleNamespace(start=10.4123, end=10.9, word=" adu", probability=0.81234)
    segment = SimpleNamespace(
        start=10.4123, end=11.7, text=" adu-adu ", avg_logprob=-0.8312,
        no_speech_prob=0.1912, compression_ratio=0.8, words=[word],
    )  # fmt: skip
    rec = segment_record(segment)
    assert rec["text"] == "adu-adu"
    assert rec["start"] == 10.412 and rec["avg_logprob"] == -0.831
    assert rec["words"] == [{"start": 10.412, "end": 10.9, "word": "adu", "prob": 0.812}]


@pytest.mark.parametrize(
    "text",
    [
        "Terima kasih telah menonton!",
        "terima kasih sudah menonton video ini.",
        "Thanks for watching!",
        "Thank you for watching.",
        "Subtitle by Amara.org",
        "Sampai jumpa di video selanjutnya",
    ],
)
def test_known_hallucinations_are_dropped_even_when_confident(text: str) -> None:
    assert drop_reason(record(text, lp=-0.2, ns=0.05), CFG) == "known Whisper hallucination"


def test_hallucination_phrase_inside_real_speech_is_kept() -> None:
    text = "oke guys terima kasih telah menonton stream hari ini, besok main lagi"
    assert drop_reason(record(text), CFG) is None


def test_low_confidence_non_speech_is_dropped() -> None:
    assert drop_reason(record("hmm", lp=-1.3, ns=0.8), CFG) == "probably not speech"
    # only one of the two signals is not enough
    assert drop_reason(record("ah ngebutuh gua", lp=-0.95, ns=0.62), CFG) is None
    assert drop_reason(record("hmm", lp=-1.3, ns=0.2), CFG) is None


def test_repetition_loops_and_empty_text_are_dropped() -> None:
    assert drop_reason(record("ya ya ya ya ya ya ya ya", cr=3.1), CFG).startswith("repetitive")
    assert drop_reason(record(" ... "), CFG) == "empty"


def test_split_segments_renumbers_kept_segments() -> None:
    kept, dropped = split_segments(
        [record("satu"), record("Terima kasih telah menonton!"), record("dua")], CFG
    )
    assert [(s["id"], s["text"]) for s in kept] == [(0, "satu"), (1, "dua")]
    assert dropped == [{"start": 1.0, "end": 2.0, "text": "Terima kasih telah menonton!",
                        "reason": "known Whisper hallucination"}]  # fmt: skip


def test_to_srt() -> None:
    segments = [
        {"start": 0.0, "end": 1.5, "text": "halo semua"},
        {"start": 3661.25, "end": 3662.0, "text": "gg"},
    ]
    assert to_srt(segments) == (
        "1\n00:00:00,000 --> 00:00:01,500\nhalo semua\n\n2\n01:01:01,250 --> 01:01:02,000\ngg\n"
    )


def test_resolve_target_video_id_and_url(cfg: AppConfig) -> None:
    for text in ("8fKUNhHQR_M", "https://youtube.com/live/8fKUNhHQR_M"):
        target = resolve_target(cfg, text)
        assert target.video_id == "8fKUNhHQR_M"
        assert target.artifacts == artifacts(cfg, "8fKUNhHQR_M")
        assert target.is_pipeline_dir


def test_resolve_target_sample_folder(cfg: AppConfig, tmp_path: Path) -> None:
    sample = tmp_path / "8fKUNhHQR_M_023300-033300"
    write_json_atomic(sample / "download.json", {"video_id": "8fKUNhHQR_M"})
    target = resolve_target(cfg, str(sample))
    assert target.video_id == "8fKUNhHQR_M"
    assert target.artifacts.root == sample.resolve()
    assert not target.is_pipeline_dir  # samples are never recorded in the state database


def test_resolve_target_work_folder_counts_as_pipeline(cfg: AppConfig) -> None:
    work = artifacts(cfg, "8fKUNhHQR_M").root
    work.mkdir(parents=True)
    target = resolve_target(cfg, str(work))
    assert target.video_id == "8fKUNhHQR_M" and target.is_pipeline_dir


def test_resolve_target_rejects_missing_folder(cfg: AppConfig, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        resolve_target(cfg, str(tmp_path / "nope"))


# --- sentence splitting ---------------------------------------------------------------------


def words(*items: tuple[float, float, str]) -> list[dict[str, Any]]:
    return [{"start": s, "end": e, "word": w, "prob": 0.9} for s, e, w in items]


def block(ws: list[dict[str, Any]]) -> dict[str, Any]:
    return {"start": ws[0]["start"], "end": ws[-1]["end"], "text": "x", "avg_logprob": -0.5,
            "no_speech_prob": 0.1, "compression_ratio": 1.0, "words": ws}  # fmt: skip


def test_resegment_splits_at_punctuation_pauses_and_max_length() -> None:
    ws = words(
        (0.0, 0.3, "Aduh."), (0.4, 0.7, "Krono"), (0.7, 1.0, "lagi"), (1.0, 1.3, "kan?"),
        (3.0, 3.3, "gak"), (3.3, 3.6, "butuh"),  # 1.7 s pause before this piece
        *[(5.0 + i, 5.9 + i, f"kata{i}") for i in range(14)],  # pause, then no pauses: max cuts
    )  # fmt: skip
    pieces = resegment(block(ws), max_s=12, pause_s=0.8)
    assert [p["text"] for p in pieces][:3] == ["Aduh.", "Krono lagi kan?", "gak butuh"]
    assert all(p["end"] - p["start"] <= 12 for p in pieces)
    assert (pieces[1]["start"], pieces[1]["end"]) == (0.4, 1.3)
    assert pieces[1]["avg_logprob"] == -0.5  # confidence is inherited from the block


def test_resegment_keeps_blocks_without_word_timings() -> None:
    rec = {**block(words((0, 1, "x"))), "words": []}
    assert resegment(rec, max_s=12, pause_s=0.8) == [rec]


def test_loop_inside_a_block_drops_only_the_looping_piece() -> None:
    real = [(0.0, 0.4, "tapi"), (0.4, 0.8, "aegisnya"), (0.8, 1.2, "udah"), (1.2, 1.6, "kepake"),
            (1.6, 2.0, "kan?")]  # fmt: skip
    loop = [(3.0 + i * 0.5, 3.4 + i * 0.5, w)
            for i, w in enumerate(["kagak", "pernah", "ngeliat", "dia", "anjir"] * 4)]  # fmt: skip
    pieces = resegment(block(words(*real, *loop)), max_s=12, pause_s=0.8)
    kept, dropped = split_segments(pieces, CFG)
    assert [s["text"] for s in kept] == ["tapi aegisnya udah kepake kan?"]
    assert dropped[0]["reason"].startswith("repetitive")


def test_compression_ratio_flags_repetition() -> None:
    assert compression_ratio("dihidupkan lagi " * 8) > 2.4
    assert compression_ratio("gak ngaruh lu yang mati bangsat gua punya heal") < 2.4
    assert compression_ratio("") == 0.0


def test_captions_without_confidence_are_not_dropped_as_non_speech() -> None:
    rec = {**record("halo semua"), "avg_logprob": None, "no_speech_prob": None,
           "compression_ratio": None}  # fmt: skip
    assert drop_reason(rec, CFG) is None


# --- YouTube captions -----------------------------------------------------------------------


def fixture_captions() -> dict[str, Any]:
    return read_json(FIXTURES / "captions.json3")


def test_parse_json3_words_sounds_and_censoring() -> None:
    ws, sounds = parse_json3(fixture_captions())
    assert [w["word"] for w in ws] == [
        "sebelum", "halo", "semua", "gimana", "kabarnya?", "lu", "***", "anjing", "mantap",
        "sesudah",
    ]  # fmt: skip
    by_word = {w["word"]: (w["start"], w["end"]) for w in ws}
    assert by_word["halo"] == (101.0, 101.4)  # ends where the next word starts
    assert by_word["kabarnya?"] == (102.3, 103.3)  # capped at 1 s
    assert by_word["anjing"] == (107.7, 108.5)  # ends at the [musik] tag
    assert by_word["mantap"] == (111.0, 112.0)  # ends with its caption line
    assert sounds == [{"t": 104.5, "tag": "tertawa"}, {"t": 108.5, "tag": "musik"},
                      {"t": 170.5, "tag": "tertawa"}]  # fmt: skip


def test_clip_to_window_shifts_into_sample_time() -> None:
    ws, sounds = parse_json3(fixture_captions())
    clipped = clip_to_window(ws, "start", 100.0, 60.0)
    assert (clipped[0]["word"], clipped[0]["start"]) == ("halo", 1.0)
    assert not {"sebelum", "sesudah"} & {w["word"] for w in clipped}
    assert clip_to_window(sounds, "t", 100.0, 60.0) == [{"t": 4.5, "tag": "tertawa"},
                                                          {"t": 8.5, "tag": "musik"}]  # fmt: skip


@pytest.mark.parametrize(
    ("keys", "language", "expected"),
    [
        (["id-orig", "id", "en", "ja"], "id", "id-orig"),
        (["id", "en"], "id", "id"),  # older listings without an -orig marker
        (["en-orig", "id", "en"], "id", None),  # YouTube misheard the language: unusable
        ([], "id", None),
        (["id-orig", "en"], None, "id-orig"),
    ],
)
def test_caption_key(keys: list[str], language: str | None, expected: str | None) -> None:
    assert caption_key(dict.fromkeys(keys, []), language) == expected


# --- choosing the source --------------------------------------------------------------------

COMPLETED = datetime(2026, 10, 26, 6, 0, tzinfo=UTC)


def sample_target(cfg: AppConfig, tmp_path: Path) -> Target:
    root = tmp_path / "8fKUNhHQR_M_000140-000240"
    write_json_atomic(root / "download.json", {
        "video_id": "8fKUNhHQR_M", "offset_s": 100.0, "vod": {"duration_s": 60.0},
        "completed_at": COMPLETED.isoformat(),
    })  # fmt: skip
    (root / "audio.wav").write_bytes(b"")
    return resolve_target(cfg, str(root))


def fake_fetch(available: bool) -> Any:
    def fetch(video_id: str, dest: Path, language: str | None, env: Any) -> str | None:
        if not available:
            return None
        dest.write_text(json.dumps(fixture_captions()), encoding="utf-8")
        return "id-orig"

    return fetch


def fake_whisper(calls: list[str]) -> Any:
    def transcribe(tc: TranscribeConfig, audio: Path) -> WhisperResult:
        calls.append(audio.name)
        rec = block(words((0.0, 0.5, "dari"), (0.5, 1.0, "whisper.")))
        engine = {"model": "large-v3", "device": "cuda", "elapsed_s": 1.0}
        return WhisperResult([rec], "id", 60.0, engine)

    return transcribe


def test_auto_uses_youtube_captions_when_present(
    cfg: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(core, "fetch_captions", fake_fetch(True))
    monkeypatch.setattr(core, "transcribe_audio", fake_whisper(calls))
    target = sample_target(cfg, tmp_path)
    doc = core.run_transcribe(cfg, target, now=COMPLETED)
    assert doc["source"] == "youtube" and calls == []
    assert [s["text"] for s in doc["segments"]] == [
        "halo semua gimana kabarnya?", "lu *** anjing", "mantap",
    ]  # fmt: skip
    assert doc["sounds"] == [{"t": 4.5, "tag": "tertawa"}, {"t": 8.5, "tag": "musik"}]
    assert read_json(target.artifacts.transcript)["source"] == "youtube"
    srt = target.artifacts.transcript_srt.read_text(encoding="utf-8")
    assert srt.startswith("1\n00:00:01,000")


def test_auto_waits_for_captions_then_falls_back_to_whisper(
    cfg: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(core, "fetch_captions", fake_fetch(False))
    monkeypatch.setattr(core, "transcribe_audio", fake_whisper(calls))
    target = sample_target(cfg, tmp_path)
    with pytest.raises(core.CaptionsNotReady):
        core.run_transcribe(cfg, target, now=COMPLETED + timedelta(hours=1))
    assert not target.artifacts.transcript.exists() and calls == []
    doc = core.run_transcribe(cfg, target, now=COMPLETED + timedelta(hours=7))
    assert doc["source"] == "whisper" and calls == ["audio.wav"]
    assert doc["sounds"] == []


def test_no_wait_and_whisper_source_skip_the_wait(
    cfg: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(core, "fetch_captions", fake_fetch(False))
    monkeypatch.setattr(core, "transcribe_audio", fake_whisper(calls))
    target = sample_target(cfg, tmp_path)
    assert core.run_transcribe(cfg, target, wait=False, now=COMPLETED)["source"] == "whisper"
    assert core.run_transcribe(cfg, target, source="whisper", now=COMPLETED)["source"] == "whisper"


def test_youtube_only_source_fails_without_captions(
    cfg: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(core, "fetch_captions", fake_fetch(False))
    with pytest.raises(core.TranscribeError):
        core.run_transcribe(cfg, sample_target(cfg, tmp_path), source="youtube", now=COMPLETED)


def test_hallucinated_sentence_is_removed_before_pause_splitting() -> None:
    # From the sample: Whisper invented this over the stream's last seconds, with a pause
    # inside it, so pause-splitting alone left "Terima kasih telah" and "menonton".
    ws = words(
        (3588.9, 3589.4, "ngamuk,"), (3589.4, 3589.9, "anjing."),
        (3591.0, 3591.6, "Terima"), (3591.6, 3592.0, "kasih"), (3592.0, 3592.4, "telah"),
        (3593.1, 3593.6, "menonton"),
    )  # fmt: skip
    cleaned, removed = remove_hallucinations(block(ws))
    assert [w["word"] for w in cleaned["words"]] == ["ngamuk,", "anjing."]
    assert removed == [{"start": 3591.0, "end": 3593.6, "text": "Terima kasih telah menonton",
                        "reason": "known Whisper hallucination"}]  # fmt: skip


def test_repeated_lines_are_dropped_but_short_repeats_kept() -> None:
    line = "kagak pernah ngeliat dia ini anjir?"
    recs = [record("tapi aegisnya udah ke pake kan?")] + [record(line)] * 4
    recs += [record("Aduh."), record("Aduh.")]
    kept, dropped = split_segments(recs, CFG)
    assert [s["text"] for s in kept] == ["tapi aegisnya udah ke pake kan?", line, "Aduh.", "Aduh."]
    assert [d["reason"] for d in dropped] == ["repeated line (decoding loop)"] * 3


def test_youtube_repeats_and_censored_words_are_kept() -> None:
    def caption(text: str) -> dict[str, Any]:
        return {**record(text), "avg_logprob": None, "no_speech_prob": None,
                "compression_ratio": None}  # fmt: skip

    recs = [caption("Let's go."), caption("Let's go."), caption("lu dari tadi ngeselin"),
            caption("lu dari tadi ngeselin"), caption("***")]  # fmt: skip
    kept, dropped = split_segments(recs, CFG)
    assert len(kept) == 5 and dropped == []
