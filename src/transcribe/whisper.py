"""faster-whisper on the GPU: the fallback when YouTube has no captions."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.config import TranscribeConfig
from src.common.timecode import format_timecode
from src.transcribe.gpu import enable_cuda_dlls
from src.transcribe.segments import segment_record

_log = logging.getLogger("autoclip.transcribe")


class WhisperError(RuntimeError):
    """Whisper failed to load or run."""


@dataclass(frozen=True)
class WhisperResult:
    records: list[dict[str, Any]]  # raw blocks with word timings, before splitting
    language: str
    duration_s: float
    engine: dict[str, Any]  # model settings and timings, stored in transcript.json


def transcribe_audio(cfg: TranscribeConfig, audio: Path) -> WhisperResult:
    model = load_model(cfg)
    started = time.monotonic()
    segments, info = _transcribe(model, cfg, audio)
    progress = _Progress(info.duration)
    records = []
    try:
        for segment in segments:  # a generator: the actual work happens while iterating
            records.append(segment_record(segment))
            progress.update(segment.end)
    except RuntimeError as exc:
        raise WhisperError(f"Whisper failed while transcribing: {exc}") from exc
    elapsed = time.monotonic() - started
    speech = getattr(info, "duration_after_vad", None)
    return WhisperResult(
        records=records,
        language=info.language,
        duration_s=round(info.duration, 3),
        engine={
            "model": cfg.model,
            "device": cfg.device,
            "compute_type": cfg.compute_type,
            "batch_size": cfg.batch_size,
            "language_probability": round(info.language_probability, 3),
            "speech_s": round(speech, 3) if speech is not None else None,
            "elapsed_s": round(elapsed, 1),
        },
    )


def load_model(cfg: TranscribeConfig) -> Any:
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    enable_cuda_dlls()
    from faster_whisper import WhisperModel  # heavy import, and must follow enable_cuda_dlls

    _log.info("Loading Whisper %s on %s (%s)", cfg.model, cfg.device, cfg.compute_type)
    try:
        return WhisperModel(cfg.model, device=cfg.device, compute_type=cfg.compute_type)
    except (RuntimeError, ValueError) as exc:
        raise WhisperError(
            f"Could not load Whisper {cfg.model} on {cfg.device}: {exc}. To run on the CPU "
            "instead, set transcribe.device: cpu and compute_type: int8 in config.yaml."
        ) from exc


def _transcribe(model: Any, cfg: TranscribeConfig, audio: Path) -> tuple[Iterable[Any], Any]:
    options: dict[str, Any] = {
        "language": cfg.language,
        "beam_size": cfg.beam_size,
        "vad_filter": cfg.vad_filter,
        "word_timestamps": True,  # captions highlight word by word
        "initial_prompt": cfg.initial_prompt,
        "hotwords": cfg.hotwords,
        # Feeding each window's text into the next causes repetition loops on long streams.
        "condition_on_previous_text": False,
    }
    if cfg.batch_size > 0:
        from faster_whisper import BatchedInferencePipeline

        pipeline = BatchedInferencePipeline(model)
        return pipeline.transcribe(str(audio), batch_size=cfg.batch_size, **options)
    return model.transcribe(str(audio), **options)


class _Progress:
    """Logs every 10 % of the audio covered."""

    def __init__(self, duration_s: float) -> None:
        self._duration = duration_s
        self._next_pct = 10
        self._started = time.monotonic()

    def update(self, position_s: float) -> None:
        if self._duration <= 0:
            return
        pct = 100 * position_s / self._duration
        if pct >= self._next_pct:
            _log.info(
                "Transcribed %d%% (%s of %s) after %s",
                pct, format_timecode(position_s), format_timecode(self._duration),
                format_timecode(time.monotonic() - self._started),
            )  # fmt: skip
            self._next_pct = (int(pct) // 10 + 1) * 10
