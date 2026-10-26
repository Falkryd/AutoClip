"""Settings from config.yaml and secrets from .env, validated when loaded."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"


class ConfigError(RuntimeError):
    """config.yaml or .env is missing or invalid."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ChannelConfig(_Section):
    id: str | None = Field(default=None, pattern=r"^UC[A-Za-z0-9_-]{22}$")


class PathsConfig(_Section):
    work_dir: Path = Path("work")
    log_dir: Path = Path("logs")
    state_dir: Path = Path("state")
    samples_dir: Path = Path("samples")


class LoggingConfig(_Section):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    max_bytes: int = Field(default=5_000_000, gt=0)
    backup_count: int = Field(default=5, ge=0)


class TriggerConfig(_Section):
    method: Literal["ytdlp", "rss", "api"] = "ytdlp"
    lookback_hours: float = Field(default=48, gt=0)
    give_up_after_hours: float = Field(default=168, gt=0)
    max_results: int = Field(default=15, ge=1, le=50)


class DownloadConfig(_Section):
    format: str = "bv*[height<=1080]+ba/b[height<=1080]"
    cookies_file: Path | None = None
    deno_path: Path | None = None
    retries: int = Field(default=5, ge=0)


class TranscribeConfig(_Section):
    # auto = YouTube's captions, waiting up to captions_wait_hours for them, else Whisper
    source: Literal["auto", "youtube", "whisper"] = "auto"
    captions_wait_hours: float = Field(default=6, ge=0)
    model: str = "large-v3"
    device: Literal["cuda", "cpu", "auto"] = "cuda"
    compute_type: str = "int8_float16"
    language: str | None = "id"
    beam_size: int = Field(default=5, ge=1)
    vad_filter: bool = True
    batch_size: int = Field(default=8, ge=0)  # 0 = sequential (slower, uses less VRAM)
    max_segment_seconds: float = Field(default=12, gt=0)
    split_pause_seconds: float = Field(default=0.8, gt=0)
    initial_prompt: str | None = None
    hotwords: str | None = None
    # Segments are dropped when no_speech_prob is above AND avg_logprob is below these...
    drop_no_speech_prob: float = Field(default=0.6, ge=0, le=1)
    drop_avg_logprob: float = Field(default=-1.0, le=0)
    # ...or when the text repeats itself this much (a decoding loop).
    drop_compression_ratio: float = Field(default=2.4, gt=0)


class OllamaConfig(_Section):
    host: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5:7b"
    num_ctx: int = Field(default=8192, ge=2048)
    temperature: float = Field(default=0.2, ge=0, le=2)
    timeout_seconds: float = Field(default=300, gt=0)


class Weights(_Section):
    chat: float = Field(default=0.1, ge=0)
    laughter: float = Field(default=0.3, ge=0)
    audio: float = Field(default=0.2, ge=0)
    llm: float = Field(default=0.4, ge=0)

    @model_validator(mode="after")
    def _not_all_zero(self) -> Weights:
        if self.chat + self.laughter + self.audio + self.llm <= 0:
            raise ValueError("at least one highlight weight must be positive")
        return self


class HighlightsConfig(_Section):
    # Max 6: each upload costs ~1,600 of the 10,000 default daily quota units.
    clips_per_stream: int = Field(default=3, ge=1, le=6)
    min_clip_seconds: float = Field(default=30, ge=5)
    max_clip_seconds: float = Field(default=60, le=180)
    window_seconds: float = Field(default=45, gt=0)
    step_seconds: float = Field(default=5, gt=0)
    shortlist_size: int = Field(default=12, ge=1)
    context_seconds: float = Field(default=15, ge=0)
    min_gap_seconds: float = Field(default=60, ge=0)
    min_llm_score: int = Field(default=4, ge=0, le=10)
    laughter_tags: list[str] = Field(default_factory=lambda: ["tertawa"])
    music_flag_seconds: float = Field(default=20, ge=0)
    weights: Weights = Weights()
    ollama: OllamaConfig = OllamaConfig()

    @model_validator(mode="after")
    def _check_lengths(self) -> HighlightsConfig:
        if self.min_clip_seconds > self.max_clip_seconds:
            raise ValueError("min_clip_seconds is larger than max_clip_seconds")
        if not self.min_clip_seconds <= self.window_seconds <= self.max_clip_seconds:
            raise ValueError("window_seconds must be between min_clip_seconds and max_clip_seconds")
        return self


class Rect(_Section):
    """A region of the source frame, as fractions of its width and height."""

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    w: float = Field(gt=0, le=1)
    h: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def _inside_frame(self) -> Rect:
        if self.x + self.w > 1 + 1e-9 or self.y + self.h > 1 + 1e-9:
            raise ValueError("region extends past the edge of the frame")
        return self


class CaptionConfig(_Section):
    enabled: bool = True
    style: Literal["word_highlight", "sentence"] = "word_highlight"
    font: str = "Arial"
    font_size: int = Field(default=64, gt=0)
    max_words_per_line: int = Field(default=4, ge=1)
    text_color: str = Field(default="#FFFFFF", pattern=r"^#[0-9A-Fa-f]{6}$")
    highlight_color: str = Field(default="#FFD400", pattern=r"^#[0-9A-Fa-f]{6}$")
    outline: int = Field(default=4, ge=0)
    y: float = Field(default=0.49, gt=0, lt=1)  # caption baseline, as a fraction of the height


class RenderConfig(_Section):
    layout: Literal["stacked", "center"] = "stacked"
    width: int = Field(default=1080, gt=0)
    height: int = Field(default=1920, gt=0)
    gameplay: Rect = Rect(x=0.236, y=0.0, w=0.528, h=1.0)
    avatar: Rect = Rect(x=0.82, y=0.62, w=0.18, h=0.38)
    avatar_panel_fraction: float = Field(default=0.4, gt=0, lt=1)
    video_encoder: Literal["h264_nvenc", "libx264"] = "h264_nvenc"
    quality: int = Field(default=23, ge=0, le=51)
    audio_bitrate: str = "160k"
    normalize_audio: bool = True  # to -14 LUFS, YouTube's playback loudness
    captions: CaptionConfig = CaptionConfig()


class UploadConfig(_Section):
    # Hard rule: never publish automatically. Any other value fails validation.
    privacy: Literal["private"] = "private"
    category_id: str = "20"
    title_template: str = "{clip_title} #Shorts"
    description_template: str = "From the stream: {stream_url}\n\n#Shorts"
    tags: list[str] = Field(default_factory=list)
    max_uploads_per_day: int = Field(default=5, ge=0, le=6)


class NotifyConfig(_Section):
    username: str = "AutoClip"
    mention_user_id: str | None = Field(default=None, pattern=r"^\d+$")


class CleanupConfig(_Section):
    delete_vod_after_render: bool = False


class AppConfig(_Section):
    channel: ChannelConfig = ChannelConfig()
    paths: PathsConfig = PathsConfig()
    logging: LoggingConfig = LoggingConfig()
    trigger: TriggerConfig = TriggerConfig()
    download: DownloadConfig = DownloadConfig()
    transcribe: TranscribeConfig = TranscribeConfig()
    highlights: HighlightsConfig = HighlightsConfig()
    render: RenderConfig = RenderConfig()
    upload: UploadConfig = UploadConfig()
    notify: NotifyConfig = NotifyConfig()
    cleanup: CleanupConfig = CleanupConfig()

    def resolve(self, path: Path) -> Path:
        """Resolve a config path relative to the project folder."""
        return path if path.is_absolute() else PROJECT_ROOT / path


class Secrets(BaseModel):
    model_config = ConfigDict(frozen=True)

    youtube_client_secret_file: Path
    youtube_token_file: Path
    discord_webhook_url: SecretStr | None = None


def load_config(path: Path | None = None) -> AppConfig:
    path = path or Path(os.environ.get("AUTOCLIP_CONFIG", DEFAULT_CONFIG_PATH))
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")
    try:
        return AppConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid config in {path}:\n{exc}") from exc


def load_secrets() -> Secrets:
    """Read secrets from the environment, loading .env first if it exists."""
    load_dotenv(ENV_PATH, override=False)

    def _path(name: str, default: str) -> Path:
        path = Path(os.environ.get(name) or default)
        return path if path.is_absolute() else PROJECT_ROOT / path

    return Secrets(
        youtube_client_secret_file=_path(
            "YOUTUBE_CLIENT_SECRET_FILE", "secrets/client_secret.json"
        ),
        youtube_token_file=_path("YOUTUBE_TOKEN_FILE", "secrets/token.json"),
        discord_webhook_url=os.environ.get("DISCORD_WEBHOOK_URL") or None,
    )
