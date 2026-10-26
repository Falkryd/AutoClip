"""Logging to logs/pipeline.log (rotated) and the console, with secrets masked."""

from __future__ import annotations

import logging
import re
import sys
from logging.handlers import RotatingFileHandler

from src.common.config import AppConfig
from src.common.paths import log_file

_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
_SECRET_PATTERNS = (
    re.compile(r"https://(?:[\w-]+\.)?discord(?:app)?\.com/api/webhooks/[^\s\"']+"),
    re.compile(r"ya29\.[\w.-]+"),  # Google access token
    re.compile(r"1//[\w-]{20,}"),  # Google refresh token
    re.compile(r"GOCSPX-[\w-]+"),  # OAuth client secret
)
_NOISY_LOGGERS = ("googleapiclient.discovery_cache", "httpx", "httpcore", "urllib3")

_configured = False


def redact(text: str) -> str:
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


class RedactingFormatter(logging.Formatter):
    """Masks secrets in the whole formatted record, tracebacks included."""

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def setup_logging(cfg: AppConfig, stage: str) -> logging.Logger:
    """Configure the root logger once per process and return the stage's logger."""
    global _configured
    if not _configured:
        _make_console_safe()
        path = log_file(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        formatter = RedactingFormatter(_FORMAT)
        file_handler = RotatingFileHandler(
            path,
            maxBytes=cfg.logging.max_bytes,
            backupCount=cfg.logging.backup_count,
            encoding="utf-8",
            delay=True,
        )
        console_handler = logging.StreamHandler(sys.stderr)
        root = logging.getLogger()
        root.setLevel(cfg.logging.level)
        for handler in (file_handler, console_handler):
            handler.setFormatter(formatter)
            root.addHandler(handler)
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)
        _configured = True
    return logging.getLogger(f"autoclip.{stage}")


def _make_console_safe() -> None:
    """Stream titles carry emoji; don't crash when Task Scheduler pipes output as cp1252."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
