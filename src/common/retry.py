"""Retry with exponential backoff, for network and API calls only."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

import httpx
from googleapiclient.errors import HttpError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

_log = logging.getLogger("autoclip.retry")

_RETRY_STATUS = {408, 429, 500, 502, 503, 504}
_TRANSIENT_TEXT = (
    "timed out",
    "connection reset",
    "connection aborted",
    "remote end closed",
    "temporarily unavailable",
    "getaddrinfo failed",
    "http error 429",
    "http error 5",
)

F = TypeVar("F", bound=Callable[..., object])


def is_transient(exc: BaseException) -> bool:
    """True for errors worth retrying. Quota errors (403) and bad input are not."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRY_STATUS
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, HttpError):
        return exc.resp.status in _RETRY_STATUS
    if isinstance(exc, ConnectionError | TimeoutError):
        return True
    if type(exc).__name__ == "DownloadError":  # yt-dlp wraps network failures in this
        message = str(exc).lower()
        return any(text in message for text in _TRANSIENT_TEXT)
    return False


def with_retries(
    attempts: int = 5,
    max_wait: float = 60,
    also_retry: tuple[type[BaseException], ...] = (),
) -> Callable[[F], F]:
    """`also_retry` adds exception types that are transient only in the caller's context."""
    return retry(
        retry=retry_if_exception(lambda exc: is_transient(exc) or isinstance(exc, also_retry)),
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=2, max=max_wait),
        before_sleep=before_sleep_log(_log, logging.WARNING),
        reraise=True,
    )
