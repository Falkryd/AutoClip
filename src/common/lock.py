"""Process lock so overlapping scheduled runs cannot work on the same thing."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO


class LockHeldError(RuntimeError):
    """Another process already holds the lock."""


if sys.platform == "win32":
    import msvcrt

    def _try_lock(fh: IO[str]) -> None:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(fh: IO[str]) -> None:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_lock(fh: IO[str]) -> None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(fh: IO[str]) -> None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Hold an OS-level lock on `path`. The OS drops it if the process dies, so no stale locks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as fh:
        try:
            _try_lock(fh)
        except OSError as exc:
            raise LockHeldError(f"Another AutoClip process holds {path.name}") from exc
        try:
            yield
        finally:
            _unlock(fh)
