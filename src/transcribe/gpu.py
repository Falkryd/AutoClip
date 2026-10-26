"""Make the CUDA libraries from the nvidia-* pip packages findable on Windows."""

from __future__ import annotations

import glob
import os
import sys

_done = False


def enable_cuda_dlls() -> None:
    """CTranslate2 loads cuBLAS and cuDNN by name, but Windows doesn't search the folders the
    nvidia-* pip packages install them into. Call before importing faster_whisper."""
    global _done
    if _done or sys.platform != "win32":
        return
    try:
        import nvidia
    except ImportError:
        return  # CPU-only install
    for root in nvidia.__path__:
        dll_dirs = {os.path.dirname(p) for p in glob.glob(os.path.join(root, "*", "bin", "*.dll"))}
        for dll_dir in sorted(dll_dirs):
            os.add_dll_directory(dll_dir)
            os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
    _done = True
