"""Shared helpers for Exp-Coder scientific benchmarks.

Kept intentionally small: seeds, result serialization (JSON + metadata),
runtime memory probing (Windows ctypes fallback when psutil is absent), and
CUDA diagnostics.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import subprocess


def git_commit(*, short: bool = True) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short" if short else "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_result(path: os.PathLike, payload: dict) -> Path:
    """Write a benchmark result JSON (adding run metadata when absent)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload.setdefault("meta", {}).setdefault("commit", git_commit())
    payload["meta"].setdefault("timestamp", utc_now_iso())
    payload["meta"].setdefault("torch_version", _torch_version())
    payload["meta"].setdefault("python", _python_version())
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def load_result(path: os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _torch_version() -> str:
    try:
        import torch

        return f"{torch.__version__}"
    except Exception:
        return "not-installed"


def _python_version() -> str:
    import sys

    return sys.version.split()[0]


def cuda_info() -> Dict[str, Any]:
    """Read-only CUDA diagnostics; never triggers CUDA init on CPU builds."""
    import torch

    info = {
        "torch_version": torch.__version__,
        "torch_build_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["cuda_total_memory_mb"] = props.total_memory / 1e6
    return info


def process_working_set_mb() -> Optional[float]:
    """Current process working set (MB) — Windows via ctypes, else None."""
    try:
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pmc = ProcessMemoryCounters()
        pmc.cb = ctypes.sizeof(pmc)
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        hproc = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(hproc, ctypes.byref(pmc), pmc.cb)
        if ok:
            return round(pmc.WorkingSetSize / 1e6, 1)
    except Exception:
        pass
    return None


def memory_mb() -> Optional[float]:
    return process_working_set_mb()


def force_seed(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    torch.manual_seed(seed)


RESULT_DIR = Path(__file__).resolve().parent / "results"
CORPUS_DIR = Path(__file__).resolve().parent / "data" / "code_sample"