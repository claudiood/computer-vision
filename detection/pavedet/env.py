"""Informações do ambiente de execução (registradas junto de cada treino)."""
from __future__ import annotations

import os
import platform
import subprocess
import sys

from .config import REPO_ROOT


def auto_device() -> str:
    """cuda:0 > mps (Apple Silicon) > cpu.

    A Ultralytics, com device=None, escolhe CUDA ou CPU, mas NÃO escolhe a GPU do
    Mac (MPS) sozinha: por isso a escolha é feita aqui.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def resolve_device(device) -> str:
    return auto_device() if device in (None, "", "auto") else str(device)


def git_info() -> dict:
    def run(*cmd):
        try:
            return subprocess.check_output(cmd, cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None

    return {"commit": run("git", "rev-parse", "--short", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no"))}


def env_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
            "cpu_count": os.cpu_count()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = p.name
            info["gpu_mem_gb"] = round(p.total_memory / 1e9, 1)
        info["mps"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except ImportError:
        info["torch"] = None
    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except ImportError:
        info["ultralytics"] = None
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        pass
    info["git"] = git_info()
    return info


def peak_rss_mb() -> float | None:
    """Pico de memória RAM do processo (MB)."""
    try:
        import resource

        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(r / 1024 / 1024, 1) if sys.platform == "darwin" else round(r / 1024, 1)
    except Exception:
        return None


def gpu_peak_mb(reset: bool = False) -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            v = torch.cuda.max_memory_allocated() / 1024 ** 2
            if reset:
                torch.cuda.reset_peak_memory_stats()
            return round(v, 1)
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return round(torch.mps.driver_allocated_memory() / 1024 ** 2, 1)
    except Exception:
        pass
    return None


def free_memory() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
