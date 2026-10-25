"""Shared helpers: seeding, device selection, timing and result I/O."""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Benchmark kernels are faster but not bit-deterministic; the lab compares
    # architectures, so speed wins. Flip to True for exact reruns.
    torch.backends.cudnn.benchmark = True


def get_device(pref: str = "auto") -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return "cpu"


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Timer:
    """Wall-clock timer with CUDA synchronisation on stop."""

    def __init__(self, device: torch.device | None = None):
        self.device = device
        self.t0 = 0.0

    def __enter__(self):
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.t0


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[json] {path}", flush=True)


def env_info(device: torch.device) -> dict:
    import torchvision

    info = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "device": device_name(device),
        "device_type": device.type,
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info["gpu_memory_gb"] = round(props.total_memory / 2**30, 1)
        info["gpu_capability"] = f"{props.major}.{props.minor}"
    return info
