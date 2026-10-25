#!/usr/bin/env python3
"""Remote stage driver for the Lab 2 Colab runs.

One stage per `colab exec` so a dead session only loses the stage in flight:

    STAGE=cnn python scripts/colab_runner.py

The script shells out to ``src/train.py`` in a *fresh* Python process (so GPU
memory is released between models), verifies the JSON report exists and
prints ``[done] <stage>``. The shell driver greps for that marker because
`colab exec` can exit 0 even when the cell raised.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("REMOTE", "/content/lab2"))
STAGE = os.environ.get("STAGE", "").strip()

# Training budget per stage. Small enough for free T4 sessions, large enough
# to reach a stable validation plateau (see README for the rationale).
STAGES: dict[str, list[str]] = {
    "cnn": [
        "--model", "cnn", "--epochs", "8", "--batch-size", "128",
        "--lr", "1e-3", "--weight-decay", "1e-4",
        "--out", "results/cnn.json",
    ],
    "vit": [
        "--model", "vit", "--epochs", "12", "--batch-size", "128",
        "--lr", "3e-4", "--weight-decay", "0.05", "--label-smoothing", "0.1",
        "--out", "results/vit.json",
    ],
    "frcnn": [
        "--model", "frcnn", "--epochs", "3", "--batch-size", "8",
        "--lr", "0.005", "--weight-decay", "5e-4",
        "--train-subset", "8000", "--test-subset", "2000",
        "--out", "results/frcnn.json",
    ],
    "vgg16": [
        "--model", "vgg16", "--epochs", "4", "--batch-size", "32",
        "--lr", "1e-4", "--weight-decay", "1e-4",
        "--train-subset", "12000", "--test-subset", "2000",
        "--out", "results/vgg16.json",
    ],
    "alexnet": [
        "--model", "alexnet", "--epochs", "4", "--batch-size", "64",
        "--lr", "1e-4", "--weight-decay", "1e-4",
        "--train-subset", "12000", "--test-subset", "2000",
        "--out", "results/alexnet.json",
    ],
}


def main() -> None:
    if STAGE not in STAGES:
        raise SystemExit(f"unknown STAGE={STAGE!r}; choose from {sorted(STAGES)}")
    if not ROOT.is_dir():
        raise SystemExit(f"remote workspace missing: {ROOT}")

    (ROOT / "results").mkdir(exist_ok=True)
    cmd = [sys.executable, "src/train.py", *STAGES[STAGE]]
    print(f"[run] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)

    report = ROOT / "results" / f"{STAGE}.json"
    if not report.is_file() or report.stat().st_size < 100:
        raise SystemExit(f"missing or empty report: {report}")
    print(f"[report] {report} ({report.stat().st_size} bytes)", flush=True)
    print(f"[done] {STAGE}", flush=True)


if __name__ == "__main__":
    main()
