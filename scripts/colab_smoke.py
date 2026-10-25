#!/usr/bin/env python3
"""One-batch GPU smoke test for the pretrained models (downloads weights).

    REMOTE=/content/lab2 colab exec -s lab2 -f scripts/colab_smoke.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("REMOTE", "/content/lab2"))

for model in ("vgg16", "alexnet"):
    cmd = [sys.executable, "src/train.py", "--model", model,
           "--epochs", "1", "--batch-size", "8",
           "--train-subset", "32", "--test-subset", "16",
           "--num-workers", "0", "--max-batches", "1",
           "--out", f"results/smoke_{model}.json"]
    print(f"[smoke] {model}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(r.stdout[-3000:], flush=True)
    print(r.stderr[-3000:], flush=True)
    if r.returncode != 0:
        raise SystemExit(f"smoke failed: {model} (rc={r.returncode})")

print("[done] smoke", flush=True)
