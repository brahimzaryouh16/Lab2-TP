#!/usr/bin/env python3
"""Print a compact summary of the Lab 2 JSON reports (for the README)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    files = sorted(ROOT.glob("results/*.json"))
    for path in files:
        if path.name.startswith("smoke_"):
            continue
        d = json.loads(path.read_text())
        print("=" * 78)
        print(f"{path.name} | task={d['task']} | device={d['environment']['device']}")
        print(f"params={d['params']:,} | meta={d['meta']}")
        best = {k: d["best"][k] for k in d["best"]
                if k in ("epoch", "accuracy", "macro_f1", "map50", "train_loss")}
        final = {k: d["final"][k] for k in d["final"]
                 if k in ("accuracy", "macro_f1", "map50", "detection_accuracy",
                          "label_accuracy", "mean_iou")}
        print(f"best={best}")
        print(f"final={final}")
        print(f"timing={d['timing']}")
        for h in d["history"]:
            keep = {k: v for k, v in h.items()
                    if k not in ("loss_parts", "lr")}
            print("  ", keep)


if __name__ == "__main__":
    main()
