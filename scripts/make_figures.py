#!/usr/bin/env python3
"""Build the Lab 2 figures from the JSON reports.

    python scripts/make_figures.py

Reads ``results/{cnn,vit,vgg16,alexnet,frcnn}.json`` and writes PNGs to
``results/figures/``. Every number in the README comes from these artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["cnn", "vit", "vgg16", "alexnet", "frcnn"]
LABELS = {"cnn": "CNN (scratch)", "vit": "ViT (scratch)",
          "vgg16": "VGG16 (pretrained)", "alexnet": "AlexNet (pretrained)",
          "frcnn": "Faster R-CNN"}
COLORS = {"cnn": "#1f77b4", "vit": "#9467bd", "vgg16": "#2ca02c",
          "alexnet": "#ff7f0e", "frcnn": "#d62728"}


def load() -> dict:
    data = {}
    for m in MODELS:
        p = ROOT / "results" / f"{m}.json"
        if p.exists():
            data[m] = json.loads(p.read_text())
    return data


def acc_of(d: dict) -> float:
    return (d["final"]["detection_accuracy"] if d["task"] == "detection"
            else d["final"]["accuracy"])


def f1_of(d: dict) -> float:
    return d["final"]["macro_f1"]


def training_curves(data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    for m, d in data.items():
        ep = [h["epoch"] for h in d["history"]]
        axes[0].plot(ep, [h["train_loss"] for h in d["history"]], marker="o",
                     ms=3, color=COLORS[m], label=LABELS[m])
        if d["task"] == "classification":
            axes[1].plot(ep, [h["test_accuracy"] * 100 for h in d["history"]],
                         marker="o", ms=3, color=COLORS[m], label=LABELS[m])
        else:
            axes[1].plot(ep, [h["detection_accuracy"] * 100
                              for h in d["history"]],
                         marker="s", ms=3, ls="--", color=COLORS[m],
                         label=f"{LABELS[m]} (det. acc.)")

    axes[0].set(title="Training loss per epoch", xlabel="epoch",
                ylabel="loss (sum for Faster R-CNN)")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.3)
    axes[1].set(title="Test accuracy per epoch", xlabel="epoch",
                ylabel="accuracy (%)")
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(70, 100.5)
    for ax in axes:
        ax.legend(fontsize=8, framealpha=0.9)
    fig.suptitle("Lab 2 — optimisation curves on MNIST (Colab T4)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_training_curves.png", dpi=150)
    plt.close(fig)


def comparison(data: dict) -> None:
    models = [m for m in MODELS if m in data]
    acc = [acc_of(data[m]) * 100 for m in models]
    f1 = [f1_of(data[m]) * 100 for m in models]
    params = [data[m]["params"] / 1e6 for m in models]
    ttrain = [data[m]["timing"]["train_sec"] for m in models]
    infer = [data[m]["timing"]["ms_per_image"] for m in models]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()

    def bars(ax, values, title, ylabel, log=False, fmt="{:.2f}", ylim=None):
        bars_ = ax.bar([LABELS[m] for m in models], values,
                       color=[COLORS[m] for m in models])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.grid(axis="y", alpha=0.3)
        if log:
            ax.set_yscale("log")
        if ylim:
            ax.set_ylim(*ylim)
        for b, v in zip(bars_, values):
            ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=8,
                        xytext=(0, 2), textcoords="offset points")

    bars(axes[0], acc, "Accuracy (test)", "%", fmt="{:.2f}",
         ylim=(90, 101))
    bars(axes[1], f1, "Macro F1 (test)", "%", fmt="{:.2f}", ylim=(90, 101))
    bars(axes[2], params, "Trainable parameters", "millions (log)",
         log=True, fmt="{:.2f}")
    bars(axes[3], ttrain, "Training time", "seconds (log)", log=True,
         fmt="{:.0f}")
    bars(axes[4], infer, "Inference latency", "ms / image (log)", log=True,
         fmt="{:.3f}")

    # Accuracy vs training cost, the trade-off a practitioner actually faces.
    ax = axes[5]
    for m in models:
        ax.scatter(data[m]["timing"]["train_sec"], acc_of(data[m]) * 100,
                   s=80, color=COLORS[m], label=LABELS[m], zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("training time (s, log)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Accuracy vs training cost")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("Lab 2 — model comparison (T4, MNIST; Faster R-CNN = "
                 "detection accuracy/F1)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_comparison.png", dpi=150)
    plt.close(fig)


def convergence(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for m in ("cnn", "vit", "vgg16", "alexnet"):
        if m not in data:
            continue
        d = data[m]
        t, acc, total = [], [], 0.0
        for h in d["history"]:
            total += h["epoch_sec"]
            t.append(total)
            acc.append(h["test_accuracy"] * 100)
        ax.plot(t, acc, marker="o", ms=3, color=COLORS[m], label=LABELS[m])
    ax.set(title="Test accuracy vs wall-clock training time",
           xlabel="cumulative training time (s)", ylabel="accuracy (%)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_convergence.png", dpi=150)
    plt.close(fig)


def frcnn_curves(data: dict) -> None:
    if "frcnn" not in data:
        return
    d = data["frcnn"]
    ep = [h["epoch"] for h in d["history"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(ep, [h["map50"] * 100 for h in d["history"]], marker="o",
            color="#d62728", label="mAP@0.5")
    ax.plot(ep, [h["detection_accuracy"] * 100 for h in d["history"]],
            marker="s", color="#1f77b4", label="detection accuracy")
    ax.plot(ep, [h["mean_iou"] * 100 for h in d["history"]], marker="^",
            color="#2ca02c", label="mean IoU (x100)")
    ax.set(title="Faster R-CNN on canvas-MNIST (128 px)",
           xlabel="epoch", ylabel="%")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_frcnn.png", dpi=150)
    plt.close(fig)


def per_class_f1(data: dict) -> None:
    models = [m for m in ("cnn", "vit", "vgg16", "alexnet") if m in data]
    if not models:
        return
    grid = np.array([data[m]["final"]["per_class"]["f1"] for m in models])
    fig, ax = plt.subplots(figsize=(10, 3.6))
    im = ax.imshow(grid, cmap="viridis", vmin=0.95, vmax=1.0, aspect="auto")
    ax.set_xticks(range(10), [str(i) for i in range(10)])
    ax.set_yticks(range(len(models)), [LABELS[m] for m in models])
    ax.set(xlabel="digit", title="Per-class F1 on the MNIST test set")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center",
                    fontsize=7, color="white" if grid[i, j] < 0.985 else "black")
    fig.colorbar(im, ax=ax, label="F1")
    fig.tight_layout()
    fig.savefig(OUT / "fig_per_class_f1.png", dpi=150)
    plt.close(fig)


def main() -> None:
    data = load()
    if not data:
        raise SystemExit("no result JSONs found; run the Colab stages first")
    training_curves(data)
    comparison(data)
    convergence(data)
    frcnn_curves(data)
    per_class_f1(data)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
