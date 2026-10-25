#!/usr/bin/env python3
"""EDA figures for the Lab 2 report (MNIST + canvas detection view)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from torchvision import datasets  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CanvasMNIST  # noqa: E402

OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data"


def samples(train) -> None:
    rng = np.random.RandomState(0)
    idx = rng.choice(len(train), 32, replace=False)
    fig, axes = plt.subplots(4, 8, figsize=(12, 6.2))
    for ax, i in zip(axes.ravel(), idx):
        img, label = train[int(i)]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(str(label), fontsize=9)
        ax.axis("off")
    fig.suptitle("MNIST training samples (one random batch)")
    fig.tight_layout()
    fig.savefig(OUT / "eda_samples.png", dpi=150)
    plt.close(fig)


def class_balance(train, test) -> None:
    c_train = np.bincount(train.targets.numpy(), minlength=10)
    c_test = np.bincount(test.targets.numpy(), minlength=10)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, counts, title, color in (
            (axes[0], c_train, f"Train split ({len(train):,} images)", "#1f77b4"),
            (axes[1], c_test, f"Test split ({len(test):,} images)", "#ff7f0e")):
        bars = ax.bar(range(10), counts, color=color)
        ax.set(title=title, xlabel="digit", ylabel="images")
        ax.set_xticks(range(10))
        ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, counts):
            ax.annotate(f"{v}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=7,
                        xytext=(0, 2), textcoords="offset points")
    fig.suptitle("MNIST class distribution")
    fig.tight_layout()
    fig.savefig(OUT / "eda_class_balance.png", dpi=150)
    plt.close(fig)
    print("train counts:", c_train.tolist())
    print("test counts:", c_test.tolist())


def pixels(train) -> None:
    data = train.data.float() / 255.0
    mean_img = data.mean(0).numpy()
    pix = train.data.numpy().ravel()[::10]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].imshow(mean_img, cmap="gray")
    axes[0].set_title("Mean image over the 60k training digits")
    axes[0].axis("off")
    axes[1].hist(pix, bins=64, log=True, color="#2ca02c")
    axes[1].set(title="Pixel value distribution (log count)",
                xlabel="pixel value (0 = black, 255 = white)",
                ylabel="count (log)")
    axes[1].grid(alpha=0.3)
    fig.suptitle("MNIST pixel statistics")
    fig.tight_layout()
    fig.savefig(OUT / "eda_pixels.png", dpi=150)
    plt.close(fig)


def canvas() -> None:
    ds = CanvasMNIST(str(DATA), train=True, canvas=128, subset=0, seed=0)
    picks = [3, 17, 42, 88]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, i in zip(axes, picks):
        img, target = ds[i]
        ax.imshow(img[0].numpy(), cmap="gray", vmin=0, vmax=1)
        box = target["boxes"][0].numpy()
        ax.add_patch(Rectangle((box[0], box[1]), box[2] - box[0],
                               box[3] - box[1], fill=False,
                               edgecolor="lime", linewidth=1.5))
        ax.set_title(f"digit {int(target['labels'][0]) - 1}")
        ax.axis("off")
    fig.suptitle("Detection view: 28x28 digit on a 128x128 canvas "
                 "(green = ground-truth box)")
    fig.tight_layout()
    fig.savefig(OUT / "eda_canvas.png", dpi=150)
    plt.close(fig)


def main() -> None:
    train = datasets.MNIST(str(DATA), train=True, download=True)
    test = datasets.MNIST(str(DATA), train=False, download=True)
    samples(train)
    class_balance(train, test)
    pixels(train)
    canvas()
    print("EDA figures written to", OUT)


if __name__ == "__main__":
    main()
