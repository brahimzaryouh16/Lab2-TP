"""MNIST data pipelines.

Two views of the same dataset:

* classification — 1x28x28 (or 3x224x224 for the ImageNet-pretrained models);
* detection      — the digit pasted on a 128x128 black canvas with a bounding
  box, so Faster R-CNN has a localisation target (MNIST itself is a
  classification dataset, not a detection one).

The Kaggle MNIST release and torchvision's MNIST loader contain the same
60k/10k split; we use torchvision so the download is reproducible.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _subsample(dataset, n: int, seed: int, offset: int):
    if not n or n >= len(dataset):
        return dataset
    g = torch.Generator().manual_seed(seed + offset)
    idx = torch.randperm(len(dataset), generator=g)[:n].tolist()
    return Subset(dataset, idx)


def classification_loaders(data_dir: str, batch_size: int,
                           train_subset: int = 0, test_subset: int = 0,
                           img_size: int = 28, rgb: bool = False,
                           num_workers: int = 2, seed: int = 0,
                           augment: bool = False):
    """MNIST loaders for CNN / ViT (28 px, 1 channel) or pretrained (224, RGB)."""
    if rgb:
        tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        ops = []
        if augment:
            ops.append(transforms.RandomCrop(img_size, padding=2))
        ops += [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
        tf = transforms.Compose(ops)

    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    train_ds = _subsample(train_ds, train_subset, seed, 0)
    test_ds = _subsample(test_ds, test_subset, seed, 1)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin,
                              drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=max(batch_size, 128),
                             shuffle=False, num_workers=num_workers,
                             pin_memory=pin)
    meta = {
        "train_size": len(train_ds),
        "test_size": len(test_ds),
        "img_size": img_size,
        "channels": 3 if rgb else 1,
        "classes": list(range(10)),
    }
    return train_loader, test_loader, meta


class CanvasMNIST(Dataset):
    """A digit pasted at a deterministic random position on a square canvas.

    Each sample has exactly one box and one label (digits 1..10; 0 is the
    detector background class).
    """

    def __init__(self, data_dir: str, train: bool, canvas: int = 128,
                 subset: int = 0, seed: int = 0, download: bool = True):
        self.base = datasets.MNIST(data_dir, train=train, download=download)
        if subset and subset < len(self.base):
            g = torch.Generator().manual_seed(seed + (0 if train else 1))
            idx = torch.randperm(len(self.base), generator=g)[:subset].tolist()
            self.indices = idx
        else:
            self.indices = list(range(len(self.base)))
        self.canvas = canvas
        self.seed = seed
        self.split = 0 if train else 1
        self.margin = 4

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        img, label = self.base[self.indices[i]]
        arr = np.asarray(img, dtype=np.uint8)  # 28x28
        h, w = arr.shape
        rng = np.random.RandomState(
            (self.seed + 1) * 1_000_003 + self.split * 100_003 + i)
        x = int(rng.randint(self.margin, self.canvas - w - self.margin + 1))
        y = int(rng.randint(self.margin, self.canvas - h - self.margin + 1))
        canvas = np.zeros((self.canvas, self.canvas), dtype=np.uint8)
        canvas[y:y + h, x:x + w] = arr
        image = torch.from_numpy(np.stack([canvas] * 3)).float() / 255.0
        target = {
            "boxes": torch.tensor([[x, y, x + w, y + h]], dtype=torch.float32),
            "labels": torch.tensor([int(label) + 1], dtype=torch.int64),
        }
        return image, target


def detection_loaders(data_dir: str, batch_size: int,
                      train_subset: int = 8000, test_subset: int = 2000,
                      canvas: int = 128, num_workers: int = 2,
                      seed: int = 0):
    train_ds = CanvasMNIST(data_dir, train=True, canvas=canvas,
                           subset=train_subset, seed=seed)
    test_ds = CanvasMNIST(data_dir, train=False, canvas=canvas,
                          subset=test_subset, seed=seed)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin,
                              collate_fn=lambda b: tuple(zip(*b)))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin,
                             collate_fn=lambda b: tuple(zip(*b)))
    meta = {
        "train_size": len(train_ds),
        "test_size": len(test_ds),
        "canvas": canvas,
        "classes": list(range(1, 11)),
    }
    return train_loader, test_loader, meta
