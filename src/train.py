"""Train one Lab 2 model and write a JSON report.

Classification:  python src/train.py --model cnn|vit|vgg16|alexnet ...
Detection:       python src/train.py --model frcnn ...
Smoke test:      python src/train.py --model cnn --epochs 1 --max-batches 2

Every run writes a self-describing JSON (config, environment, per-epoch
history, final metrics, timing) so the README tables and figures are
generated from artifacts rather than retyped.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import classification_loaders, detection_loaders  # noqa: E402
from src.metrics import (average_precision, confusion_matrix,  # noqa: E402
                         precision_recall_f1)
from src.models import CLASSIFIERS, ViT, build_frcnn  # noqa: E402
from src.utils import (Timer, count_params, device_name, env_info,  # noqa: E402
                       get_device, save_json, set_seed)

DEFAULTS = {
    "cnn": {"img_size": 28, "rgb": False, "batch_size": 128, "lr": 1e-3,
            "epochs": 8, "weight_decay": 1e-4},
    "vit": {"img_size": 28, "rgb": False, "batch_size": 128, "lr": 3e-4,
            "epochs": 12, "weight_decay": 0.05},
    "vgg16": {"img_size": 224, "rgb": True, "batch_size": 64, "lr": 1e-4,
              "epochs": 4, "weight_decay": 1e-4},
    "alexnet": {"img_size": 224, "rgb": True, "batch_size": 64, "lr": 1e-4,
                "epochs": 4, "weight_decay": 1e-4},
    "frcnn": {"img_size": 128, "rgb": False, "batch_size": 8, "lr": 0.005,
              "epochs": 4, "weight_decay": 5e-4},
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    choices=["cnn", "vit", "vgg16", "alexnet", "frcnn"])
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out", default="")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--weight-decay", type=float, default=-1.0)
    ap.add_argument("--train-subset", type=int, default=0)
    ap.add_argument("--test-subset", type=int, default=0)
    ap.add_argument("--img-size", type=int, default=0)
    ap.add_argument("--canvas", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    ap.add_argument("--max-batches", type=int, default=0,
                    help="limit train/eval batches (smoke test)")
    ap.add_argument("--save", default="", help="optional state_dict path")
    args = ap.parse_args()

    d = DEFAULTS[args.model]
    args.epochs = args.epochs or d["epochs"]
    args.batch_size = args.batch_size or d["batch_size"]
    args.lr = args.lr or d["lr"]
    args.weight_decay = (d["weight_decay"] if args.weight_decay < 0
                         else args.weight_decay)
    if args.img_size:
        d["img_size"] = args.img_size
    if not args.out:
        args.out = str(ROOT / "results" / f"{args.model}.json")
    return args


def _limit(loader, max_batches: int):
    if not max_batches:
        return loader

    class _Limited:
        def __init__(self, base):
            self.base = base

        def __iter__(self):
            for i, batch in enumerate(self.base):
                if i >= max_batches:
                    break
                yield batch

        def __len__(self):
            return min(len(self.base), max_batches)

    return _Limited(loader)


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------
def run_classification(args, device):
    cfg = DEFAULTS[args.model]
    train_loader, test_loader, meta = classification_loaders(
        args.data_dir, args.batch_size, args.train_subset, args.test_subset,
        img_size=cfg["img_size"], rgb=cfg["rgb"], num_workers=args.num_workers,
        seed=args.seed, augment=args.augment)
    train_loader = _limit(train_loader, args.max_batches)
    test_loader = _limit(test_loader, args.max_batches)

    if args.model == "vit":
        model = ViT(img_size=cfg["img_size"], patch_size=4, in_chans=1,
                    num_classes=10)
    elif args.model == "cnn":
        model = CLASSIFIERS["cnn"](num_classes=10)
    elif args.model == "vgg16":
        model = CLASSIFIERS["vgg16"](num_classes=10, pretrained=True)
    else:
        model = CLASSIFIERS["alexnet"](num_classes=10, pretrained=True)
    model.to(device)

    if args.freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        head = model.head if args.model == "vit" else model.classifier[6]
        for p in head.parameters():
            p.requires_grad = True

    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "sgd":
        opt = SGD(params, lr=args.lr, momentum=0.9,
                  weight_decay=args.weight_decay)
    else:
        opt = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    history = []
    best = {"epoch": 0, "accuracy": -1.0, "macro_f1": -1.0, "loss": None}
    t_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_epoch = time.perf_counter()
        running, seen = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            if args.model == "vit":
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * y.size(0)
            seen += y.size(0)
        sched.step()
        train_loss = running / max(seen, 1)

        model.eval()
        preds, labels = [], []
        t_eval = time.perf_counter()
        with torch.no_grad():
            for x, y in test_loader:
                out = model(x.to(device, non_blocking=True))
                preds.append(out.argmax(1).cpu().numpy())
                labels.append(y.numpy())
        eval_sec = time.perf_counter() - t_eval
        y_true = np.concatenate(labels)
        y_pred = np.concatenate(preds)
        cm = confusion_matrix(y_true, y_pred, 10)
        m = precision_recall_f1(cm)

        rec = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "test_accuracy": m["accuracy"],
            "test_macro_f1": m["macro_f1"],
            "lr": round(opt.param_groups[0]["lr"], 8),
            "epoch_sec": round(time.perf_counter() - t_epoch, 2),
        }
        history.append(rec)
        print(f"[{args.model}] epoch {epoch}/{args.epochs} "
              f"loss={train_loss:.4f} acc={m['accuracy']:.4f} "
              f"f1={m['macro_f1']:.4f} ({rec['epoch_sec']:.1f}s)",
              flush=True)
        if m["accuracy"] > best["accuracy"]:
            best = {"epoch": epoch, "accuracy": m["accuracy"],
                    "macro_f1": m["macro_f1"], "loss": round(train_loss, 4)}

    train_sec = time.perf_counter() - t_start
    n_test = len(y_true)

    # Final metrics on the last epoch (also the best for these schedules).
    cm = confusion_matrix(y_true, y_pred, 10)
    final = precision_recall_f1(cm)
    payload = {
        "model": args.model,
        "task": "classification",
        "config": vars(args),
        "environment": env_info(device),
        "params": count_params(model),
        "meta": meta,
        "history": history,
        "best": best,
        "final": final,
        "timing": {
            "train_sec": round(train_sec, 2),
            "eval_sec": round(eval_sec, 2),
            "sec_per_epoch": round(train_sec / args.epochs, 2),
            "ms_per_image": round(1000 * eval_sec / max(n_test, 1), 3),
        },
    }
    if args.save:
        torch.save(model.state_dict(), args.save)
        payload["checkpoint"] = args.save
    return payload


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
def run_detection(args, device):
    train_loader, test_loader, meta = detection_loaders(
        args.data_dir, args.batch_size,
        train_subset=args.train_subset or 8000,
        test_subset=args.test_subset or 2000,
        canvas=args.canvas, num_workers=args.num_workers, seed=args.seed)
    train_loader = _limit(train_loader, args.max_batches)
    test_loader = _limit(test_loader, args.max_batches)

    model = build_frcnn(num_classes=11, pretrained=True,
                        min_size=args.canvas, max_size=args.canvas)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = SGD(params, lr=args.lr, momentum=0.9,
              weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=max(args.epochs, 1))

    history = []
    t_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_epoch = time.perf_counter()
        running = 0.0
        parts = {}
        n_batches = 0
        for images, targets in train_loader:
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True)
                        for k, v in t.items()} for t in targets]
            opt.zero_grad(set_to_none=True)
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            running += float(loss.detach())
            for k, v in loss_dict.items():
                parts[k] = parts.get(k, 0.0) + float(v.detach())
            n_batches += 1
        sched.step()
        train_loss = running / max(n_batches, 1)

        # ---- evaluation -------------------------------------------------
        model.eval()
        t_eval = time.perf_counter()
        preds, gts = [], []
        top_labels, top_boxes, gt_labels, gt_boxes = [], [], [], []
        with torch.no_grad():
            for images, targets in test_loader:
                images = [img.to(device) for img in images]
                outputs = model(images)
                for out, tgt in zip(outputs, targets):
                    pb = out["boxes"].cpu().numpy()
                    ps = out["scores"].cpu().numpy()
                    pl = out["labels"].cpu().numpy()
                    preds.append({"boxes": pb, "scores": ps, "labels": pl})
                    gts.append({"boxes": tgt["boxes"].numpy(),
                                "labels": tgt["labels"].numpy()})
                    if len(ps):
                        k = int(np.argmax(ps))
                        top_boxes.append(pb[k])
                        top_labels.append(int(pl[k]))
                    else:
                        top_boxes.append(np.zeros(4, dtype=np.float32))
                        top_labels.append(0)
                    gt_boxes.append(tgt["boxes"].numpy()[0])
                    gt_labels.append(int(tgt["labels"].numpy()[0]))
        eval_sec = time.perf_counter() - t_eval

        from src.metrics import box_iou
        ious = np.array([box_iou(pb, gb)
                         for pb, gb in zip(top_boxes, gt_boxes)])
        label_ok = np.array([p == g for p, g in zip(top_labels, gt_labels)])
        det_ok = label_ok & (ious >= 0.5)
        ap = average_precision(preds, gts, num_classes=10, iou_thr=0.5)

        # 11-class confusion (0 = no detection / background) for a macro F1
        # comparable with the classifier; the macro average uses digits 1..10.
        cm = confusion_matrix(np.array(gt_labels), np.array(top_labels), 11)
        m11 = precision_recall_f1(cm)
        digit_f1 = float(np.mean(m11["per_class"]["f1"][1:]))

        rec = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "loss_parts": {k: round(v / max(n_batches, 1), 4)
                           for k, v in parts.items()},
            "label_accuracy": round(float(label_ok.mean()), 4),
            "detection_accuracy": round(float(det_ok.mean()), 4),
            "mean_iou": round(float(ious.mean()), 4),
            "map50": ap["map50"],
            "macro_f1": round(digit_f1, 4),
            "lr": round(opt.param_groups[0]["lr"], 8),
            "epoch_sec": round(time.perf_counter() - t_epoch, 2),
        }
        history.append(rec)
        print(f"[frcnn] epoch {epoch}/{args.epochs} loss={train_loss:.4f} "
              f"det_acc={det_ok.mean():.4f} mAP50={ap['map50']:.4f} "
              f"iou={ious.mean():.3f} ({rec['epoch_sec']:.1f}s)", flush=True)

    train_sec = time.perf_counter() - t_start
    n_test = len(gt_labels)
    final = {
        "label_accuracy": history[-1]["label_accuracy"],
        "detection_accuracy": history[-1]["detection_accuracy"],
        "mean_iou": history[-1]["mean_iou"],
        "map50": history[-1]["map50"],
        "macro_f1": history[-1]["macro_f1"],
        "per_class": m11["per_class"],
        "ap_per_class": ap["ap_per_class"],
    }
    payload = {
        "model": "frcnn",
        "task": "detection",
        "config": vars(args),
        "environment": env_info(device),
        "params": count_params(model),
        "meta": meta,
        "history": history,
        "best": max(history, key=lambda r: r["map50"]),
        "final": final,
        "timing": {
            "train_sec": round(train_sec, 2),
            "eval_sec": round(eval_sec, 2),
            "sec_per_epoch": round(train_sec / args.epochs, 2),
            "ms_per_image": round(1000 * eval_sec / max(n_test, 1), 3),
        },
    }
    if args.save:
        torch.save(model.state_dict(), args.save)
        payload["checkpoint"] = args.save
    return payload


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"[env] {device_name(device)} | torch {torch.__version__}", flush=True)
    payload = (run_detection(args, device) if args.model == "frcnn"
               else run_classification(args, device))
    save_json(args.out, payload)
    print("[done] train", flush=True)


if __name__ == "__main__":
    main()
