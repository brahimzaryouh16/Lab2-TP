"""Classification and detection metrics implemented with NumPy only.

Keeping these local (instead of sklearn/torchmetrics) means the Colab
environment needs nothing beyond torch/torchvision.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     num_classes: int = 10) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        cm[t, p] += 1
    return cm


def precision_recall_f1(cm: np.ndarray) -> dict:
    """Per-class precision/recall/F1 plus macro and weighted averages."""
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)

    return {
        "per_class": {
            "precision": precision.round(4).tolist(),
            "recall": recall.round(4).tolist(),
            "f1": f1.round(4).tolist(),
            "support": support.tolist(),
        },
        "macro_precision": round(float(precision.mean()), 4),
        "macro_recall": round(float(recall.mean()), 4),
        "macro_f1": round(float(f1.mean()), 4),
        "weighted_f1": round(float((f1 * support).sum() / max(support.sum(), 1)), 4),
        "accuracy": round(float(tp.sum() / max(cm.sum(), 1)), 4),
    }


# --------------------------------------------------------------------------
# Detection (one annotated digit per image)
# --------------------------------------------------------------------------
def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two xyxy boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def average_precision(preds: list[dict], gts: list[dict],
                      num_classes: int = 10, iou_thr: float = 0.5) -> dict:
    """All-point AP@IoU per class (COCO-style), averaged over present classes.

    preds[i] = {"boxes": (N,4) xyxy, "scores": (N,), "labels": (N,)}
    gts[i]   = {"boxes": (M,4) xyxy, "labels": (M,)}
    """
    # Flatten predictions, keeping the image index so a GT box can be matched
    # at most once.
    flat = []
    for img_idx, p in enumerate(preds):
        for box, score, label in zip(p["boxes"], p["scores"], p["labels"]):
            flat.append((float(score), img_idx, int(label), np.asarray(box)))
    flat.sort(key=lambda x: -x[0])

    aps = {}
    for cls in range(1, num_classes + 1):
        n_gt = sum(int((g["labels"] == cls).sum()) for g in gts)
        if n_gt == 0:
            continue
        matched: set[tuple[int, int]] = set()
        tp, fp = [], []
        for _, img_idx, label, box in flat:
            if label != cls:
                continue
            g = gts[img_idx]
            best_iou, best_j = 0.0, -1
            for j, (gb, gl) in enumerate(zip(g["boxes"], g["labels"])):
                if int(gl) != cls or (img_idx, j) in matched:
                    continue
                iou = box_iou(box, np.asarray(gb))
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_thr and best_j >= 0:
                matched.add((img_idx, best_j))
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)

        if not tp:
            aps[cls] = 0.0
            continue
        tp_c = np.cumsum(tp).astype(float)
        fp_c = np.cumsum(fp).astype(float)
        recall = tp_c / n_gt
        precision = tp_c / np.maximum(tp_c + fp_c, 1e-9)
        # All-point interpolation: integrate precision over recall steps.
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        aps[cls] = float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())

    return {
        "map50": round(float(np.mean(list(aps.values()))) if aps else 0.0, 4),
        "ap_per_class": {str(k): round(v, 4) for k, v in aps.items()},
        "num_classes_present": len(aps),
    }
