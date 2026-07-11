from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MetricConfig:
    threshold_min: float = 0.05
    threshold_max: float = 0.95
    threshold_steps: int = 37


def average_precision(score: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    s = score[valid].reshape(-1).astype(np.float32)
    y = gt[valid].reshape(-1).astype(bool)
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(y.sum()), 1)
    prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - prev) * precision))


def threshold_metrics(score: np.ndarray, gt: np.ndarray, valid: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (score >= threshold) & valid
    target = gt & valid
    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target & valid).sum()
    fn = np.logical_and(~pred & valid, target).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    iou = tp / max(tp + fp + fn, 1)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
    }


def best_f1_metrics(score: np.ndarray, gt: np.ndarray, valid: np.ndarray, config: MetricConfig) -> dict[str, float]:
    best = {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}
    for threshold in np.linspace(config.threshold_min, config.threshold_max, config.threshold_steps):
        current = threshold_metrics(score, gt, valid, float(threshold))
        if current["f1"] > best["f1"]:
            best = current
    return best


def fpr_at_95_tpr(score: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    s = score[valid].reshape(-1).astype(np.float32)
    y = gt[valid].reshape(-1).astype(bool)
    pos = int(y.sum())
    neg = int((~y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    tpr = tp / pos
    idx = int(np.searchsorted(tpr, 0.95, side="left"))
    idx = min(idx, len(fp) - 1)
    return float(fp[idx] / neg)


def evaluate_heatmap(score: np.ndarray, gt: np.ndarray, valid: np.ndarray, config: MetricConfig) -> dict[str, float]:
    out = best_f1_metrics(score, gt, valid, config)
    out["ap"] = average_precision(score, gt, valid)
    out["fpr95"] = fpr_at_95_tpr(score, gt, valid)
    return out
