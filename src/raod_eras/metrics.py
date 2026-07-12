from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .refinement import connected_components


@dataclass
class MetricConfig:
    threshold_min: float = 0.05
    threshold_max: float = 0.95
    threshold_steps: int = 37


class PixelMetricAccumulator:
    """Streaming dataset-level metrics without retaining every score map."""

    def __init__(self, threshold: float = 0.5, bins: int = 2048) -> None:
        self.threshold = threshold
        self.bins = bins
        self.positive_hist = np.zeros(bins, dtype=np.int64)
        self.negative_hist = np.zeros(bins, dtype=np.int64)
        self.tp = self.fp = self.fn = 0

    def update(self, score: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> None:
        values = np.clip(score[valid].astype(np.float32), 0.0, 1.0)
        target = gt[valid].astype(bool)
        indices = np.minimum((values * (self.bins - 1)).astype(np.int64), self.bins - 1)
        self.positive_hist += np.bincount(indices[target], minlength=self.bins)
        self.negative_hist += np.bincount(indices[~target], minlength=self.bins)
        pred = values >= self.threshold
        self.tp += int(np.logical_and(pred, target).sum())
        self.fp += int(np.logical_and(pred, ~target).sum())
        self.fn += int(np.logical_and(~pred, target).sum())

    def compute(self) -> dict[str, float]:
        precision = self.tp / max(self.tp + self.fp, 1)
        recall = self.tp / max(self.tp + self.fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        iou = self.tp / max(self.tp + self.fp + self.fn, 1)
        tp_curve = np.cumsum(self.positive_hist[::-1])
        fp_curve = np.cumsum(self.negative_hist[::-1])
        positives = int(self.positive_hist.sum())
        negatives = int(self.negative_hist.sum())
        curve_precision = tp_curve / np.maximum(tp_curve + fp_curve, 1)
        curve_recall = tp_curve / max(positives, 1)
        previous_recall = np.concatenate([[0.0], curve_recall[:-1]])
        ap = float(np.sum((curve_recall - previous_recall) * curve_precision))
        if positives and negatives:
            index = min(int(np.searchsorted(curve_recall, 0.95, side="left")), len(fp_curve) - 1)
            fpr95 = float(fp_curve[index] / negatives)
        else:
            fpr95 = float("nan")
        return {
            "threshold": float(self.threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "iou": float(iou),
            "ap": ap,
            "fpr95": fpr95,
            "positive_pixels": float(positives),
            "negative_pixels": float(negatives),
        }


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
    return {"threshold": float(threshold), **binary_metrics(pred, gt, valid)}


def binary_metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    pred = pred.astype(bool) & valid
    target = gt & valid
    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target & valid).sum()
    fn = np.logical_and(~pred & valid, target).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    iou = tp / max(tp + fp + fn, 1)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
    }


def component_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
    match_iou: float = 0.10,
) -> dict[str, float]:
    pred_labels, pred_components = connected_components(pred & valid)
    gt_labels, gt_components = connected_components(gt & valid)
    del pred_components, gt_components
    pred_ids = [int(value) for value in np.unique(pred_labels) if value > 0]
    gt_ids = [int(value) for value in np.unique(gt_labels) if value > 0]
    matches: list[tuple[float, int, int]] = []
    for pred_id in pred_ids:
        pred_mask = pred_labels == pred_id
        for gt_id in np.unique(gt_labels[pred_mask]):
            if gt_id <= 0:
                continue
            gt_mask = gt_labels == gt_id
            intersection = int(np.logical_and(pred_mask, gt_mask).sum())
            union = int(np.logical_or(pred_mask, gt_mask).sum())
            iou = intersection / max(union, 1)
            if iou >= match_iou:
                matches.append((iou, pred_id, int(gt_id)))
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    for _, pred_id, gt_id in sorted(matches, reverse=True):
        if pred_id not in used_pred and gt_id not in used_gt:
            used_pred.add(pred_id)
            used_gt.add(gt_id)
    tp = len(used_pred)
    precision = tp / max(len(pred_ids), 1)
    recall = tp / max(len(gt_ids), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "component_precision": float(precision),
        "component_recall": float(recall),
        "component_f1": float(f1),
        "pred_components": float(len(pred_ids)),
        "gt_components": float(len(gt_ids)),
    }


def boundary_f1(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, tolerance: int = 2) -> float:
    pred = pred.astype(bool) & valid
    gt = gt.astype(bool) & valid
    pred_boundary = pred ^ ndimage.binary_erosion(pred)
    gt_boundary = gt ^ ndimage.binary_erosion(gt)
    structure = ndimage.generate_binary_structure(2, 1)
    pred_near = ndimage.binary_dilation(pred_boundary, structure=structure, iterations=tolerance)
    gt_near = ndimage.binary_dilation(gt_boundary, structure=structure, iterations=tolerance)
    precision = np.logical_and(pred_boundary, gt_near).sum() / max(pred_boundary.sum(), 1)
    recall = np.logical_and(gt_boundary, pred_near).sum() / max(gt_boundary.sum(), 1)
    return float(2 * precision * recall / max(precision + recall, 1e-8))


def evaluate_binary(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    out = {f"fixed_{key}": value for key, value in binary_metrics(pred, gt, valid).items()}
    out.update(component_metrics(pred, gt, valid))
    out["boundary_f1"] = boundary_f1(pred, gt, valid)
    return out


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
