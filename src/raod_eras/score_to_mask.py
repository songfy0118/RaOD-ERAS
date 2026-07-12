from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .priors import ego_lane_prior, trapezoid_road_prior


@dataclass(frozen=True)
class PromptConfig:
    threshold: float = 0.70
    min_area_ratio: float = 0.00005
    max_area_ratio: float = 0.12
    box_expansion: float = 0.08
    max_prompts: int = 12
    point_threshold: float = 0.70
    max_point_prompts: int = 50
    point_candidate_limit: int = 4096
    mask_nms_iou: float = 0.70
    min_mask_score: float = 0.38


@dataclass(frozen=True)
class PromptResult:
    mask: np.ndarray
    refined_score: np.ndarray
    boxes: tuple[tuple[int, int, int, int], ...] = ()
    points: tuple[tuple[int, int], ...] = ()
    point_threshold: float = 0.70


def score_boxes(score: np.ndarray, config: PromptConfig, risk_aware: bool) -> list[tuple[int, int, int, int]]:
    h, w = score.shape
    road = trapezoid_road_prior((h, w))
    lane = ego_lane_prior((h, w))
    if risk_aware:
        roi = road > 0.20
        values = score[roi]
        adaptive = float(np.quantile(values, 0.985)) if values.size else config.threshold
        threshold = float(np.clip(max(0.58, min(config.threshold, adaptive)), 0.0, 1.0))
    else:
        threshold = config.threshold

    seeds = score >= threshold
    seeds = ndimage.binary_closing(seeds, structure=np.ones((3, 3), dtype=bool))
    labels, count = ndimage.label(seeds)
    image_area = h * w
    min_area = max(4, int(config.min_area_ratio * image_area))
    max_area = int(config.max_area_ratio * image_area)
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for label in range(1, count + 1):
        ys, xs = np.nonzero(labels == label)
        area = len(xs)
        if area < min_area or area > max_area:
            continue
        mean_score = float(score[ys, xs].mean())
        road_overlap = float(road[ys, xs].mean())
        lane_overlap = float(lane[ys, xs].mean())
        if risk_aware and road_overlap < 0.12:
            continue
        rank = mean_score if not risk_aware else mean_score * (0.70 + 0.20 * road_overlap + 0.10 * lane_overlap)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        pad = int(config.box_expansion * max(x1 - x0 + 1, y1 - y0 + 1))
        candidates.append((rank, (max(0, x0 - pad), max(0, y0 - pad), min(w - 1, x1 + pad), min(h - 1, y1 + pad))))
    candidates.sort(reverse=True)
    boxes = [box for _, box in candidates[: config.max_prompts]]
    if risk_aware and len(boxes) < min(4, config.max_prompts):
        boxes.extend(_peak_boxes(score, road, boxes, config.max_prompts - len(boxes)))
    return boxes[: config.max_prompts]


def _peak_boxes(
    score: np.ndarray,
    road: np.ndarray,
    existing: list[tuple[int, int, int, int]],
    limit: int,
) -> list[tuple[int, int, int, int]]:
    if limit <= 0:
        return []
    h, w = score.shape
    weighted = score * (0.35 + 0.65 * road)
    window = max(9, int(round(0.035 * min(h, w))) | 1)
    maxima = weighted == ndimage.maximum_filter(weighted, size=window, mode="nearest")
    roi_values = weighted[road > 0.15]
    cutoff = float(np.quantile(roi_values, 0.99)) if roi_values.size else 0.75
    ys, xs = np.nonzero(maxima & (weighted >= max(cutoff, 0.55)) & (road > 0.15))
    order = np.argsort(weighted[ys, xs])[::-1]
    half_w = max(10, int(0.035 * w))
    half_h = max(10, int(0.050 * h))
    boxes: list[tuple[int, int, int, int]] = []
    centers = [((x0 + x1) // 2, (y0 + y1) // 2) for x0, y0, x1, y1 in existing]
    for index in order:
        x, y = int(xs[index]), int(ys[index])
        if any((x - cx) ** 2 + (y - cy) ** 2 < window**2 for cx, cy in centers):
            continue
        box = (max(0, x - half_w), max(0, y - half_h), min(w - 1, x + half_w), min(h - 1, y + half_h))
        boxes.append(box)
        centers.append((x, y))
        if len(boxes) >= limit:
            break
    return boxes


def sam_score_to_mask(
    predictor: object,
    rgb: np.ndarray,
    score: np.ndarray,
    config: PromptConfig = PromptConfig(),
    risk_aware: bool = False,
    image_already_set: bool = False,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    boxes = score_boxes(score, config, risk_aware)
    output = np.zeros(score.shape, dtype=bool)
    if not boxes:
        return output, boxes
    if not image_already_set:
        predictor.set_image(rgb)
    for box in boxes:
        masks, sam_scores, _ = predictor.predict(box=np.asarray(box), multimask_output=risk_aware)
        if not risk_aware:
            selected = int(np.argmax(sam_scores))
        else:
            qualities: list[float] = []
            for mask, sam_score in zip(masks, sam_scores):
                inside = float(score[mask].mean()) if mask.any() else 0.0
                ring = ndimage.binary_dilation(mask, iterations=3) & ~mask
                outside = float(score[ring].mean()) if ring.any() else 0.0
                contrast = float(np.clip(inside - outside, -1.0, 1.0))
                qualities.append(0.55 * float(sam_score) + 0.45 * contrast)
            selected = int(np.argmax(qualities))
        output |= masks[selected]
    return output, boxes


def farthest_point_prompts(
    score: np.ndarray,
    config: PromptConfig = PromptConfig(),
    road_aware: bool = False,
) -> list[tuple[int, int]]:
    """Sample spatially diverse high-anomaly pixels, following UGainS FPS prompting."""
    h, w = score.shape
    road = trapezoid_road_prior((h, w))
    lane = ego_lane_prior((h, w))
    eligible = score >= config.point_threshold
    ranking = score.copy()
    if road_aware:
        eligible &= road > 0.15
        ranking *= 0.65 + 0.25 * road + 0.10 * lane
    ys, xs = np.nonzero(eligible)
    if len(xs) == 0:
        return []

    values = ranking[ys, xs]
    if len(xs) > config.point_candidate_limit:
        keep = np.argpartition(values, -config.point_candidate_limit)[-config.point_candidate_limit :]
        xs, ys, values = xs[keep], ys[keep], values[keep]
    coordinates = np.column_stack((xs / max(w - 1, 1), ys / max(h - 1, 1))).astype(np.float32)
    selected = [int(np.argmax(values))]
    min_distance = np.sum((coordinates - coordinates[selected[0]]) ** 2, axis=1)
    limit = min(config.max_point_prompts, len(xs))
    while len(selected) < limit:
        weighted_distance = min_distance * (0.50 + 0.50 * values)
        next_index = int(np.argmax(weighted_distance))
        if min_distance[next_index] <= 1e-8:
            break
        selected.append(next_index)
        distance = np.sum((coordinates - coordinates[next_index]) ** 2, axis=1)
        min_distance = np.minimum(min_distance, distance)
    return [(int(xs[index]), int(ys[index])) for index in selected]


def _mask_nms(
    candidates: list[tuple[float, float, np.ndarray]],
    iou_threshold: float,
) -> list[tuple[float, float, np.ndarray]]:
    kept: list[tuple[float, float, np.ndarray]] = []
    for candidate in sorted(candidates, key=lambda item: item[0], reverse=True):
        mask = candidate[2]
        duplicate = False
        for _, _, previous in kept:
            intersection = int(np.logical_and(mask, previous).sum())
            union = int(np.logical_or(mask, previous).sum())
            if intersection / max(union, 1) > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def _feedback_score(score: np.ndarray, candidates: list[tuple[float, float, np.ndarray]]) -> np.ndarray:
    feedback = np.zeros(score.shape, dtype=np.float32)
    support = np.zeros(score.shape, dtype=np.float32)
    for anomaly_score, sam_score, mask in candidates:
        weight = float(np.clip(0.75 * anomaly_score + 0.25 * sam_score, 0.0, 1.0))
        feedback[mask] += weight
        support[mask] += 1.0
    feedback = np.divide(feedback, np.maximum(support, 1.0))
    combined = score * np.where(support > 0, 1.0, 0.82) + 0.35 * feedback
    maximum = float(combined.max())
    minimum = float(combined.min())
    return ((combined - minimum) / max(maximum - minimum, 1e-8)).astype(np.float32)


def _point_mask_candidates(
    predictor: object,
    score: np.ndarray,
    points: list[tuple[int, int]],
    road_aware: bool,
) -> list[tuple[float, float, np.ndarray]]:
    candidates: list[tuple[float, float, np.ndarray]] = []
    road = trapezoid_road_prior(score.shape)
    lane = ego_lane_prior(score.shape)
    for x, y in points:
        masks, sam_scores, _ = predictor.predict(
            point_coords=np.asarray([[x, y]], dtype=np.float32),
            point_labels=np.asarray([1], dtype=np.int32),
            multimask_output=True,
        )
        for mask, sam_score in zip(masks, sam_scores):
            if not mask.any():
                continue
            anomaly_score = float(score[mask].mean())
            if road_aware:
                ring = ndimage.binary_dilation(mask, iterations=3) & ~mask
                outside = float(score[ring].mean()) if ring.any() else 0.0
                contrast = float(np.clip(anomaly_score - outside, -1.0, 1.0))
                spatial = float(0.75 * road[mask].mean() + 0.25 * lane[mask].mean())
                rank = 0.45 * anomaly_score + 0.25 * float(sam_score) + 0.20 * contrast + 0.10 * spatial
            else:
                rank = anomaly_score
            candidates.append((rank, float(sam_score), mask.astype(bool)))
    return candidates


def sam_point_prompts(
    predictor: object,
    score: np.ndarray,
    config: PromptConfig = PromptConfig(),
    road_aware: bool = False,
) -> PromptResult:
    """Convert FPS point prompts into an instance-refined anomaly map and mask."""
    points = farthest_point_prompts(score, config, road_aware=road_aware)
    candidates = _point_mask_candidates(predictor, score, points, road_aware)
    kept = _mask_nms(candidates, config.mask_nms_iou)
    accepted = [item for item in kept if item[0] >= config.min_mask_score]
    output = np.zeros(score.shape, dtype=bool)
    for _, _, mask in accepted:
        output |= mask
    return PromptResult(output, _feedback_score(score, accepted), points=tuple(points))


def sam_risk_box_prompts(
    predictor: object,
    score: np.ndarray,
    config: PromptConfig = PromptConfig(),
) -> PromptResult:
    """Select one boundary-consistent SAM mask per road-aware anomaly box."""
    boxes = score_boxes(score, config, risk_aware=True)
    selected_candidates: list[tuple[float, float, np.ndarray]] = []
    for box in boxes:
        masks, sam_scores, _ = predictor.predict(box=np.asarray(box), multimask_output=True)
        alternatives: list[tuple[float, float, np.ndarray]] = []
        for mask, sam_score in zip(masks, sam_scores):
            if not mask.any():
                continue
            inside = float(score[mask].mean())
            ring = ndimage.binary_dilation(mask, iterations=3) & ~mask
            outside = float(score[ring].mean()) if ring.any() else 0.0
            contrast = float(np.clip(inside - outside, -1.0, 1.0))
            quality = 0.55 * float(sam_score) + 0.45 * contrast
            alternatives.append((quality, float(sam_score), mask.astype(bool)))
        if alternatives:
            selected_candidates.append(max(alternatives, key=lambda item: item[0]))
    kept = _mask_nms(selected_candidates, config.mask_nms_iou)
    output = np.zeros(score.shape, dtype=bool)
    for _, _, mask in kept:
        output |= mask
    return PromptResult(output, _feedback_score(score, kept), boxes=tuple(boxes))


def sam_hybrid_prompts(
    predictor: object,
    score: np.ndarray,
    config: PromptConfig = PromptConfig(),
) -> PromptResult:
    """Road-aware boxes plus FPS points with mask-to-score feedback."""
    boxes = score_boxes(score, config, risk_aware=True)
    candidates: list[tuple[float, float, np.ndarray]] = []
    road = trapezoid_road_prior(score.shape)
    lane = ego_lane_prior(score.shape)
    for box in boxes:
        masks, sam_scores, _ = predictor.predict(box=np.asarray(box), multimask_output=True)
        for mask, sam_score in zip(masks, sam_scores):
            if not mask.any():
                continue
            inside = float(score[mask].mean())
            ring = ndimage.binary_dilation(mask, iterations=3) & ~mask
            outside = float(score[ring].mean()) if ring.any() else 0.0
            contrast = float(np.clip(inside - outside, -1.0, 1.0))
            spatial = float(0.75 * road[mask].mean() + 0.25 * lane[mask].mean())
            rank = 0.40 * inside + 0.25 * float(sam_score) + 0.25 * contrast + 0.10 * spatial
            candidates.append((rank, float(sam_score), mask.astype(bool)))

    points = farthest_point_prompts(score, config, road_aware=True)
    candidates.extend(_point_mask_candidates(predictor, score, points, road_aware=True))
    kept = _mask_nms(candidates, config.mask_nms_iou)
    accepted = [item for item in kept if item[0] >= config.min_mask_score]
    output = np.zeros(score.shape, dtype=bool)
    for _, _, mask in accepted:
        output |= mask
    return PromptResult(
        output,
        _feedback_score(score, accepted),
        boxes=tuple(boxes),
        points=tuple(points),
    )
