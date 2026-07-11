from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .priors import ego_lane_prior, near_field_weight, normalize_score, trapezoid_road_prior


@dataclass
class ERASVariant:
    name: str
    road_boost: float
    lane_boost: float
    component_weight: float
    threshold: float
    min_area: int = 12
    max_area_ratio: float = 0.12


DEFAULT_ERAS_VARIANTS = [
    ERASVariant("eras_light", road_boost=0.25, lane_boost=0.20, component_weight=0.15, threshold=0.55),
    ERASVariant("eras_balanced", road_boost=0.45, lane_boost=0.35, component_weight=0.25, threshold=0.50),
    ERASVariant("eras_recall", road_boost=0.75, lane_boost=0.55, component_weight=0.35, threshold=0.45),
]


def refine_heatmap(score: np.ndarray, variant: ERASVariant) -> tuple[np.ndarray, list[dict[str, float]]]:
    score = normalize_score(score)
    h, w = score.shape
    road = trapezoid_road_prior((h, w))
    lane = ego_lane_prior((h, w))
    y_weight = near_field_weight((h, w))

    refined = score * (1.0 + variant.road_boost * road) * (1.0 + variant.lane_boost * lane * y_weight)
    refined = normalize_score(refined)

    components = score_components(refined, road, lane, variant)
    rescored = np.zeros_like(refined, dtype=np.float32)
    for comp in components:
        y0, y1, x0, x1 = int(comp["y0"]), int(comp["y1"]), int(comp["x0"]), int(comp["x1"])
        region = refined[y0 : y1 + 1, x0 : x1 + 1]
        mask = region >= variant.threshold
        rescored[y0 : y1 + 1, x0 : x1 + 1][mask] = max(float(comp["risk_score"]), 0.0)

    final = normalize_score((1.0 - variant.component_weight) * refined + variant.component_weight * normalize_score(rescored))
    return final, components


def score_components(score: np.ndarray, road: np.ndarray, lane: np.ndarray, variant: ERASVariant) -> list[dict[str, float]]:
    labels, comps = connected_components(score >= variant.threshold)
    del labels
    h, w = score.shape
    image_area = h * w
    max_area = variant.max_area_ratio * image_area
    target_area = max(1.0, 0.006 * image_area)
    out: list[dict[str, float]] = []
    for label, (ys, xs) in comps.items():
        area = len(xs)
        if area < variant.min_area or area > max_area:
            continue
        mean_score = float(score[ys, xs].mean())
        road_overlap = float(road[ys, xs].mean())
        lane_overlap = float(lane[ys, xs].mean())
        size_prior = float(np.exp(-abs(np.log(area + 1.0) - np.log(target_area + 1.0)) / 1.25))
        risk_score = mean_score * (1.0 + variant.road_boost * road_overlap) * (1.0 + variant.lane_boost * lane_overlap) * size_prior
        out.append(
            {
                "label": float(label),
                "area": float(area),
                "mean_score": mean_score,
                "road_overlap": road_overlap,
                "lane_overlap": lane_overlap,
                "size_prior": size_prior,
                "risk_score": float(risk_score),
                "x0": float(xs.min()),
                "y0": float(ys.min()),
                "x1": float(xs.max()),
                "y1": float(ys.max()),
            }
        )
    out.sort(key=lambda item: item["risk_score"], reverse=True)
    return out


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, dict[int, tuple[np.ndarray, np.ndarray]]]:
    try:
        from scipy import ndimage

        labels, num_labels = ndimage.label(mask.astype(bool))
        objects = ndimage.find_objects(labels)
        packed: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for label in range(1, num_labels + 1):
            obj = objects[label - 1]
            if obj is None:
                continue
            local = labels[obj] == label
            local_ys, local_xs = np.nonzero(local)
            y_slice, x_slice = obj
            packed[label] = (local_ys + y_slice.start, local_xs + x_slice.start)
        return labels.astype(np.int32, copy=False), packed
    except Exception:
        pass

    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    comps: dict[int, tuple[list[int], list[int]]] = {}
    label = 0
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or labels[y, x] != 0:
                continue
            label += 1
            stack = [(y, x)]
            labels[y, x] = label
            ys: list[int] = []
            xs: list[int] = []
            while stack:
                cy, cx = stack.pop()
                ys.append(cy)
                xs.append(cx)
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = label
                        stack.append((ny, nx))
            comps[label] = (ys, xs)
    packed = {k: (np.asarray(v[0], dtype=np.int32), np.asarray(v[1], dtype=np.int32)) for k, v in comps.items()}
    return labels, packed
