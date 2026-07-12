from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy import ndimage

from .priors import ego_lane_prior, near_field_weight, normalize_score, trapezoid_road_prior


@dataclass(frozen=True)
class ObjectRefinementConfig:
    low_threshold: float = 0.36
    high_threshold: float = 0.62
    min_area_ratio: float = 0.00025
    max_area_ratio: float = 0.10
    min_boundary_contrast: float = -0.03
    min_confidence: float = 0.42
    feedback_gain: float = 0.45
    background_suppression: float = 0.72
    morphology_radius: int = 2
    max_instances: int = 20


@dataclass(frozen=True)
class AnomalyInstance:
    instance_id: int
    bbox_xyxy: tuple[int, int, int, int]
    area: int
    mean_anomaly: float
    boundary_contrast: float
    road_overlap: float
    lane_overlap: float
    near_weight: float
    confidence: float
    uncertainty: float
    distance_proxy: float

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["bbox_xyxy"] = list(self.bbox_xyxy)
        return row


def refine_objects(
    score: np.ndarray,
    config: ObjectRefinementConfig = ObjectRefinementConfig(),
) -> tuple[np.ndarray, np.ndarray, list[AnomalyInstance]]:
    score = normalize_score(score)
    h, w = score.shape
    road = trapezoid_road_prior((h, w))
    lane = ego_lane_prior((h, w))
    near = np.repeat(near_field_weight((h, w)), w, axis=1)

    structure = _disk(config.morphology_radius)
    high_seeds = ndimage.binary_opening(score >= config.high_threshold, structure=_disk(1))
    candidates = score >= config.low_threshold
    candidates = ndimage.binary_closing(candidates, structure=structure)
    candidates = ndimage.binary_fill_holes(candidates)

    labels, components = _seeded_components(high_seeds, candidates)
    records: list[tuple[AnomalyInstance, tuple[np.ndarray, np.ndarray]]] = []
    image_area = h * w
    min_area = max(12, int(config.min_area_ratio * image_area))
    max_area = max(min_area + 1, int(config.max_area_ratio * image_area))

    for label, (ys, xs) in components.items():
        area = int(len(xs))
        if area < min_area or area > max_area or not np.any(high_seeds[ys, xs]):
            continue
        component = labels == label
        ring = ndimage.binary_dilation(component, structure=structure) & ~component
        ring_values = score[ring]
        mean_anomaly = float(score[component].mean())
        boundary_mean = float(ring_values.mean()) if ring_values.size else mean_anomaly
        boundary_contrast = mean_anomaly - boundary_mean
        if boundary_contrast < config.min_boundary_contrast:
            continue

        road_overlap = float(road[component].mean())
        lane_overlap = float(lane[component].mean())
        near_value = float(near[component].mean())
        contrast_term = float(np.clip((boundary_contrast + 0.10) / 0.35, 0.0, 1.0))
        area_ratio = area / image_area
        area_quality = float(np.exp(-max(0.0, area_ratio - 0.025) / 0.035))
        confidence = float(
            np.clip(
                0.44 * mean_anomaly
                + 0.18 * contrast_term
                + 0.14 * road_overlap
                + 0.10 * lane_overlap
                + 0.08 * near_value,
                0.0,
                1.0,
            )
            * area_quality
        )
        if confidence < config.min_confidence:
            continue
        y_bottom = int(ys.max())
        distance_proxy = float(np.clip(1.0 - y_bottom / max(h - 1, 1), 0.0, 1.0))
        records.append(
            (
                AnomalyInstance(
                    instance_id=0,
                    bbox_xyxy=(int(xs.min()), int(ys.min()), int(xs.max()), y_bottom),
                    area=area,
                    mean_anomaly=mean_anomaly,
                    boundary_contrast=boundary_contrast,
                    road_overlap=road_overlap,
                    lane_overlap=lane_overlap,
                    near_weight=near_value,
                    confidence=confidence,
                    uncertainty=1.0 - confidence,
                    distance_proxy=distance_proxy,
                ),
                (ys, xs),
            )
        )

    records.sort(
        key=lambda item: item[0].confidence * item[0].near_weight * (1.0 - item[0].distance_proxy),
        reverse=True,
    )
    records = records[: config.max_instances]
    accepted = np.zeros_like(candidates, dtype=bool)
    instance_confidence = np.zeros_like(score, dtype=np.float32)
    instances: list[AnomalyInstance] = []
    for instance_id, (instance, (ys, xs)) in enumerate(records, start=1):
        instance = replace(instance, instance_id=instance_id)
        instances.append(instance)
        accepted[ys, xs] = True
        instance_confidence[ys, xs] = instance.confidence

    feedback = score * config.background_suppression
    if instances:
        boosted = score * (1.0 + config.feedback_gain * instance_confidence)
        feedback[accepted] = boosted[accepted]
    return normalize_score(feedback), accepted, instances


def _disk(radius: int) -> np.ndarray:
    radius = max(int(radius), 1)
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def _seeded_components(
    high_seeds: np.ndarray,
    candidates: np.ndarray,
) -> tuple[np.ndarray, dict[int, tuple[np.ndarray, np.ndarray]]]:
    seed_labels, num_seeds = ndimage.label(high_seeds)
    if num_seeds == 0:
        return np.zeros_like(seed_labels, dtype=np.int32), {}
    _, nearest = ndimage.distance_transform_edt(~high_seeds, return_indices=True)
    nearest_labels = seed_labels[tuple(nearest)]
    assigned = np.where(candidates, nearest_labels, 0).astype(np.int32)
    labels = np.zeros_like(assigned, dtype=np.int32)
    components: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    output_label = 0
    for seed_label in range(1, num_seeds + 1):
        local_labels, local_count = ndimage.label(assigned == seed_label)
        for local_label in range(1, local_count + 1):
            ys, xs = np.nonzero(local_labels == local_label)
            if not xs.size:
                continue
            output_label += 1
            labels[ys, xs] = output_label
            components[output_label] = (ys, xs)
    return labels, components
