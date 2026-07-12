from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .object_refinement import AnomalyInstance
from .priors import ego_lane_prior, near_field_weight, normalize_score, trapezoid_road_prior


@dataclass(frozen=True)
class TrajectoryScore:
    name: str
    total_cost: float
    anomaly_cost: float
    lane_cost: float
    minimum_clearance: float
    feasible: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def plan_risk_response(
    heatmap: np.ndarray,
    instances: list[AnomalyInstance],
) -> dict[str, object]:
    heatmap = normalize_score(heatmap)
    h, w = heatmap.shape
    road = trapezoid_road_prior((h, w))
    lane = ego_lane_prior((h, w))
    near = np.repeat(near_field_weight((h, w)), w, axis=1)
    risk_map = normalize_score(heatmap * (0.55 + 0.25 * road + 0.20 * lane) * near)

    candidates = [
        _score_curve("keep_lane", 0.00, risk_map, lane, instances),
        _score_curve("shift_left", -0.22, risk_map, lane, instances),
        _score_curve("shift_right", 0.22, risk_map, lane, instances),
    ]
    safe = [candidate for candidate in candidates if candidate.feasible]
    if safe:
        selected = min(safe, key=lambda item: item.total_cost)
        action = selected.name
    else:
        selected = TrajectoryScore("brake", 1.0, 1.0, 0.0, 0.0, True)
        candidates.append(selected)
        action = "brake_or_stop"

    max_risk = max((instance.confidence * (1.0 - instance.distance_proxy) for instance in instances), default=0.0)
    return {
        "selected_action": action,
        "selected_trajectory": selected.to_dict(),
        "candidate_trajectories": [candidate.to_dict() for candidate in candidates],
        "num_instances": len(instances),
        "maximum_instance_risk": float(max_risk),
        "risk_map_mean": float(risk_map.mean()),
        "scope": "image-plane rule validation; not a vehicle controller",
    }


def _score_curve(
    name: str,
    lateral_offset: float,
    risk_map: np.ndarray,
    lane: np.ndarray,
    instances: list[AnomalyInstance],
) -> TrajectoryScore:
    h, w = risk_map.shape
    ys = np.linspace(int(0.35 * h), h - 1, 80).astype(np.int32)
    progress = (ys - ys.min()) / max(ys.max() - ys.min(), 1)
    xs = (0.5 * w + lateral_offset * w * progress**1.35).astype(np.int32)
    xs = np.clip(xs, 0, w - 1)
    sampled = risk_map[ys, xs]
    anomaly_cost = float(np.quantile(sampled, 0.90))
    lane_cost = float(1.0 - lane[ys, xs].mean())

    minimum_clearance = 1.0
    for instance in instances:
        x0, y0, x1, y1 = instance.bbox_xyxy
        active = (ys >= y0) & (ys <= y1)
        if not np.any(active):
            continue
        center_x = 0.5 * (x0 + x1)
        equivalent_radius = np.sqrt(instance.area / np.pi)
        horizontal = np.maximum(np.abs(xs[active] - center_x) - equivalent_radius, 0.0) / max(w, 1)
        clearance = float(horizontal.min())
        minimum_clearance = min(minimum_clearance, clearance)

    clearance_penalty = float(np.exp(-minimum_clearance / 0.06))
    total_cost = 0.58 * anomaly_cost + 0.17 * lane_cost + 0.25 * clearance_penalty
    feasible = minimum_clearance >= 0.015 and anomaly_cost < 0.72
    return TrajectoryScore(name, float(total_cost), anomaly_cost, lane_cost, minimum_clearance, feasible)
