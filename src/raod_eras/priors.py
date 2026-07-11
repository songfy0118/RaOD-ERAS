from __future__ import annotations

import numpy as np


def normalize_score(score: np.ndarray) -> np.ndarray:
    score = score.astype(np.float32)
    lo = float(np.min(score))
    hi = float(np.max(score))
    if hi - lo < 1e-8:
        return np.zeros_like(score, dtype=np.float32)
    return (score - lo) / (hi - lo)


def trapezoid_road_prior(shape: tuple[int, int], top_y: float = 0.30, top_half_width: float = 0.10) -> np.ndarray:
    h, w = shape
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    half_width = top_half_width + 0.86 * yy
    return ((np.abs(xs) <= half_width) & (yy >= top_y)).astype(np.float32)


def soft_road_prior(shape: tuple[int, int]) -> np.ndarray:
    hard = trapezoid_road_prior(shape)
    return np.where(hard > 0, 1.0, 0.15).astype(np.float32)


def ego_lane_prior(shape: tuple[int, int], width_ratio: float = 0.36) -> np.ndarray:
    h, w = shape
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]
    center = 0.5 * w
    half_width = (w * width_ratio) * (0.25 + 0.75 * yy)
    return (np.abs(xs - center) <= half_width).astype(np.float32)


def near_field_weight(shape: tuple[int, int], min_weight: float = 0.35) -> np.ndarray:
    h, _ = shape
    return np.linspace(min_weight, 1.0, h, dtype=np.float32)[:, None]
