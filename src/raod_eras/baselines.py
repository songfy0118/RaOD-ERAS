from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from .priors import normalize_score, trapezoid_road_prior


def road_contrast_heatmap(rgb: np.ndarray) -> np.ndarray:
    rgb_f = rgb.astype(np.float32) / 255.0
    h, w, _ = rgb_f.shape
    road = trapezoid_road_prior((h, w))
    sample_region = road.copy()
    sample_region[: int(0.58 * h)] = 0
    road_pixels = rgb_f[sample_region > 0]
    if len(road_pixels) < 100:
        road_pixels = rgb_f[road > 0]

    road_color = np.median(road_pixels, axis=0)
    color_dist = np.linalg.norm(rgb_f - road_color, axis=2)

    gray = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.float32) / 255.0
    blur = np.asarray(
        Image.fromarray((gray * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=7)),
        dtype=np.float32,
    ) / 255.0
    texture = np.abs(gray - blur)
    gy, gx = np.gradient(blur)
    edge = np.sqrt(gx * gx + gy * gy)

    y_weight = np.linspace(0.35, 1.0, h, dtype=np.float32)[:, None]
    score = 0.62 * normalize_score(color_dist) + 0.24 * normalize_score(texture) + 0.14 * normalize_score(edge)
    return normalize_score(score * (0.45 + 0.85 * road) * y_weight)
