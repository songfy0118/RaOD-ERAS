"""Ground-truth-free inference using the public experiment's shared front end."""
from __future__ import annotations

import numpy as np
from PIL import Image

from .baselines import road_contrast_heatmap
from .dino_features import multiscale_road_prototype_heatmap, road_prototype_heatmap
from .priors import ego_lane_prior, near_field_weight, normalize_score, trapezoid_road_prior
from .score_to_mask import PromptConfig, sam_box_ablation


def anomaly_score(encoder, image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    width, height = image.size
    rgb = np.asarray(image)
    dino = road_prototype_heatmap(encoder, image, (height, width))
    multiscale = multiscale_road_prototype_heatmap(encoder, image, (height, width))
    local = road_contrast_heatmap(rgb)
    road = trapezoid_road_prior((height, width))
    lane = ego_lane_prior((height, width))
    near = np.repeat(near_field_weight((height, width)), width, axis=1)
    candidate = normalize_score((0.78 * multiscale + 0.22 * local) * (0.62 + 0.20 * road + 0.18 * lane * near))
    return normalize_score(0.80 * dino + 0.20 * candidate)


def predict_image(encoder, predictor, image: Image.Image, config: PromptConfig | None = None):
    """Return continuous scores and the full A-D ablation's D mask. No GT input."""
    image = image.convert("RGB")
    score = anomaly_score(encoder, image)
    expected_shape = (image.height, image.width)
    if score.shape != expected_shape or not np.isfinite(score).all():
        raise ValueError("Anomaly score must be finite and match the processed image shape.")
    predictor.set_image(np.asarray(image))
    result = sam_box_ablation(
        predictor, score, config or PromptConfig(),
        road_aware=True, boundary_aware=True, feedback=True,
    )
    if result.mask.shape != expected_shape or result.refined_score.shape != expected_shape:
        raise ValueError("Prediction arrays do not match the processed image shape.")
    if result.mask.dtype != np.bool_ or not np.isfinite(result.refined_score).all():
        raise ValueError("Expected a boolean mask and finite refined scores.")
    return score, result
