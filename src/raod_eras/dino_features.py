from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .priors import normalize_score, trapezoid_road_prior


@dataclass
class DINOEncoder:
    model_name: str = "dinov2_vits14"
    input_size: int = 518

    def __post_init__(self) -> None:
        try:
            import torch
            import torchvision.transforms as transforms
        except Exception as exc:
            raise RuntimeError("PyTorch and torchvision are required for DINO experiments.") from exc
        self.torch = torch
        self.transforms = transforms
        self.model = torch.hub.load("facebookresearch/dinov2", self.model_name)
        self.model.eval()

    def patch_features(self, image: Image.Image, input_size: int | None = None) -> np.ndarray:
        size = input_size or self.input_size
        size = max(14, int(round(size / 14)) * 14)
        prep = self.transforms.Compose(
            [
                self.transforms.Resize((size, size)),
                self.transforms.ToTensor(),
                self.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        x = prep(image.convert("RGB")).unsqueeze(0)
        with self.torch.no_grad():
            tokens = self.model.forward_features(x)["x_norm_patchtokens"][0]
        feats = tokens.cpu().numpy()
        grid = int(np.sqrt(feats.shape[0]))
        return feats.reshape(grid, grid, -1)


def road_prototype_heatmap(encoder: DINOEncoder, image: Image.Image, out_shape: tuple[int, int]) -> np.ndarray:
    return _road_prototype_heatmap(encoder, image, out_shape, encoder.input_size)


def multiscale_road_prototype_heatmap(
    encoder: DINOEncoder,
    image: Image.Image,
    out_shape: tuple[int, int],
    scales: tuple[float, ...] = (0.75, 1.0),
) -> np.ndarray:
    heatmaps = [
        _road_prototype_heatmap(encoder, image, out_shape, int(encoder.input_size * scale))
        for scale in scales
    ]
    return normalize_score(np.mean(heatmaps, axis=0))


def _road_prototype_heatmap(
    encoder: DINOEncoder,
    image: Image.Image,
    out_shape: tuple[int, int],
    input_size: int,
) -> np.ndarray:
    feats = encoder.patch_features(image, input_size=input_size)
    grid = feats.shape[0]
    road = trapezoid_road_prior((grid, grid)) > 0
    proto = feats[road].mean(axis=0)
    proto = proto / (np.linalg.norm(proto) + 1e-8)

    flat = feats.reshape(-1, feats.shape[-1])
    flat = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8)
    score = 1.0 - flat @ proto
    score = normalize_score(score.reshape(grid, grid))

    h, w = out_shape
    score_img = Image.fromarray((score * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)
    return normalize_score(np.asarray(score_img))
