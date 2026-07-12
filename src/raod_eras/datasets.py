from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import DatasetConfig


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image_path: Path
    gt_path: Path


@dataclass
class LoadedSample:
    sample: Sample
    image: Image.Image
    rgb: np.ndarray
    gt: np.ndarray
    valid: np.ndarray


class SMIYCRoadObstacleDataset:
    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    def list_samples(self) -> list[Sample]:
        samples: list[Sample] = []
        for image_path in sorted(self.config.image_dir.glob(self.config.image_glob)):
            gt_path = self.config.gt_dir / f"{image_path.stem}{self.config.gt_suffix}"
            if gt_path.exists():
                samples.append(Sample(image_path.stem, image_path, gt_path))
        if not samples:
            raise FileNotFoundError(f"No image/GT pairs found under {self.config.image_dir}")
        return samples

    def load(self, sample: Sample) -> LoadedSample:
        image = Image.open(sample.image_path).convert("RGB")
        gt_arr = np.asarray(Image.open(sample.gt_path).convert("L"))
        gt = gt_arr == self.config.positive_label
        valid = gt_arr != self.config.ignore_label
        rgb = np.asarray(image.resize((gt.shape[1], gt.shape[0]), Image.Resampling.BILINEAR))
        return LoadedSample(sample=sample, image=image, rgb=rgb, gt=gt, valid=valid)
