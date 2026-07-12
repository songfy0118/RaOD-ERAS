from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MethodConfig:
    use_roadcontrast: bool = True
    use_dino: bool = True
    use_eras: bool = True
    dino_model: str = "dinov2_vits14"
    dino_input_size: int = 518
    threshold_min: float = 0.05
    threshold_max: float = 0.95
    threshold_steps: int = 37
    output_threshold: float = 0.70
    full_ablations: bool = False
    per_image_rank_metrics: bool = False
    risk_fusion_alpha: float = 0.20
    calibration_sweep: bool = False
    threshold_sweep: bool = False
    min_component_area_ratio: float = 0.00005
    cleanup_sweep: bool = False


@dataclass
class DatasetConfig:
    name: str = "smiyc_road_obstacle"
    image_dir: Path = Path("data/smiyc_road_obstacle/paper_subset/images")
    gt_dir: Path = Path("data/smiyc_road_obstacle/paper_subset/gt")
    image_glob: str = "validation_*.webp"
    gt_suffix: str = "_labels_semantic.png"
    positive_label: int = 1
    ignore_label: int = 255


@dataclass
class OutputConfig:
    output_dir: Path = Path("outputs/research_experiment")
    heatmap_dir: str = "heatmaps"
    binary_dir: str = "binary_masks"
    object_mask_dir: str = "object_masks"
    report_dir: str = "reports"


@dataclass
class ExperimentConfig:
    root: Path = Path.cwd()
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    max_samples: int | None = None
    sample_strategy: str = "stratified"
    sample_offset: int = 0

    def resolve(self) -> "ExperimentConfig":
        self.dataset.image_dir = self._resolve(self.dataset.image_dir)
        self.dataset.gt_dir = self._resolve(self.dataset.gt_dir)
        self.output.output_dir = self._resolve(self.output.output_dir)
        return self

    def _resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else self.root / path
