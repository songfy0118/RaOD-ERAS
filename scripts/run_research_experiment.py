from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.raod_eras.config import DatasetConfig, ExperimentConfig, MethodConfig, OutputConfig
from src.raod_eras.experiment import run_experiment


def dataset_config(name: str) -> DatasetConfig:
    if name == "smiyc":
        return DatasetConfig(
            name="smiyc_road_obstacle",
            image_dir=Path("data/smiyc_road_obstacle/paper_subset/images"),
            gt_dir=Path("data/smiyc_road_obstacle/paper_subset/gt"),
            image_glob="validation_*.webp",
            gt_suffix="_labels_semantic.png",
        )
    if name == "road_anomaly":
        return DatasetConfig(
            name="road_anomaly21",
            image_dir=Path("data/road_anomaly/paper_subset/images"),
            gt_dir=Path("data/road_anomaly/paper_subset/gt"),
            image_glob="validation*.jpg",
            gt_suffix="_labels_semantic.png",
        )
    if name == "street_hazards":
        return DatasetConfig(
            name="street_hazards_partial",
            image_dir=Path("data/street_hazards/paper_subset/images"),
            gt_dir=Path("data/street_hazards/paper_subset/gt"),
            image_glob="*.png",
            gt_suffix="_labels_semantic.png",
            positive_label=14,
            ignore_label=255,
        )
    raise ValueError(f"Unknown dataset: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full RaOD-ERAS research experiment.")
    parser.add_argument("--dataset", choices=["smiyc", "road_anomaly", "street_hazards"], default="smiyc")
    parser.add_argument("--no-dino", action="store_true", help="Disable DINOv2 and run only lightweight baselines.")
    parser.add_argument("--max-samples", type=int, default=None, help="Run a quick subset for debugging.")
    parser.add_argument("--out", type=Path, default=None, help="Output directory.")
    args = parser.parse_args()
    out_dir = args.out or Path(f"outputs/research_experiment_{args.dataset}")

    config = ExperimentConfig(
        root=ROOT,
        dataset=dataset_config(args.dataset),
        method=MethodConfig(use_dino=not args.no_dino),
        output=OutputConfig(output_dir=out_dir),
        max_samples=args.max_samples,
    )
    summary = run_experiment(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
