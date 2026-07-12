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
    if name == "unified":
        return DatasetConfig(
            name="unified_road_anomaly_eval",
            image_dir=Path("data/unified_road_anomaly_eval/images"),
            gt_dir=Path("data/unified_road_anomaly_eval/gt_binary"),
            image_glob="*.*",
            gt_suffix=".png",
            positive_label=255,
            ignore_label=-1,
        )
    raise ValueError(f"Unknown dataset: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full RaOD-ERAS research experiment.")
    parser.add_argument(
        "--dataset",
        choices=["smiyc", "road_anomaly", "street_hazards", "unified"],
        default="smiyc",
    )
    parser.add_argument("--no-dino", action="store_true", help="Disable DINOv2 and run only lightweight baselines.")
    parser.add_argument("--max-samples", type=int, default=None, help="Run a quick subset for debugging.")
    parser.add_argument(
        "--sample-strategy",
        choices=["stratified", "head"],
        default="stratified",
        help="Use round-robin source sampling for unified subsets (default), or legacy head sampling.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output directory.")
    parser.add_argument(
        "--output-threshold",
        type=float,
        default=0.5,
        help="Fixed inference threshold used for binary masks and warning events.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.output_threshold <= 1.0:
        parser.error("--output-threshold must be between 0 and 1.")
    out_dir = args.out or Path(f"outputs/research_experiment_{args.dataset}")

    config = ExperimentConfig(
        root=ROOT,
        dataset=dataset_config(args.dataset),
        method=MethodConfig(use_dino=not args.no_dino, output_threshold=args.output_threshold),
        output=OutputConfig(output_dir=out_dir),
        max_samples=args.max_samples,
        sample_strategy=args.sample_strategy,
    )
    summary = run_experiment(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
