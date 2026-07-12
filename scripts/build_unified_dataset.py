from __future__ import annotations

import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_research_experiment import dataset_config


@dataclass(frozen=True)
class SourceDataset:
    key: str
    display_name: str
    citation_key: str
    positive_label: int


SOURCES = [
    SourceDataset("smiyc", "SMIYC RoadObstacle", "chan2021smiyc", 1),
    SourceDataset("road_anomaly", "RoadAnomaly21", "chan2021smiyc", 1),
    SourceDataset("street_hazards", "StreetHazards partial", "hendrycks2019scaling", 14),
]


def standardized_gt(gt_path: Path, positive_label: int, ignore_label: int) -> tuple[Image.Image, dict[str, int]]:
    gt = np.asarray(Image.open(gt_path).convert("L"))
    valid = gt != ignore_label
    anomaly = gt == positive_label
    out = np.zeros_like(gt, dtype=np.uint8)
    out[anomaly] = 1
    out[~valid] = 255
    stats = {
        "width": int(gt.shape[1]),
        "height": int(gt.shape[0]),
        "anomaly_pixels": int(anomaly.sum()),
        "valid_pixels": int(valid.sum()),
    }
    return Image.fromarray(out), stats


def main() -> None:
    out_root = ROOT / "data" / "unified_road_anomaly_eval"
    image_out = out_root / "images"
    gt_out = out_root / "gt_labels"
    meta_out = out_root / "metadata"
    for path in (image_out, gt_out, meta_out):
        path.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for source in SOURCES:
        cfg = dataset_config(source.key)
        cfg.image_dir = ROOT / cfg.image_dir
        cfg.gt_dir = ROOT / cfg.gt_dir
        for image_path in sorted(cfg.image_dir.glob(cfg.image_glob)):
            gt_path = cfg.gt_dir / f"{image_path.stem}{cfg.gt_suffix}"
            if not gt_path.exists():
                continue
            sample_id = f"{source.key}__{image_path.stem}"
            image_target = image_out / f"{sample_id}{image_path.suffix.lower()}"
            gt_target = gt_out / f"{sample_id}.png"
            shutil.copy2(image_path, image_target)
            gt_img, stats = standardized_gt(gt_path, source.positive_label, cfg.ignore_label)
            gt_img.save(gt_target)
            rows.append(
                {
                    "sample_id": sample_id,
                    "source_dataset": source.display_name,
                    "source_key": source.key,
                    "citation_key": source.citation_key,
                    "image_path": image_target.relative_to(out_root).as_posix(),
                    "gt_label_path": gt_target.relative_to(out_root).as_posix(),
                    **stats,
                }
            )

    rows.sort(key=lambda row: str(row["sample_id"]))
    with (meta_out / "samples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (meta_out / "samples.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    summary = {
        "name": "Unified Road Anomaly Evaluation Set",
        "num_samples": len(rows),
        "sources": [
            {
                "key": source.key,
                "name": source.display_name,
                "citation_key": source.citation_key,
                "num_samples": sum(1 for row in rows if row["source_key"] == source.key),
            }
            for source in SOURCES
        ],
        "note": "This is a standardized evaluation split assembled from public datasets; it is not a newly collected dataset.",
        "label_encoding": {"normal": 0, "anomaly": 1, "ignore": 255},
        "protocol": "Original dataset void pixels remain ignored during evaluation.",
    }
    (meta_out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
