from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RUNS = {
    "SMIYC RoadObstacle": ROOT / "outputs" / "research_experiment_smiyc" / "metrics.json",
    "RoadAnomaly21": ROOT / "outputs" / "research_experiment_road_anomaly" / "metrics.json",
    "StreetHazards partial": ROOT / "outputs" / "research_experiment_street_hazards_149" / "metrics.json",
}

METHODS = ["roadcontrast", "roadcontrast_eras_light", "dino", "dino_eras_light", "dino_eras_balanced"]
METRICS = ["ap", "f1", "iou", "precision", "recall", "fpr95"]


def fmt(value: float | None, lower_better: bool = False) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def main() -> None:
    out_dir = ROOT / "paper" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Quantitative Digest",
        "",
        "Lower is better for FPR95; higher is better for all other metrics.",
        "",
    ]
    csv_lines = ["dataset,method," + ",".join(METRICS)]
    for dataset, path in RUNS.items():
        metrics = json.loads(path.read_text(encoding="utf-8"))
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append("| Method | AP | F1 | IoU | Precision | Recall | FPR95 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for method in METHODS:
            values = [metrics.get(f"{method}_{name}") for name in METRICS]
            if all(value is None for value in values):
                continue
            lines.append("| " + method + " | " + " | ".join(fmt(v) for v in values) + " |")
            csv_lines.append(dataset + "," + method + "," + ",".join("" if v is None else f"{v:.6f}" for v in values))
        lines.append("")
    (out_dir / "quantitative_digest.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "quantitative_digest.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    print(out_dir / "quantitative_digest.md")


if __name__ == "__main__":
    main()
