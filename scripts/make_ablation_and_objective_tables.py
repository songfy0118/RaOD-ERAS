from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RUNS = {
    "SMIYC RoadObstacle": ROOT / "outputs" / "research_experiment_smiyc" / "metrics.json",
    "RoadAnomaly21": ROOT / "outputs" / "research_experiment_road_anomaly" / "metrics.json",
    "StreetHazards partial": ROOT / "outputs" / "research_experiment_street_hazards_149" / "metrics.json",
}

METHODS = [
    "roadcontrast",
    "roadcontrast_eras_light",
    "roadcontrast_eras_balanced",
    "roadcontrast_eras_recall",
    "dino",
    "dino_eras_light",
    "dino_eras_balanced",
    "dino_eras_recall",
]


def value(metrics: dict[str, float], method: str, metric: str) -> float | None:
    raw = metrics.get(f"{method}_{metric}")
    return None if raw is None else float(raw)


def objective(metrics: dict[str, float], method: str) -> float | None:
    ap = value(metrics, method, "ap")
    f1 = value(metrics, method, "f1")
    fpr95 = value(metrics, method, "fpr95")
    if ap is None or f1 is None or fpr95 is None:
        return None
    return 0.40 * (1.0 - ap) + 0.35 * (1.0 - f1) + 0.25 * fpr95


def main() -> None:
    out_dir = ROOT / "paper" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_rows: list[dict[str, object]] = []
    md = [
        "# Ablation and Operating-Point Selection",
        "",
        "The validation objective is lower-is-better:",
        "",
        "`L = 0.40 * (1 - AP) + 0.35 * (1 - F1) + 0.25 * FPR95`",
        "",
        "This is not a neural-network training loss. It is the paper's operating-point selection criterion for a training-free pipeline.",
        "",
    ]

    for dataset, path in RUNS.items():
        metrics = json.loads(path.read_text(encoding="utf-8"))
        ranked: list[tuple[float, str]] = []
        for method in METHODS:
            score = objective(metrics, method)
            if score is not None:
                ranked.append((score, method))
                csv_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "objective": score,
                        "ap": value(metrics, method, "ap"),
                        "f1": value(metrics, method, "f1"),
                        "iou": value(metrics, method, "iou"),
                        "precision": value(metrics, method, "precision"),
                        "recall": value(metrics, method, "recall"),
                        "fpr95": value(metrics, method, "fpr95"),
                    }
                )
        ranked.sort()
        md.append(f"## {dataset}")
        md.append("")
        md.append("| Rank | Method | Objective | AP | F1 | IoU | Recall | FPR95 |")
        md.append("|---:|---|---:|---:|---:|---:|---:|---:|")
        for rank, (score, method) in enumerate(ranked[:6], start=1):
            md.append(
                "| {rank} | {method} | {obj:.4f} | {ap:.4f} | {f1:.4f} | {iou:.4f} | {recall:.4f} | {fpr95:.4f} |".format(
                    rank=rank,
                    method=method,
                    obj=score,
                    ap=value(metrics, method, "ap") or 0.0,
                    f1=value(metrics, method, "f1") or 0.0,
                    iou=value(metrics, method, "iou") or 0.0,
                    recall=value(metrics, method, "recall") or 0.0,
                    fpr95=value(metrics, method, "fpr95") or 0.0,
                )
            )
        md.append("")

    fields = ["dataset", "method", "objective", "ap", "f1", "iou", "precision", "recall", "fpr95"]
    with (out_dir / "ablation_objective.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    (out_dir / "ablation_objective.md").write_text("\n".join(md), encoding="utf-8")
    print(out_dir / "ablation_objective.md")


if __name__ == "__main__":
    main()
