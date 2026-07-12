from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METHODS = ("A_basic_box", "B_road_box", "C_boundary_select", "D_full_feedback")
STEPS = (
    ("B_road_box", "A_basic_box", "road-aware boxes"),
    ("C_boundary_select", "B_road_box", "boundary-consistent selection"),
    ("D_full_feedback", "C_boundary_select", "mask feedback"),
)
BINARY_METRICS = ("fixed_precision", "fixed_recall", "fixed_f1", "fixed_iou", "component_f1", "boundary_f1")


def bootstrap(values: np.ndarray, iterations: int, seed: int) -> dict[str, float]:
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(iterations, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return {
        "n": int(len(values)),
        "mean_difference": float(values.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "probability_positive": float((means > 0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = json.loads(args.results.read_text(encoding="utf-8"))
    rows = result["per_image"]
    scopes = {"aggregate": rows}
    for source in sorted({str(row["source"]) for row in rows}):
        scopes[source] = [row for row in rows if row["source"] == source]

    statistics: dict[str, object] = {"binary_steps": {}, "feedback_heatmap": {}}
    for scope, selected in scopes.items():
        statistics["binary_steps"][scope] = {}
        for method, baseline, label in STEPS:
            statistics["binary_steps"][scope][label] = {}
            for metric in BINARY_METRICS:
                difference = np.asarray(
                    [float(row[f"{method}_{metric}"]) - float(row[f"{baseline}_{metric}"]) for row in selected]
                )
                statistics["binary_steps"][scope][label][metric] = bootstrap(difference, args.iterations, args.seed)
        if all("feedback_ap" in row for row in selected):
            ap_difference = np.asarray([float(row["feedback_ap"]) - float(row["base_ap"]) for row in selected])
            fpr_difference = np.asarray([float(row["base_fpr95"]) - float(row["feedback_fpr95"]) for row in selected])
            statistics["feedback_heatmap"][scope] = {
                "ap_improvement": bootstrap(ap_difference, args.iterations, args.seed),
                "fpr95_reduction": bootstrap(fpr_difference, args.iterations, args.seed),
            }

    output_dir = args.results.parent / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ablation_statistics.json").write_text(json.dumps(statistics, indent=2), encoding="utf-8")

    lines = [
        "# RiskPrompt-SAM sequential ablation",
        "",
        f"Samples: {result['num_samples']}.",
        "",
        "C and D intentionally share the same binary mask. Their difference is evaluated on continuous heatmap AP/FPR95.",
        "",
        "## Pixel-micro main table",
        "",
        "| Variant | Precision | Recall | F1 | IoU | AP | FPR95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = result["aggregate"][method]
        lines.append(
            f"| {method} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} | "
            f"{item['iou']:.4f} | {item['ap']:.4f} | {item['fpr95']:.4f} |"
        )
    lines.extend(["", "## Sequential paired image-macro differences", ""])
    lines.append("| Scope | Module | Metric | Mean difference | 95% CI | P(positive) |")
    lines.append("|---|---|---|---:|---:|---:|")
    for scope, modules in statistics["binary_steps"].items():
        for label, metrics in modules.items():
            for metric in ("fixed_f1", "fixed_iou", "component_f1", "boundary_f1"):
                item = metrics[metric]
                lines.append(
                    f"| {scope} | {label} | {metric} | {item['mean_difference']:.4f} | "
                    f"[{item['ci95_low']:.4f}, {item['ci95_high']:.4f}] | {item['probability_positive']:.3f} |"
                )
    lines.extend(["", "## Feedback heatmap differences", ""])
    lines.append("| Scope | Metric | Mean improvement | 95% CI | P(positive) |")
    lines.append("|---|---|---:|---:|---:|")
    for scope, metrics in statistics["feedback_heatmap"].items():
        for metric, item in metrics.items():
            lines.append(
                f"| {scope} | {metric} | {item['mean_difference']:.4f} | "
                f"[{item['ci95_low']:.4f}, {item['ci95_high']:.4f}] | {item['probability_positive']:.3f} |"
            )
    if not statistics["feedback_heatmap"]:
        base = result["aggregate"]["C_boundary_select"]
        full = result["aggregate"]["D_full_feedback"]
        lines.append(
            f"| aggregate | AP (pixel-micro, no image-bootstrap) | {full['ap'] - base['ap']:.4f} | n/a | n/a |"
        )
        lines.append(
            f"| aggregate | FPR95 reduction (pixel-micro) | {base['fpr95'] - full['fpr95']:.4f} | n/a | n/a |"
        )
    (output_dir / "ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_dir / "ablation_report.md")


if __name__ == "__main__":
    main()
