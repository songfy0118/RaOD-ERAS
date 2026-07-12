from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METHODS = ("threshold", "s2m_style", "ugains_style", "riskprompt")
METRICS = ("fixed_precision", "fixed_recall", "fixed_f1", "fixed_iou", "component_f1", "boundary_f1")


def source_of(sample_id: str) -> str:
    return sample_id.split("__", maxsplit=1)[0]


def bootstrap_difference(
    rows: list[dict[str, object]],
    ours: str,
    baseline: str,
    metric: str,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    difference = np.asarray(
        [float(row[f"{ours}_{metric}"]) - float(row[f"{baseline}_{metric}"]) for row in rows],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(iterations, len(difference)))
    means = difference[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "n": float(len(difference)),
        "mean_difference": float(difference.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "probability_positive": float((means > 0).mean()),
    }


def metric_table(result: dict[str, object], scope: str) -> list[str]:
    values = result["aggregate"] if scope == "aggregate" else result["by_source"][scope]
    lines = [
        f"### {scope}",
        "",
        "| Method | Precision | Recall | F1 | IoU | AP | FPR95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = values[method]
        lines.append(
            f"| {method} | {item['precision']:.4f} | {item['recall']:.4f} | "
            f"{item['f1']:.4f} | {item['iou']:.4f} | {item['ap']:.4f} | {item['fpr95']:.4f} |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper statistics from a completed prompt comparison.")
    parser.add_argument("results", type=Path)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = json.loads(args.results.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = result["per_image"]
    scopes = {"aggregate": rows}
    for source in sorted({source_of(str(row["id"])) for row in rows}):
        scopes[source] = [row for row in rows if source_of(str(row["id"])) == source]

    bootstrap: dict[str, object] = {}
    for scope, selected in scopes.items():
        bootstrap[scope] = {}
        for baseline in ("s2m_style", "ugains_style"):
            bootstrap[scope][baseline] = {
                metric: bootstrap_difference(
                    selected,
                    "riskprompt",
                    baseline,
                    metric,
                    args.iterations,
                    args.seed,
                )
                for metric in METRICS
            }

    seconds = np.asarray([float(row["seconds"]) for row in rows], dtype=np.float64)
    summary = {
        "num_samples": len(rows),
        "source_counts": {scope: len(selected) for scope, selected in scopes.items() if scope != "aggregate"},
        "runtime_seconds": {
            "median": float(np.median(seconds)),
            "p90": float(np.quantile(seconds, 0.90)),
            "mean": float(seconds.mean()),
            "total": float(seconds.sum()),
            "measurement_context": (
                "cached metric reevaluation; not model inference latency"
                if float(np.median(seconds)) < 1.0
                else "end-to-end model inference and evaluation"
            ),
        },
        "bootstrap_paired_macro_difference": bootstrap,
    }

    report_dir = args.results.parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "paper_statistics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    public_summary = {
        "device": result["device"],
        "num_samples": result["num_samples"],
        "aggregate": result["aggregate"],
        "by_source": result["by_source"],
        "fixed_thresholds": result["fixed_thresholds"],
        "protocol": result["protocol"],
        "statistics": summary,
    }
    (args.results.parent / "results_summary.json").write_text(
        json.dumps(public_summary, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# RiskPrompt-SAM experiment statistics",
        "",
        f"Samples: {len(rows)}; source counts: {summary['source_counts']}.",
        "",
        "Dataset-level values below are pixel-micro aggregates. Bootstrap intervals are paired image-macro differences.",
        "",
    ]
    for scope in ("aggregate", *sorted(result["by_source"])):
        lines.extend(metric_table(result, scope))
        lines.append("")
    lines.extend(
        [
            "## Paired bootstrap: RiskPrompt minus baseline",
            "",
            "| Scope | Baseline | Metric | Mean difference | 95% CI | P(diff>0) |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for scope, baselines in bootstrap.items():
        for baseline, metrics in baselines.items():
            for metric in ("fixed_f1", "fixed_iou", "component_f1", "boundary_f1"):
                item = metrics[metric]
                lines.append(
                    f"| {scope} | {baseline} | {metric} | {item['mean_difference']:.4f} | "
                    f"[{item['ci95_low']:.4f}, {item['ci95_high']:.4f}] | {item['probability_positive']:.3f} |"
                )
    lines.extend(
        [
            "",
            "## Runtime",
            "",
            f"Median {summary['runtime_seconds']['median']:.2f}s/image; "
            f"P90 {summary['runtime_seconds']['p90']:.2f}s/image; total {summary['runtime_seconds']['total']:.1f}s.",
        ]
    )
    (report_dir / "paper_statistics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report_dir / "paper_statistics.md")


if __name__ == "__main__":
    main()
