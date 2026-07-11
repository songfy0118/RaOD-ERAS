from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def make_method_grid(
    image_paths: list[Path],
    gt_paths: list[Path],
    heatmap_paths: dict[str, list[Path]],
    output_path: Path,
    max_rows: int = 8,
) -> None:
    names = list(heatmap_paths.keys())
    rows = min(max_rows, len(image_paths))
    cols = 2 + len(names)
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows))
    if rows == 1:
        axes = axes[None, :]

    for r in range(rows):
        rgb = np.asarray(Image.open(image_paths[r]).convert("RGB").resize((512, 288), Image.Resampling.BILINEAR))
        gt = np.asarray(Image.open(gt_paths[r]).convert("L").resize((512, 288), Image.Resampling.NEAREST)) == 1
        panels: list[tuple[str, np.ndarray, str]] = [("RGB + GT", rgb, "rgb"), ("GT", gt, "mask")]
        for name in names:
            arr = np.asarray(Image.open(heatmap_paths[name][r]).convert("L").resize((512, 288), Image.Resampling.BILINEAR)) / 255.0
            panels.append((name, arr, "heat"))

        for c, (title, arr, kind) in enumerate(panels):
            ax = axes[r, c]
            if kind == "rgb":
                ax.imshow(arr)
                ax.contour(gt, levels=[0.5], colors=["lime"], linewidths=1.0)
            elif kind == "mask":
                ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(arr, cmap="RdYlBu_r", vmin=0, vmax=1)
                ax.contour(gt, levels=[0.5], colors=["lime"], linewidths=1.0)
            if r == 0:
                ax.set_title(title)
            ax.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_markdown_result_table(path: Path, summary: dict[str, float], method_names: list[str]) -> None:
    lines = [
        "# Experiment Results",
        "",
        "| Method | AP | F1 | IoU | Precision | Recall | FPR95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in method_names:
        lines.append(
            "| {name} | {ap:.4f} | {f1:.4f} | {iou:.4f} | {precision:.4f} | {recall:.4f} | {fpr95:.4f} |".format(
                name=name,
                ap=summary.get(f"{name}_ap", float("nan")),
                f1=summary.get(f"{name}_f1", float("nan")),
                iou=summary.get(f"{name}_iou", float("nan")),
                precision=summary.get(f"{name}_precision", float("nan")),
                recall=summary.get(f"{name}_recall", float("nan")),
                fpr95=summary.get(f"{name}_fpr95", float("nan")),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
