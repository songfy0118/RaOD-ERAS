from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def make_method_grid(
    image_paths: list[Path],
    gt_masks: list[np.ndarray],
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
        gt = np.asarray(
            Image.fromarray(gt_masks[r].astype(np.uint8)).resize((512, 288), Image.Resampling.NEAREST),
            dtype=bool,
        )
        panels: list[tuple[str, np.ndarray, str]] = [("RGB + GT", rgb, "rgb"), ("GT", gt, "mask")]
        for name in names:
            arr = np.asarray(Image.open(heatmap_paths[name][r]).convert("RGB").resize((512, 288), Image.Resampling.BILINEAR))
            panels.append((name, arr, "heat_rgb"))

        for c, (title, arr, kind) in enumerate(panels):
            ax = axes[r, c]
            if kind == "rgb":
                ax.imshow(arr)
                ax.contour(gt, levels=[0.5], colors=["lime"], linewidths=1.0)
            elif kind == "mask":
                ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(arr)
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
        "| Method | AP | Oracle F1 | Fixed F1 | Fixed IoU | Component F1 | Boundary F1 | FPR95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in method_names:
        lines.append(
            "| {name} | {ap:.4f} | {f1:.4f} | {fixed_f1:.4f} | {fixed_iou:.4f} | {component_f1:.4f} | {boundary_f1:.4f} | {fpr95:.4f} |".format(
                name=name,
                ap=summary.get(f"{name}_ap", float("nan")),
                f1=summary.get(f"{name}_f1", float("nan")),
                fixed_f1=summary.get(f"{name}_fixed_f1", float("nan")),
                fixed_iou=summary.get(f"{name}_fixed_iou", float("nan")),
                component_f1=summary.get(f"{name}_component_f1", float("nan")),
                boundary_f1=summary.get(f"{name}_boundary_f1", float("nan")),
                fpr95=summary.get(f"{name}_fpr95", float("nan")),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
