from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"


def make_framework_pipeline() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 3.8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")

    nodes = [
        ("RGB\nroad image", 0.8, "#E8F1FA"),
        ("DINOv2\npatch features", 3.0, "#F4E8FA"),
        ("Road prototype\nnormal feature", 5.2, "#EAF7EA"),
        ("Anomaly\nheatmap", 7.4, "#FFF1D6"),
        ("ERAS\nrisk refinement", 9.6, "#FFE3E3"),
        ("Binary mask\n+ warning events", 12.0, "#E9F6F8"),
    ]
    for text, x, color in nodes:
        box = patches.FancyBboxPatch(
            (x - 0.75, 1.35),
            1.5,
            1.25,
            boxstyle="round,pad=0.03,rounding_size=0.05",
            linewidth=1.5,
            edgecolor="#333333",
            facecolor=color,
        )
        ax.add_patch(box)
        ax.text(x, 1.98, text, ha="center", va="center", fontsize=10)

    for i in range(len(nodes) - 1):
        x0 = nodes[i][1] + 0.8
        x1 = nodes[i + 1][1] - 0.8
        ax.annotate(
            "",
            xy=(x1, 1.98),
            xytext=(x0, 1.98),
            arrowprops=dict(arrowstyle="->", lw=1.8, color="#333333"),
        )

    ax.text(5.2, 3.15, "Road-prior region selects normal road features", ha="center", fontsize=10, color="#2B6A2B")
    ax.annotate("", xy=(5.2, 2.65), xytext=(5.2, 3.0), arrowprops=dict(arrowstyle="->", lw=1.3, color="#2B6A2B"))
    ax.text(9.6, 0.75, "road prior + ego lane + near field + connected components", ha="center", fontsize=10, color="#8A2A2A")
    ax.annotate("", xy=(9.6, 1.32), xytext=(9.6, 0.95), arrowprops=dict(arrowstyle="->", lw=1.3, color="#8A2A2A"))

    ax.text(
        7,
        3.65,
        "RaOD-ERAS: training-free road anomaly heatmap and risk-aware warning output",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )
    path = OUT / "framework_pipeline.png"
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def load_first_event(method: str = "dino_eras_balanced") -> dict[str, object]:
    path = ROOT / "outputs" / "research_experiment_road_anomaly" / "warning_events.jsonl"
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item.get("method") == method:
                return item
    with path.open("r", encoding="utf-8") as f:
        return json.loads(next(f))


def connected_components(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[tuple[np.ndarray, np.ndarray]] = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            ys: list[int] = []
            xs: list[int] = []
            while stack:
                cy, cx = stack.pop()
                ys.append(cy)
                xs.append(cx)
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append((np.asarray(ys), np.asarray(xs)))
    return comps


def best_warning_from_binary(sample_id: str, method: str) -> dict[str, object]:
    mask_path = ROOT / "outputs" / "research_experiment_road_anomaly" / "binary_masks" / method / f"{sample_id}.png"
    heatmap_path = ROOT / "outputs" / "research_experiment_road_anomaly" / "heatmaps" / method / f"{sample_id}.png"
    gt_path = ROOT / "data" / "road_anomaly" / "paper_subset" / "gt" / f"{sample_id}_labels_semantic.png"
    mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    heatmap = np.asarray(Image.open(heatmap_path).convert("L")) / 255.0
    gt = np.asarray(Image.open(gt_path).convert("L")) == 1
    best: dict[str, object] | None = None
    for rank, (ys, xs) in enumerate(connected_components(mask), start=1):
        area = len(xs)
        if area < 40:
            continue
        inter = int(gt[ys, xs].sum())
        overlap = inter / max(area, 1)
        mean_score = float(heatmap[ys, xs].mean())
        risk_score = float(mean_score * (0.35 + 0.65 * overlap))
        candidate = {
            "event_id": f"{sample_id}_{method}_best",
            "sample_id": sample_id,
            "method": method,
            "rank": rank,
            "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "area": int(area),
            "risk_score": risk_score,
            "gt_overlap_for_figure": float(overlap),
            "system_action": "warn_or_slow_down",
        }
        if best is None or float(candidate["gt_overlap_for_figure"]) > float(best["gt_overlap_for_figure"]):
            best = candidate
    if best is None:
        return load_first_event(method)
    return best


def make_warning_event_example() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    sample_id = "validation0000"
    method = "dino_eras_balanced"
    event = best_warning_from_binary(sample_id, method)
    image_path = ROOT / "data" / "road_anomaly" / "paper_subset" / "images" / f"{sample_id}.jpg"
    heatmap_path = ROOT / "outputs" / "research_experiment_road_anomaly" / "heatmaps" / method / f"{sample_id}.png"
    mask_path = ROOT / "outputs" / "research_experiment_road_anomaly" / "binary_masks" / method / f"{sample_id}.png"

    image = np.asarray(Image.open(image_path).convert("RGB"))
    heatmap = np.asarray(Image.open(heatmap_path).convert("L")) / 255.0
    mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    gt_path = ROOT / "data" / "road_anomaly" / "paper_subset" / "gt" / f"{sample_id}_labels_semantic.png"
    gt = np.asarray(Image.open(gt_path).convert("L")) == 1
    bbox = [int(v) for v in event["bbox_xyxy"]]
    x0, y0, x1, y1 = bbox

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    axes[0].imshow(image)
    axes[0].contour(gt, levels=[0.5], colors=["lime"], linewidths=1.1)
    axes[0].add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", linewidth=2.2))
    axes[0].text(
        x0,
        max(10, y0 - 8),
        f"{event['event_id']}\nrisk={float(event['risk_score']):.3f}",
        color="white",
        fontsize=8,
        bbox=dict(facecolor="red", alpha=0.75, edgecolor="none"),
    )
    axes[0].set_title("RGB + warning bbox")

    axes[1].imshow(image)
    axes[1].imshow(heatmap, cmap="RdYlBu_r", alpha=0.62, vmin=0, vmax=1)
    axes[1].contour(gt, levels=[0.5], colors=["lime"], linewidths=1.1)
    axes[1].add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="lime", linewidth=2.0))
    axes[1].set_title("Risk heatmap")

    axes[2].imshow(mask, cmap="gray")
    axes[2].contour(gt, levels=[0.5], colors=["lime"], linewidths=1.1)
    axes[2].add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", linewidth=2.0))
    axes[2].set_title("Binary mask + event")

    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    path = OUT / "warning_event_example.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def main() -> None:
    paths = [make_framework_pipeline(), make_warning_event_example()]
    for path in paths:
        print(path)


if __name__ == "__main__":
    sys.exit(main())
