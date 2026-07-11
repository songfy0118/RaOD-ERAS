from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_research_experiment import dataset_config


DATASET_OUTPUTS = {
    "smiyc": ROOT / "outputs" / "research_experiment_smiyc",
    "road_anomaly": ROOT / "outputs" / "research_experiment_road_anomaly",
    "street_hazards": ROOT / "outputs" / "research_experiment_street_hazards_149",
}

METHOD_BY_DATASET = {
    "smiyc": "dino_eras_light",
    "road_anomaly": "dino_eras_balanced",
    "street_hazards": "dino_eras_light",
}
BASELINE = "dino"
OUT_DIR = ROOT / "paper" / "figures" / "publication_panels"


def load_rgb(path: Path, size: tuple[int, int]) -> Image.Image:
    return Image.open(path).convert("RGB").resize(size, Image.Resampling.BILINEAR)


def load_mask(path: Path, size: tuple[int, int], positive_label: int = 1) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L").resize(size, Image.Resampling.NEAREST))
    return arr == positive_label if positive_label != 255 else arr > 0


def overlay_contour(ax: plt.Axes, mask: np.ndarray, color: str = "lime", linewidth: float = 1.2) -> None:
    if mask.any():
        ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=linewidth)


def boxes_from_mask(mask: np.ndarray, max_boxes: int = 4) -> list[tuple[int, int, int, int]]:
    try:
        from scipy import ndimage

        labels, num = ndimage.label(mask)
        objects = ndimage.find_objects(labels)
        comps: list[tuple[int, int, int, int, int]] = []
        for label in range(1, num + 1):
            obj = objects[label - 1]
            if obj is None:
                continue
            ys, xs = np.nonzero(labels[obj] == label)
            if len(xs) < 20:
                continue
            y_slice, x_slice = obj
            comps.append((len(xs), x_slice.start, y_slice.start, x_slice.stop - 1, y_slice.stop - 1))
        comps.sort(reverse=True)
        return [(x0, y0, x1, y1) for _, x0, y0, x1, y1 in comps[:max_boxes]]
    except Exception:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return []
        return [(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))]


def risk_components(mask: np.ndarray, max_components: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    road_y_min = int(0.30 * h)
    min_area = max(80, int(h * w * 0.0007))
    max_area = int(h * w * 0.45)
    try:
        from scipy import ndimage

        labels, num = ndimage.label(mask)
        comps: list[tuple[float, np.ndarray]] = []
        relaxed: list[tuple[float, np.ndarray]] = []
        yy, xx = np.mgrid[0:h, 0:w]
        lane_center = 1.0 - np.minimum(np.abs(xx / max(w - 1, 1) - 0.5) / 0.5, 1.0)
        near = yy / max(h - 1, 1)
        for label in range(1, num + 1):
            comp = labels == label
            area = int(comp.sum())
            if area < min_area:
                continue
            ys = np.nonzero(comp)[0]
            if ys.mean() < road_y_min:
                continue
            risk = area * (0.55 + 0.45 * float(lane_center[comp].mean())) * (0.65 + 0.35 * float(near[comp].mean()))
            relaxed.append((risk, comp))
            if area <= max_area:
                comps.append((risk, comp))
        if not comps:
            comps = relaxed
        comps.sort(key=lambda item: item[0], reverse=True)
        return [comp for _, comp in comps[:max_components]]
    except Exception:
        return [mask]


def filtered_warning_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    for comp in risk_components(mask):
        out |= comp
    return out


def draw_warning_boxes(rgb: Image.Image, mask: np.ndarray) -> Image.Image:
    out = rgb.copy()
    draw = ImageDraw.Draw(out)
    for i, (x0, y0, x1, y1) in enumerate(boxes_from_mask(mask), start=1):
        draw.rectangle([x0, y0, x1, y1], outline=(255, 40, 40), width=4)
        draw.rectangle([x0, max(0, y0 - 24), x0 + 90, y0], fill=(255, 40, 40))
        draw.text((x0 + 5, max(0, y0 - 21)), f"risk {i}", fill=(255, 255, 255))
    return out


def make_panel(dataset_key: str, image_path: Path, gt_path: Path, sample_id: str, positive_label: int) -> Path | None:
    output_root = DATASET_OUTPUTS[dataset_key]
    method = METHOD_BY_DATASET[dataset_key]
    baseline_heatmap = output_root / "heatmaps" / BASELINE / f"{sample_id}.png"
    heatmap_path = output_root / "heatmaps" / method / f"{sample_id}.png"
    binary_path = output_root / "binary_masks" / method / f"{sample_id}.png"
    if not heatmap_path.exists() or not baseline_heatmap.exists() or not binary_path.exists():
        return None

    size = (640, 360)
    rgb = load_rgb(image_path, size)
    gt = load_mask(gt_path, size, positive_label)
    pred = np.asarray(Image.open(binary_path).convert("L").resize(size, Image.Resampling.NEAREST)) > 0
    system_mask = filtered_warning_mask(pred)
    warning = draw_warning_boxes(rgb, system_mask)

    panels = [
        ("Input", np.asarray(rgb), "rgb"),
        ("GT", gt, "mask"),
        ("DINOv2 heatmap", np.asarray(Image.open(baseline_heatmap).convert("RGB").resize(size)), "rgb"),
        (f"RaOD-ERAS ({method})", np.asarray(Image.open(heatmap_path).convert("RGB").resize(size)), "rgb_contour"),
        ("Risk-filtered mask", system_mask, "mask"),
        ("Warning output", np.asarray(warning), "rgb"),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(17.5, 3.1))
    for ax, (title, arr, kind) in zip(axes, panels):
        if kind == "mask":
            ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(arr)
        if kind in {"rgb", "rgb_contour"} and title != "Warning output":
            overlay_contour(ax, gt, "lime", 1.0)
        if kind == "rgb_contour":
            overlay_contour(ax, system_mask, "red", 0.9)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.02, wspace=0.025)
    out_path = OUT_DIR / dataset_key / f"{sample_id}_panel.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=260)
    plt.close(fig)
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, str]] = []
    for dataset_key in DATASET_OUTPUTS:
        cfg = dataset_config(dataset_key)
        cfg.image_dir = ROOT / cfg.image_dir
        cfg.gt_dir = ROOT / cfg.gt_dir
        count = 0
        for image_path in sorted(cfg.image_dir.glob(cfg.image_glob)):
            sample_id = image_path.stem
            gt_path = cfg.gt_dir / f"{sample_id}{cfg.gt_suffix}"
            if not gt_path.exists():
                continue
            panel = make_panel(dataset_key, image_path, gt_path, sample_id, cfg.positive_label)
            if panel is None:
                continue
            index_rows.append(
                {
                    "dataset": dataset_key,
                    "sample_id": sample_id,
                    "panel_path": panel.relative_to(ROOT).as_posix(),
                    "image_path": image_path.relative_to(ROOT).as_posix(),
                    "gt_path": gt_path.relative_to(ROOT).as_posix(),
                }
            )
            count += 1
        print(f"{dataset_key}: {count} panels")

    with (OUT_DIR / "figure_index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
        writer.writeheader()
        writer.writerows(index_rows)
    (OUT_DIR / "figure_index.json").write_text(json.dumps(index_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    make_contact_sheet(index_rows)


def make_contact_sheet(index_rows: list[dict[str, str]]) -> None:
    selected: list[Path] = []
    for dataset in ("smiyc", "road_anomaly", "street_hazards"):
        dataset_rows = [row for row in index_rows if row["dataset"] == dataset]
        ranked = sorted(dataset_rows, key=lambda row: gt_area(ROOT / row["gt_path"]), reverse=True)
        selected.extend(Path(ROOT / row["panel_path"]) for row in ranked[:2])
    if not selected:
        return
    rows = []
    target_w = 1800
    for path in selected[:6]:
        img = Image.open(path).convert("RGB")
        scale = target_w / img.width
        rows.append(img.resize((target_w, int(img.height * scale)), Image.Resampling.LANCZOS))
    total_h = sum(img.height for img in rows)
    sheet = Image.new("RGB", (target_w, total_h), "white")
    y = 0
    for img in rows:
        sheet.paste(img, (0, y))
        y += img.height
    sheet.save(ROOT / "paper" / "figures" / "main_qualitative_figure.png", quality=95)


def gt_area(path: Path) -> int:
    arr = np.asarray(Image.open(path).convert("L"))
    return int((arr > 0).sum())


if __name__ == "__main__":
    main()
