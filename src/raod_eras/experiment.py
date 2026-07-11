from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .baselines import road_contrast_heatmap
from .config import ExperimentConfig
from .datasets import SMIYCRoadObstacleDataset
from .dino_features import DINOEncoder, road_prototype_heatmap
from .io_utils import save_binary, save_csv, save_heatmap, save_json, save_jsonl
from .metrics import MetricConfig, evaluate_heatmap
from .priors import ego_lane_prior, near_field_weight, normalize_score, trapezoid_road_prior
from .refinement import DEFAULT_ERAS_VARIANTS, connected_components, refine_heatmap
from .reporting import make_method_grid, write_markdown_result_table


def summarize(rows: list[dict[str, object]]) -> dict[str, float]:
    summary: dict[str, float] = {"num_samples": float(len(rows))}
    keys = [key for key in rows[0].keys() if key not in {"id"} and not key.endswith("_seconds")]
    for key in keys:
        values = [float(row[key]) for row in rows if key in row and not np.isnan(float(row[key]))]
        summary[key] = float(np.mean(values)) if values else float("nan")
    for key in [key for key in rows[0].keys() if key.endswith("_seconds")]:
        values = [float(row[key]) for row in rows if key in row]
        summary[key] = float(np.mean(values)) if values else float("nan")
    return summary


def warning_events_from_binary_score(
    sample_id: str,
    method_name: str,
    score: np.ndarray,
    threshold: float,
    max_events: int = 20,
) -> list[dict[str, object]]:
    """Convert the final thresholded output into ranked downstream warning events."""
    h, w = score.shape
    road = trapezoid_road_prior((h, w))
    lane = ego_lane_prior((h, w))
    near = np.repeat(near_field_weight((h, w)), w, axis=1)
    _, comps = connected_components(score >= threshold)
    image_area = h * w
    min_area = max(24, int(image_area * 0.00008))
    max_area = int(image_area * 0.20)
    target_area = max(1.0, image_area * 0.008)
    events: list[dict[str, object]] = []

    for label, (ys, xs) in comps.items():
        area = int(len(xs))
        if area < min_area or area > max_area:
            continue
        mean_score = float(score[ys, xs].mean())
        road_overlap = float(road[ys, xs].mean())
        lane_overlap = float(lane[ys, xs].mean())
        near_weight = float(near[ys, xs].mean())
        size_prior = float(np.exp(-abs(np.log(area + 1.0) - np.log(target_area + 1.0)) / 1.35))
        risk_score = mean_score * (0.6 + 0.4 * road_overlap) * (0.65 + 0.35 * lane_overlap) * (0.75 + 0.25 * near_weight) * (0.4 + 0.6 * size_prior)
        events.append(
            {
                "event_id": f"{sample_id}_{method_name}_{int(label):03d}",
                "sample_id": sample_id,
                "method": method_name,
                "component_label": int(label),
                "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "area": area,
                "risk_score": float(risk_score),
                "mean_score": mean_score,
                "threshold": float(threshold),
                "road_overlap": road_overlap,
                "lane_overlap": lane_overlap,
                "near_weight": near_weight,
                "system_action": "warn_or_slow_down",
            }
        )

    events.sort(key=lambda item: float(item["risk_score"]), reverse=True)
    for rank, item in enumerate(events[:max_events], start=1):
        item["rank"] = rank
    return events[:max_events]


def run_experiment(config: ExperimentConfig) -> dict[str, float]:
    config.resolve()
    dataset = SMIYCRoadObstacleDataset(config.dataset)
    samples = dataset.list_samples()
    if config.max_samples:
        samples = samples[: config.max_samples]

    metric_config = MetricConfig(
        threshold_min=config.method.threshold_min,
        threshold_max=config.method.threshold_max,
        threshold_steps=config.method.threshold_steps,
    )
    encoder = None
    if config.method.use_dino:
        encoder = DINOEncoder(config.method.dino_model, config.method.dino_input_size)

    rows: list[dict[str, object]] = []
    method_names: set[str] = set()
    image_paths: list[Path] = []
    gt_paths: list[Path] = []
    heatmap_paths: dict[str, list[Path]] = {}
    warning_events: list[dict[str, object]] = []

    for sample in samples:
        loaded = dataset.load(sample)
        image_paths.append(sample.image_path)
        gt_paths.append(sample.gt_path)
        outputs: dict[str, np.ndarray] = {}
        timings: dict[str, float] = {}

        if config.method.use_roadcontrast:
            t0 = time.perf_counter()
            outputs["roadcontrast"] = road_contrast_heatmap(loaded.rgb)
            timings["roadcontrast_seconds"] = time.perf_counter() - t0

        if config.method.use_dino and encoder is not None:
            t0 = time.perf_counter()
            outputs["dino"] = road_prototype_heatmap(encoder, loaded.image, loaded.gt.shape)
            timings["dino_seconds"] = time.perf_counter() - t0

        if config.method.use_eras:
            for base_name, base_score in list(outputs.items()):
                refined_outputs: dict[str, np.ndarray] = {}
                for variant in DEFAULT_ERAS_VARIANTS:
                    method_name = f"{base_name}_{variant.name}"
                    t0 = time.perf_counter()
                    refined_outputs[variant.name], _ = refine_heatmap(base_score, variant)
                    outputs[method_name] = refined_outputs[variant.name]
                    timings[f"{method_name}_seconds"] = time.perf_counter() - t0
                if base_name == "dino" and "eras_light" in refined_outputs:
                    t0 = time.perf_counter()
                    outputs["dino_eras_guarded"] = normalize_score(0.78 * base_score + 0.22 * refined_outputs["eras_light"])
                    timings["dino_eras_guarded_seconds"] = time.perf_counter() - t0

        row: dict[str, object] = {"id": sample.sample_id}
        row.update(timings)
        for method_name, score in outputs.items():
            method_names.add(method_name)
            metrics = evaluate_heatmap(score, loaded.gt, loaded.valid, metric_config)
            for metric_name, value in metrics.items():
                row[f"{method_name}_{metric_name}"] = value
            heatmap_path = config.output.output_dir / config.output.heatmap_dir / method_name / f"{sample.sample_id}.png"
            binary_path = config.output.output_dir / config.output.binary_dir / method_name / f"{sample.sample_id}.png"
            save_heatmap(heatmap_path, score)
            save_binary(binary_path, score, metrics["threshold"])
            heatmap_paths.setdefault(method_name, []).append(heatmap_path)
            warning_events.extend(warning_events_from_binary_score(sample.sample_id, method_name, score, metrics["threshold"]))
        rows.append(row)

    summary = summarize(rows)
    method_list = sorted(method_names)
    save_csv(config.output.output_dir / "comparison_table.csv", rows)
    save_json(config.output.output_dir / "metrics.json", summary)
    save_jsonl(config.output.output_dir / "warning_events.jsonl", warning_events)
    write_markdown_result_table(config.output.output_dir / config.output.report_dir / "result_table.md", summary, method_list)
    selected_methods = [name for name in ["roadcontrast", "roadcontrast_eras_balanced", "dino", "dino_eras_light", "dino_eras_balanced"] if name in heatmap_paths]
    if selected_methods:
        make_method_grid(
            image_paths,
            gt_paths,
            {name: heatmap_paths[name] for name in selected_methods},
            config.output.output_dir / config.output.report_dir / "method_grid.png",
        )
    return summary
