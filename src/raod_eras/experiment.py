from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .baselines import road_contrast_heatmap
from .config import ExperimentConfig
from .datasets import SMIYCRoadObstacleDataset, Sample
from .dino_features import DINOEncoder, multiscale_road_prototype_heatmap, road_prototype_heatmap
from .io_utils import save_binary, save_csv, save_heatmap, save_json, save_jsonl, save_mask
from .metrics import MetricConfig, PixelMetricAccumulator, evaluate_binary, evaluate_heatmap
from .object_refinement import AnomalyInstance, refine_objects
from .priors import ego_lane_prior, near_field_weight, normalize_score, trapezoid_road_prior
from .refinement import DEFAULT_ERAS_VARIANTS, connected_components, refine_heatmap
from .reporting import make_method_grid, write_markdown_result_table
from .risk_planning import plan_risk_response


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


def summarize_risk_plans(plans: list[dict[str, object]]) -> dict[str, object]:
    actions: dict[str, int] = {}
    clearances: list[float] = []
    for plan in plans:
        action = str(plan["selected_action"])
        actions[action] = actions.get(action, 0) + 1
        selected = plan["selected_trajectory"]
        if isinstance(selected, dict):
            clearances.append(float(selected["minimum_clearance"]))
    return {
        "num_plans": len(plans),
        "action_counts": actions,
        "non_braking_rate": float(sum(action != "brake_or_stop" for action in (str(p["selected_action"]) for p in plans)) / max(len(plans), 1)),
        "mean_selected_clearance": float(np.mean(clearances)) if clearances else 0.0,
        "scope": "image-plane rule validation; not a vehicle controller",
    }


def sample_source(sample_id: str) -> str:
    for source in ("road_anomaly", "smiyc", "street_hazards"):
        if sample_id.startswith(f"{source}__"):
            return source
    return "single_source"


def select_samples(samples: list[Sample], limit: int | None, strategy: str) -> list[Sample]:
    if not limit or limit >= len(samples):
        return samples
    if strategy == "head":
        return samples[:limit]
    if strategy != "stratified":
        raise ValueError(f"Unknown sample strategy: {strategy}")
    groups: dict[str, list[Sample]] = {}
    for sample in samples:
        groups.setdefault(sample_source(sample.sample_id), []).append(sample)
    selected: list[Sample] = []
    indices = {name: 0 for name in groups}
    while len(selected) < limit:
        progressed = False
        for name in sorted(groups):
            index = indices[name]
            if index < len(groups[name]) and len(selected) < limit:
                selected.append(groups[name][index])
                indices[name] += 1
                progressed = True
        if not progressed:
            break
    return selected


def warning_events_from_binary_score(
    sample_id: str,
    method_name: str,
    score: np.ndarray,
    threshold: float,
    binary_mask: np.ndarray | None = None,
    max_events: int = 20,
) -> list[dict[str, object]]:
    """Convert the final thresholded output into ranked downstream warning events."""
    h, w = score.shape
    road = trapezoid_road_prior((h, w))
    lane = ego_lane_prior((h, w))
    near = np.repeat(near_field_weight((h, w)), w, axis=1)
    _, comps = connected_components(score >= threshold if binary_mask is None else binary_mask)
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
    samples = select_samples(samples, config.max_samples, config.sample_strategy)

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
    gt_masks: list[np.ndarray] = []
    heatmap_paths: dict[str, list[Path]] = {}
    warning_events: list[dict[str, object]] = []
    risk_plans: list[dict[str, object]] = []
    aggregate: dict[str, PixelMetricAccumulator] = {}
    source_aggregate: dict[str, dict[str, PixelMetricAccumulator]] = {}

    for sample in samples:
        loaded = dataset.load(sample)
        image_paths.append(sample.image_path)
        gt_masks.append(loaded.gt)
        outputs: dict[str, np.ndarray] = {}
        object_masks: dict[str, np.ndarray] = {}
        object_instances: dict[str, list[AnomalyInstance]] = {}
        timings: dict[str, float] = {}

        if config.method.use_roadcontrast:
            t0 = time.perf_counter()
            outputs["roadcontrast"] = road_contrast_heatmap(loaded.rgb)
            timings["roadcontrast_seconds"] = time.perf_counter() - t0

        if config.method.use_dino and encoder is not None:
            t0 = time.perf_counter()
            outputs["dino"] = road_prototype_heatmap(encoder, loaded.image, loaded.gt.shape)
            timings["dino_seconds"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            outputs["dino_multiscale"] = multiscale_road_prototype_heatmap(
                encoder,
                loaded.image,
                loaded.gt.shape,
            )
            timings["dino_multiscale_seconds"] = time.perf_counter() - t0

            h, w = loaded.gt.shape
            road = trapezoid_road_prior((h, w))
            lane = ego_lane_prior((h, w))
            near = np.repeat(near_field_weight((h, w)), w, axis=1)
            local = outputs.get("roadcontrast", np.zeros_like(outputs["dino_multiscale"]))
            risk_prior = 0.62 + 0.20 * road + 0.18 * lane * near
            outputs["dino_risk_heatmap"] = normalize_score(
                (0.78 * outputs["dino_multiscale"] + 0.22 * local) * risk_prior
            )

        if config.method.use_eras:
            for base_name, base_score in list(outputs.items()):
                if base_name not in {"roadcontrast", "dino", "dino_risk_heatmap"}:
                    continue
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

        for source_name in (
            "roadcontrast_eras_balanced",
            "dino_eras_balanced",
            "dino_risk_heatmap_eras_balanced",
        ):
            if source_name not in outputs:
                continue
            method_name = source_name.replace("eras_balanced", "closed_loop")
            t0 = time.perf_counter()
            closed_heatmap, closed_mask, instances = refine_objects(outputs[source_name])
            outputs[method_name] = closed_heatmap
            object_masks[method_name] = closed_mask
            object_instances[method_name] = instances
            timings[f"{method_name}_seconds"] = time.perf_counter() - t0

        row: dict[str, object] = {"id": sample.sample_id}
        row.update(timings)
        for method_name, score in outputs.items():
            method_names.add(method_name)
            output_threshold = config.method.output_threshold
            binary_mask = score >= output_threshold
            metrics = evaluate_heatmap(score, loaded.gt, loaded.valid, metric_config)
            metrics.update(evaluate_binary(binary_mask, loaded.gt, loaded.valid))
            if method_name in object_masks:
                object_metrics = evaluate_binary(object_masks[method_name], loaded.gt, loaded.valid)
                metrics.update({f"object_{key}": value for key, value in object_metrics.items()})
            for metric_name, value in metrics.items():
                row[f"{method_name}_{metric_name}"] = value
            aggregate.setdefault(method_name, PixelMetricAccumulator(output_threshold)).update(
                score, loaded.gt, loaded.valid
            )
            source = sample_source(sample.sample_id)
            source_aggregate.setdefault(source, {}).setdefault(
                method_name, PixelMetricAccumulator(output_threshold)
            ).update(score, loaded.gt, loaded.valid)
            heatmap_path = config.output.output_dir / config.output.heatmap_dir / method_name / f"{sample.sample_id}.png"
            binary_path = config.output.output_dir / config.output.binary_dir / method_name / f"{sample.sample_id}.png"
            save_heatmap(heatmap_path, score)
            save_binary(binary_path, score, output_threshold)
            if method_name in object_masks:
                object_path = config.output.output_dir / config.output.object_mask_dir / method_name / f"{sample.sample_id}.png"
                save_mask(object_path, object_masks[method_name])
            heatmap_paths.setdefault(method_name, []).append(heatmap_path)
            warning_events.extend(
                warning_events_from_binary_score(
                    sample.sample_id,
                    method_name,
                    score,
                    output_threshold,
                    binary_mask=binary_mask,
                )
            )
            if method_name in object_instances:
                plan = plan_risk_response(score, object_instances[method_name])
                plan.update(
                    {
                        "sample_id": sample.sample_id,
                        "method": method_name,
                        "instances": [instance.to_dict() for instance in object_instances[method_name]],
                    }
                )
                risk_plans.append(plan)
        rows.append(row)

    summary = summarize(rows)
    method_list = sorted(method_names)
    save_csv(config.output.output_dir / "comparison_table.csv", rows)
    save_json(config.output.output_dir / "metrics.json", summary)
    save_json(
        config.output.output_dir / "aggregate_metrics.json",
        {name: accumulator.compute() for name, accumulator in sorted(aggregate.items())},
    )
    save_json(
        config.output.output_dir / "dataset_breakdown.json",
        {
            source: {name: accumulator.compute() for name, accumulator in sorted(methods.items())}
            for source, methods in sorted(source_aggregate.items())
        },
    )
    save_jsonl(config.output.output_dir / "warning_events.jsonl", warning_events)
    save_jsonl(config.output.output_dir / "risk_plans.jsonl", risk_plans)
    save_json(config.output.output_dir / "risk_summary.json", summarize_risk_plans(risk_plans))
    save_json(
        config.output.output_dir / "ablation_manifest.json",
        {
            "ordered_chain": [
                "dino",
                "dino_multiscale",
                "dino_risk_heatmap",
                "dino_risk_heatmap_eras_balanced",
                "dino_risk_heatmap_closed_loop",
            ],
            "notes": {
                "dino": "single-scale road-prototype baseline",
                "dino_multiscale": "multi-scale road-prototype heatmap",
                "dino_risk_heatmap": "multi-scale heatmap plus local contrast and road/lane risk prior",
                "dino_risk_heatmap_eras_balanced": "risk heatmap plus ERAS spatial refinement",
                "dino_risk_heatmap_closed_loop": "object verification and mask-to-heatmap feedback",
            },
        },
    )
    write_markdown_result_table(config.output.output_dir / config.output.report_dir / "result_table.md", summary, method_list)
    selected_methods = [
        name
        for name in [
            "roadcontrast",
            "roadcontrast_closed_loop",
            "dino",
            "dino_multiscale",
            "dino_risk_heatmap",
            "dino_risk_heatmap_closed_loop",
        ]
        if name in heatmap_paths
    ]
    if selected_methods:
        make_method_grid(
            image_paths,
            gt_masks,
            {name: heatmap_paths[name] for name in selected_methods},
            config.output.output_dir / config.output.report_dir / "method_grid.png",
        )
    return summary
