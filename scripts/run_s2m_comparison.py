from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_research_experiment import dataset_config
from src.raod_eras.baselines import road_contrast_heatmap
from src.raod_eras.datasets import SMIYCRoadObstacleDataset
from src.raod_eras.dino_features import DINOEncoder, multiscale_road_prototype_heatmap, road_prototype_heatmap
from src.raod_eras.experiment import remove_small_components, sample_source, select_samples
from src.raod_eras.io_utils import ensure_dir, save_heatmap, save_json, save_mask
from src.raod_eras.metrics import PixelMetricAccumulator, evaluate_binary
from src.raod_eras.priors import ego_lane_prior, near_field_weight, normalize_score, trapezoid_road_prior
from src.raod_eras.score_to_mask import PromptConfig, sam_point_prompts, sam_risk_box_prompts, sam_score_to_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled comparison of score-to-SAM prompting strategies.")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs/s2m_comparison"))
    parser.add_argument("--ugains-threshold", type=float, default=0.70)
    parser.add_argument("--riskprompt-threshold", type=float, default=0.70)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--sam-checkpoint", type=Path, default=Path("external/S2M_official/tools/sam_vit_b_01ec64.pth"))
    args = parser.parse_args()

    import torch
    from segment_anything import SamPredictor, sam_model_registry

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint=str(ROOT / args.sam_checkpoint)).to(device)
    predictor = SamPredictor(sam)
    encoder = DINOEncoder("dinov2_vits14", 518)
    dataset = SMIYCRoadObstacleDataset(dataset_config("unified"))
    samples = select_samples(dataset.list_samples(), args.max_samples, "stratified", args.sample_offset)
    methods = ("threshold", "s2m_style", "ugains_style", "riskprompt")
    aggregate = {name: PixelMetricAccumulator(0.70) for name in methods}
    sweep_values = np.linspace(0.50, 0.90, 17)
    sweeps = {
        name: {float(value): PixelMetricAccumulator(float(value)) for value in sweep_values}
        for name in ("ugains_style", "riskprompt")
    }
    by_source: dict[str, dict[str, PixelMetricAccumulator]] = {}
    per_image: list[dict[str, object]] = []
    output_dir = ROOT / args.out
    prompt_config = PromptConfig()

    for sample in samples:
        started = time.perf_counter()
        loaded = dataset.load(sample)
        cache_path = output_dir / "cache" / f"{sample.sample_id}.npz"
        if cache_path.exists() and not args.no_cache:
            cached = np.load(cache_path)
            score = cached["score"]
            ugains_score = cached["ugains_score"]
            riskprompt_score = cached["riskprompt_score"]
            threshold_mask = cached["threshold_mask"].astype(bool)
            s2m_mask = cached["s2m_mask"].astype(bool)
            riskprompt_mask = cached["riskprompt_mask"].astype(bool)
            ugains_points = int(cached["ugains_points"])
            riskprompt_boxes = int(cached["riskprompt_boxes"])
        else:
            dino = road_prototype_heatmap(encoder, loaded.image, loaded.gt.shape)
            multiscale = multiscale_road_prototype_heatmap(encoder, loaded.image, loaded.gt.shape)
            local = road_contrast_heatmap(loaded.rgb)
            h, w = loaded.gt.shape
            road = trapezoid_road_prior((h, w))
            lane = ego_lane_prior((h, w))
            near = np.repeat(near_field_weight((h, w)), w, axis=1)
            candidate = normalize_score((0.78 * multiscale + 0.22 * local) * (0.62 + 0.20 * road + 0.18 * lane * near))
            score = normalize_score(0.80 * dino + 0.20 * candidate)
            rgb = np.asarray(loaded.image.resize((w, h), Image.Resampling.BILINEAR))
            predictor.set_image(rgb)
            threshold_mask = remove_small_components(score >= 0.70, 0.00005)
            s2m_mask, _ = sam_score_to_mask(
                predictor,
                rgb,
                score,
                prompt_config,
                risk_aware=False,
                image_already_set=True,
            )
            ugains = sam_point_prompts(predictor, score, prompt_config, road_aware=False)
            riskprompt = sam_risk_box_prompts(predictor, score, prompt_config)
            ugains_score = ugains.refined_score
            riskprompt_score = riskprompt.refined_score
            riskprompt_mask = riskprompt.mask
            ugains_points = len(ugains.points)
            riskprompt_boxes = len(riskprompt.boxes)
            ensure_dir(cache_path.parent)
            np.savez_compressed(
                cache_path,
                score=score.astype(np.float16),
                ugains_score=ugains_score.astype(np.float16),
                riskprompt_score=riskprompt_score.astype(np.float16),
                threshold_mask=threshold_mask,
                s2m_mask=s2m_mask,
                riskprompt_mask=riskprompt_mask,
                ugains_points=np.asarray(ugains_points),
                riskprompt_boxes=np.asarray(riskprompt_boxes),
            )
        masks = {
            "threshold": threshold_mask,
            "s2m_style": s2m_mask,
            "ugains_style": remove_small_components(ugains_score >= args.ugains_threshold, 0.00005),
            "riskprompt": riskprompt_mask,
        }
        heatmaps = {
            "threshold": score,
            "s2m_style": score,
            "ugains_style": ugains_score,
            "riskprompt": riskprompt_score,
        }
        row: dict[str, object] = {"id": sample.sample_id, "seconds": time.perf_counter() - started}
        source = sample_source(sample.sample_id)
        by_source.setdefault(source, {name: PixelMetricAccumulator(0.70) for name in methods})
        for name, mask in masks.items():
            aggregate[name].update(heatmaps[name], loaded.gt, loaded.valid, binary_pred=mask)
            by_source[source][name].update(heatmaps[name], loaded.gt, loaded.valid, binary_pred=mask)
            row.update({f"{name}_{key}": value for key, value in evaluate_binary(mask, loaded.gt, loaded.valid).items()})
            save_mask(output_dir / "masks" / name / f"{sample.sample_id}.png", mask)
            if name in ("threshold", "ugains_style", "riskprompt"):
                save_heatmap(output_dir / "heatmaps" / name / f"{sample.sample_id}.png", heatmaps[name])
        for name in sweeps:
            for threshold, accumulator in sweeps[name].items():
                candidate_mask = remove_small_components(heatmaps[name] >= threshold, 0.00005)
                accumulator.update(heatmaps[name], loaded.gt, loaded.valid, binary_pred=candidate_mask)
        row.update(
            {
                "ugains_points": ugains_points,
                "riskprompt_points": 0,
                "riskprompt_boxes": riskprompt_boxes,
            }
        )
        per_image.append(row)

    result = {
        "device": device,
        "num_samples": len(samples),
        "aggregate": {name: metric.compute() for name, metric in aggregate.items()},
        "by_source": {source: {name: metric.compute() for name, metric in values.items()} for source, values in by_source.items()},
        "threshold_sweep": {
            name: {f"{threshold:.3f}": accumulator.compute() for threshold, accumulator in values.items()}
            for name, values in sweeps.items()
        },
        "fixed_thresholds": {
            "threshold": 0.70,
            "ugains_style": args.ugains_threshold,
            "riskprompt": "mask-level boundary-consistency selection",
            "riskprompt_heatmap_sweep_reference": args.riskprompt_threshold,
        },
        "per_image": per_image,
        "protocol": (
            "Controlled prompt comparison with one shared DINOv2 heatmap and one shared SAM-B. "
            "S2M-style and UGainS-style reproduce prompt mechanisms, not official end-to-end checkpoints."
        ),
    }
    save_json(output_dir / "results.json", result)
    console_summary = {
        "device": device,
        "num_samples": len(samples),
        "aggregate": result["aggregate"],
        "by_source": result["by_source"],
        "results": str(output_dir / "results.json"),
    }
    print(json.dumps(console_summary, indent=2))


if __name__ == "__main__":
    main()
