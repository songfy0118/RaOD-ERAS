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
from src.raod_eras.datasets import SMIYCRoadObstacleDataset
from src.raod_eras.experiment import sample_source, select_samples
from src.raod_eras.io_utils import ensure_dir, save_heatmap, save_json, save_mask
from src.raod_eras.metrics import PixelMetricAccumulator, average_precision, evaluate_binary, fpr_at_95_tpr
from src.raod_eras.score_to_mask import PromptConfig, sam_box_ablation


METHODS = (
    "A_basic_box",
    "B_road_box",
    "C_boundary_select",
    "D_full_feedback",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential RiskPrompt-SAM module ablation.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs/riskprompt_ablation_full_189"))
    parser.add_argument(
        "--source-cache",
        type=Path,
        default=Path("outputs/riskprompt_full_189/cache"),
        help="Frozen cache from the completed shared-front-end experiment.",
    )
    parser.add_argument(
        "--ablation-cache",
        type=Path,
        default=Path("outputs/riskprompt_ablation_cache"),
        help="Shared cache for the newly required B-only SAM masks.",
    )
    parser.add_argument("--save-visuals", action="store_true")
    parser.add_argument(
        "--per-image-heatmap-statistics",
        action="store_true",
        help="Compute expensive per-image AP/FPR95 for paired heatmap bootstrap.",
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--sam-checkpoint", type=Path, default=Path("external/S2M_official/tools/sam_vit_b_01ec64.pth"))
    args = parser.parse_args()

    import torch
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ModuleNotFoundError:
        bundled_sam = ROOT / "external" / "UGainS_official" / "ugains" / "models"
        if not bundled_sam.exists():
            raise
        sys.path.insert(0, str(bundled_sam))
        from segment_anything import SamPredictor, sam_model_registry

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint=str(ROOT / args.sam_checkpoint)).to(device)
    if not hasattr(sam, "device"):
        sam.device = torch.device(device)
    predictor = SamPredictor(sam)
    dataset = SMIYCRoadObstacleDataset(dataset_config("unified"))
    samples = select_samples(dataset.list_samples(), args.max_samples, "stratified", args.sample_offset)
    if not samples:
        raise RuntimeError("No samples selected.")

    output_dir = ROOT / args.out
    source_cache = ROOT / args.source_cache
    ablation_cache = ROOT / args.ablation_cache
    config = PromptConfig()
    aggregate = {name: PixelMetricAccumulator(0.70) for name in METHODS}
    by_source: dict[str, dict[str, PixelMetricAccumulator]] = {}
    per_image: list[dict[str, object]] = []

    for index, sample in enumerate(samples, start=1):
        started = time.perf_counter()
        loaded = dataset.load(sample)
        frozen_path = source_cache / f"{sample.sample_id}.npz"
        if not frozen_path.exists():
            raise FileNotFoundError(f"Missing frozen source cache: {frozen_path}")
        frozen = np.load(frozen_path)
        score = frozen["score"].astype(np.float32)
        basic_mask = frozen["s2m_mask"].astype(bool)
        boundary_mask = frozen["riskprompt_mask"].astype(bool)
        feedback_score = frozen["riskprompt_score"].astype(np.float32)

        b_cache_path = ablation_cache / f"{sample.sample_id}.npz"
        if b_cache_path.exists() and not args.no_cache:
            b_cached = np.load(b_cache_path)
            required = {"road_mask", "boundary_mask", "feedback_score", "road_boxes", "cache_version"}
            cache_valid = required.issubset(b_cached.files) and int(b_cached["cache_version"]) == 2
        else:
            cache_valid = False
        if cache_valid:
            road_mask = b_cached["road_mask"].astype(bool)
            boundary_mask = b_cached["boundary_mask"].astype(bool)
            feedback_score = b_cached["feedback_score"].astype(np.float32)
            road_boxes = int(b_cached["road_boxes"])
            inference_seconds = 0.0
        else:
            h, w = loaded.gt.shape
            rgb = np.asarray(loaded.image.resize((w, h), Image.Resampling.BILINEAR))
            predictor.set_image(rgb)
            inference_started = time.perf_counter()
            road_result = sam_box_ablation(
                predictor,
                score,
                config,
                road_aware=True,
                boundary_aware=False,
                feedback=False,
            )
            boundary_result = sam_box_ablation(
                predictor,
                score,
                config,
                road_aware=True,
                boundary_aware=True,
                feedback=True,
            )
            inference_seconds = time.perf_counter() - inference_started
            road_mask = road_result.mask
            boundary_mask = boundary_result.mask
            feedback_score = boundary_result.refined_score
            road_boxes = len(road_result.boxes)
            ensure_dir(b_cache_path.parent)
            np.savez_compressed(
                b_cache_path,
                road_mask=road_mask,
                boundary_mask=boundary_mask,
                feedback_score=feedback_score.astype(np.float16),
                road_boxes=np.asarray(road_boxes),
                inference_seconds=np.asarray(inference_seconds),
                cache_version=np.asarray(2),
            )

        masks = {
            "A_basic_box": basic_mask,
            "B_road_box": road_mask,
            "C_boundary_select": boundary_mask,
            "D_full_feedback": boundary_mask,
        }
        heatmaps = {
            "A_basic_box": score,
            "B_road_box": score,
            "C_boundary_select": score,
            "D_full_feedback": feedback_score,
        }
        source = sample_source(sample.sample_id)
        by_source.setdefault(source, {name: PixelMetricAccumulator(0.70) for name in METHODS})
        row: dict[str, object] = {
            "id": sample.sample_id,
            "source": source,
            "seconds": time.perf_counter() - started,
            "new_sam_inference_seconds": inference_seconds,
            "road_boxes": road_boxes,
        }
        if args.per_image_heatmap_statistics:
            row.update(
                {
                    "base_ap": average_precision(score, loaded.gt, loaded.valid),
                    "base_fpr95": fpr_at_95_tpr(score, loaded.gt, loaded.valid),
                    "feedback_ap": average_precision(feedback_score, loaded.gt, loaded.valid),
                    "feedback_fpr95": fpr_at_95_tpr(feedback_score, loaded.gt, loaded.valid),
                }
            )

        binary_metrics: dict[str, dict[str, float]] = {}
        for name in METHODS:
            aggregate[name].update(heatmaps[name], loaded.gt, loaded.valid, binary_pred=masks[name])
            by_source[source][name].update(heatmaps[name], loaded.gt, loaded.valid, binary_pred=masks[name])
            if name == "D_full_feedback":
                binary_metrics[name] = binary_metrics["C_boundary_select"]
            else:
                binary_metrics[name] = evaluate_binary(masks[name], loaded.gt, loaded.valid)
            row.update({f"{name}_{key}": value for key, value in binary_metrics[name].items()})
            if args.save_visuals:
                save_mask(output_dir / "masks" / name / f"{sample.sample_id}.png", masks[name])
        if args.save_visuals:
            save_heatmap(output_dir / "heatmaps" / "base" / f"{sample.sample_id}.png", score)
            save_heatmap(output_dir / "heatmaps" / "feedback" / f"{sample.sample_id}.png", feedback_score)
        per_image.append(row)
        print(f"[{index:03d}/{len(samples):03d}] {sample.sample_id} B-boxes={road_boxes} new-SAM={inference_seconds:.2f}s")

    result = {
        "device": device,
        "num_samples": len(samples),
        "methods": list(METHODS),
        "aggregate": {name: value.compute() for name, value in aggregate.items()},
        "by_source": {
            source: {name: value.compute() for name, value in methods.items()}
            for source, methods in by_source.items()
        },
        "per_image": per_image,
        "protocol": {
            "dataset": "189-pair partial-public unified evaluation manifest",
            "shared_front_end": "frozen cached DINOv2 score map",
            "shared_segmenter": "SAM ViT-B",
            "gt_use": "metrics only; never read by prompt or inference functions",
            "A_basic_box": "ordinary score boxes + highest SAM confidence; no feedback",
            "B_road_box": "road-aware adaptive boxes + highest SAM confidence; no feedback",
            "C_boundary_select": "B + inside-versus-ring boundary selection; no feedback",
            "D_full_feedback": "C + accepted-mask feedback into the continuous heatmap",
            "expected_identity": "C and D share the same binary masks; D tests heatmap ranking only",
        },
    }
    save_json(output_dir / "results.json", result)
    print(json.dumps({"num_samples": len(samples), "aggregate": result["aggregate"], "results": str(output_dir / "results.json")}, indent=2))


if __name__ == "__main__":
    main()
