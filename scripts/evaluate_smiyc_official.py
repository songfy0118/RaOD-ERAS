from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "external" / "road-anomaly-benchmark"
sys.path.insert(0, str(BENCHMARK))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate cached RiskPrompt scores with the official SMIYC suite.")
    parser.add_argument("--method-name", default="RiskPromptSAM-v2")
    parser.add_argument("--cache", type=Path, default=Path("outputs/riskprompt_ablation_cache"))
    parser.add_argument("--score-key", default="feedback_score")
    args = parser.parse_args()

    from road_anomaly_benchmark.evaluation import Evaluation

    cache_dir = ROOT / args.cache
    evaluation = Evaluation(method_name=args.method_name, dataset_name="ObstacleTrack-validation")
    records: list[dict[str, object]] = []
    for frame in evaluation.get_frames():
        cache_path = cache_dir / f"smiyc__{frame.fid}.npz"
        if not cache_path.exists():
            raise FileNotFoundError(cache_path)
        cached = np.load(cache_path)
        if args.score_key not in cached.files:
            raise KeyError(f"{args.score_key} not found in {cache_path}; keys={cached.files}")
        score = cached[args.score_key].astype(np.float32)
        if score.shape != frame.image.shape[:2]:
            raise ValueError(f"Shape mismatch for {frame.fid}: score={score.shape}, image={frame.image.shape[:2]}")
        if not np.isfinite(score).all() or score.min() < 0.0 or score.max() > 1.0:
            raise ValueError(f"Invalid score range for {frame.fid}: [{score.min()}, {score.max()}]")
        evaluation.save_output(frame, score)
        records.append(
            {
                "fid": frame.fid,
                "shape": list(score.shape),
                "min": float(score.min()),
                "max": float(score.max()),
            }
        )
    evaluation.wait_to_finish_saving()

    pixel_result = evaluation.calculate_metric_from_saved_outputs("PixBinaryClass", frame_vis=False)
    segment_result = evaluation.calculate_metric_from_saved_outputs("SegEval-ObstacleTrack", frame_vis=False)
    summary = {
        "method": args.method_name,
        "dataset": "ObstacleTrack-validation",
        "num_frames": len(records),
        "prediction_validation": records,
        "pixel_result_repr": repr(pixel_result),
        "segment_result_repr": repr(segment_result),
        "protocol": "Official SegmentMeIfYouCan road-anomaly-benchmark Evaluation API.",
    }
    safe_name = args.method_name.replace("/", "_").replace("\\", "_")
    output = ROOT / "outputs" / "smiyc_official_protocol" / f"protocol_run_summary_{safe_name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
