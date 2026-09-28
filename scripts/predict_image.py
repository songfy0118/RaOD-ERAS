"""Run frozen DINOv2 + SAM on one RGB image, without a label mask."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from src.raod_eras.inference import predict_image
from src.raod_eras.io_utils import save_heatmap, save_mask


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/single_image"))
    parser.add_argument("--max-side", type=int, default=1024, help="Resize longest side; 0 keeps native resolution.")
    args = parser.parse_args(argv)
    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    if not args.sam_checkpoint.is_file():
        parser.error(f"SAM checkpoint not found: {args.sam_checkpoint}. See REPRODUCE.md.")
    if args.sam_checkpoint.stat().st_size < 1024:
        parser.error("SAM checkpoint is too small; it may be a Git LFS pointer.")
    if args.max_side < 0:
        parser.error("--max-side must be zero or positive")
    try:
        import torch
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as exc:
        parser.error(f"Missing model dependency: {exc}. Install requirements.txt.")
    from src.raod_eras.dino_features import DINOEncoder

    with Image.open(args.image) as opened:
        image = opened.convert("RGB")
    original_size = image.size
    if args.max_side:
        image.thumbnail((args.max_side, args.max_side), Image.Resampling.LANCZOS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint=str(args.sam_checkpoint)).to(device).eval()
    predictor = SamPredictor(sam)
    encoder = DINOEncoder("dinov2_vits14", 518)
    if device == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    score, result = predict_image(encoder, predictor, image)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if not np.isfinite(score).all() or not np.isfinite(result.refined_score).all():
        raise ValueError("Non-finite inference scores; refusing to export results.")
    args.out.mkdir(parents=True, exist_ok=True)
    save_mask(args.out / "mask.png", result.mask)
    save_heatmap(args.out / "anomaly.png", result.refined_score)
    np.savez_compressed(args.out / "scores.npz", base_score=score, refined_score=result.refined_score, mask=result.mask)
    overlay = np.asarray(image).copy()
    overlay[result.mask] = (0.45 * overlay[result.mask] + 0.55 * np.array([255, 60, 20])).astype(np.uint8)
    Image.fromarray(overlay).save(args.out / "overlay.png")
    metadata = dict(input=str(args.image), original_size=list(original_size), output_size=list(image.size),
                    device=device, torch_version=torch.__version__, seconds=elapsed,
                    sam_checkpoint=args.sam_checkpoint.name, dino_model=encoder.model_name,
                    boxes=[list(box) for box in result.boxes],
                    scope="Single-image inference; no ground truth or benchmark metrics.",
                    method="Public RiskPrompt-SAM D ablation; not a vehicle-control system.")
    (args.out / "result.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved mask, overlay, scores and metadata to {args.out}")


if __name__ == "__main__":
    main()
