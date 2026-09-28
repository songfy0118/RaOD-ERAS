# Running inference and experiments

Run commands from the repository root with Python 3.10+. The environment does not need a specific Conda name.

## 1. Install and test

Install a matching PyTorch/torchvision build for your device, then:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/predict_image.py --help
```

For lightweight tests only, the dependencies are `numpy pillow scipy matplotlib`. The unit suite does not download or run pretrained models. End-to-end inference additionally requires PyTorch, torchvision, Segment Anything and weights.

## 2. Single-image inference

Download SAM **ViT-B**, `sam_vit_b_01ec64.pth`, from [Meta's official checkpoint list](https://github.com/facebookresearch/segment-anything#model-checkpoints). DINOv2 ViT-S/14 loads from [Meta's Torch Hub repository](https://github.com/facebookresearch/dinov2) and downloads on first use. Respect the upstream weight/code licenses.

```bash
python scripts/predict_image.py --image path/to/road.jpg --sam-checkpoint path/to/sam_vit_b_01ec64.pth --out outputs/example
```

The CLI checks the image and checkpoint paths before loading models. It refuses tiny checkpoint placeholders. The default resize bounds memory use; `--max-side 0` preserves native resolution. A resize can change predictions and must not be treated as the archived benchmark protocol.

## 3. Prepare the archived 189-pair evaluation

```bash
git lfs install
git lfs pull --include="dist/unified_road_anomaly_eval_189.zip"
python -m zipfile -e dist/unified_road_anomaly_eval_189.zip data/unified_road_anomaly_eval
```

The archive should be **239,849,600 bytes**, with SHA-256
`73214f3a1a1a8123f481551cc6818a13c23fe10eb9041d21a3b95a325b28d5c0`.
Check against [the manifest](dist/unified_road_anomaly_eval_189.manifest.json). A roughly 134-byte file is an LFS pointer, not a usable ZIP. LFS access depends on repository availability/quota. The source datasets retain their original access and usage terms.

The loader expects `images/` and `gt_labels/` inside `data/unified_road_anomaly_eval/`; label values are 0/1/255. Place the SAM checkpoint at `external/S2M_official/tools/sam_vit_b_01ec64.pth`, or pass `--sam-checkpoint` to both inference stages below.

## 4. Rebuild caches and evaluate

Start with one image:

```bash
python scripts/run_s2m_comparison.py --max-samples 1 --out outputs/benchmark_smoke
python scripts/run_prompt_ablation.py --max-samples 1 --source-cache outputs/benchmark_smoke/cache --ablation-cache outputs/ablation_smoke_cache --out outputs/ablation_smoke --save-visuals
```

The archived full procedure:

```bash
python scripts/run_s2m_comparison.py --max-samples 189 --ugains-threshold 0.60 --out outputs/riskprompt_full_189
python scripts/run_prompt_ablation.py --max-samples 189 --source-cache outputs/riskprompt_full_189/cache --out outputs/riskprompt_ablation_full_189_v2 --save-visuals
python scripts/analyze_ablation_results.py outputs/riskprompt_ablation_full_189_v2/results.json
```

Caches are keyed by sample name, not by code or weight fingerprints. Use new output/cache folders, or `--no-cache` for both generation stages, whenever code, data, weights or configuration changes. Existing summary files do not substitute for missing per-image caches.

## 5. Official SMIYC protocol

```bash
python scripts/evaluate_smiyc_official.py --method-name RiskPromptSAM-v2 --cache outputs/riskprompt_ablation_cache --score-key feedback_score
```

This separately requires the [official road-anomaly-benchmark](https://github.com/segmentmeifyoucan/road-anomaly-benchmark) under `external/road-anomaly-benchmark`, its dependencies and configured `ObstacleTrack-validation` data. The repository's 189-image summary is not a replacement for that benchmark's setup.

The portfolio maintenance checks ran unit tests and one-image inference, not a fresh full 189-image or official-protocol evaluation. Historical measurements remain identified as such in [docs/RESULTS.md](docs/RESULTS.md).
