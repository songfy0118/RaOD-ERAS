# Validation record — 28 September 2026

This maintenance release adds a standalone input-image path around the public D ablation. It does not change the archived method thresholds or benchmark tables.

## Pass 1: repository and scope review

- Compared the public snapshot with its stored 189-image results and documented the RPSAN / RiskPrompt-SAM naming.
- Identified the missing label-free single-image CLI and added it using the existing scoring/prompt implementation.
- Kept later local manuscript/code changes out of this public-snapshot update.
- Corrected the documented unified label directory to `gt_labels` and removed dependence on a particular Conda environment name.

## Pass 2: tests and real model execution

```bash
python -m unittest discover -s tests -v
python scripts/predict_image.py --help
python scripts/predict_image.py --image path/to/synthetic__0000.jpg --sam-checkpoint path/to/sam_vit_b_01ec64.pth --out outputs/portfolio_smoke --max-side 512
```

All **12 tests passed**: the eight existing algorithm/metric tests plus four input/API/CLI tests. The new tests cover a uniform scene, invalid scores before SAM, image/score handoff without GT, and CLI help/missing inputs.

A real inference run used the existing local DINOv2 cache and SAM ViT-B checkpoint with PyTorch 2.7.0 and CUDA. A 2,048 × 1,024 synthetic road example was resized to 512 × 256. The run wrote a boolean mask, two finite score arrays in [0, 1], a PNG overlay, heatmap and JSON metadata. The saved PNG mask matched the NPZ mask pixel-for-pixel.

The overlay was visually inspected. It includes false-positive regions; this check establishes that inference and exports work, not that the example achieves good obstacle accuracy.

## Pass 3: public documentation and result consistency

- Checked the preserved A–D table against the tracked result JSON.
- Checked the official upstream model/evaluator links and local documentation links.
- Reviewed output dimensions, resize semantics, timing boundaries and invalid-input handling.
- Checked the Git diff for whitespace and unintended data/checkpoint additions.

## Limits

No full 189-image rerun or official SMIYC evaluation was performed in this maintenance pass. Published tables remain clearly labeled as archived results. Real inference used available local weights; a completely fresh external installation/download and remote Git LFS availability were not validated. DINOv2 currently follows the upstream Torch Hub branch, so future upstream changes can affect setup.
