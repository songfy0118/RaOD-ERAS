# Frozen Experiment Protocol

## Method Freeze

- Calibration split: first 10 deterministic stratified unified samples.
- Validation split: next 20 deterministic stratified samples (`sample_offset=10`).
- DINOv2 backbone: `dinov2_vits14`, input size 518.
- Risk fusion: conservative residual blend with `alpha=0.20`.
- Heatmap output: conservative risk-guided DINO score.
- Binary output: ERAS-balanced mask at fixed threshold 0.70, followed by removal of components smaller than 0.00005 of image area.
- No test GT is used by the model, threshold, or object refinement.
- Object-feedback remains an experimental ablation and is not the final output.

## Why Dual Output Is Required

Continuous anomaly ranking and binary object segmentation have different objectives. The risk-guided score is retained for AP/FPR95, while ERAS converts it into an object-like fixed-threshold mask for precision/recall/F1/IoU. Replacing the ranked heatmap with the ERAS score harms StreetHazards AP, so the final method exposes both outputs explicitly.

## Disjoint 20-Sample Validation

| Dataset | Method | AP | FPR95 | Precision | Recall | F1 | IoU |
|---|---|---:|---:|---:|---:|---:|---:|
| RoadAnomaly21 | DINO | 0.1116 | 0.9915 | 0.0927 | 0.2953 | 0.1411 | 0.0759 |
| RoadAnomaly21 | RaOD-ERAS final | **0.1131** | **0.9873** | **0.2764** | 0.0772 | **0.1207** | **0.0642** |
| SMIYC | DINO | 0.0008 | 0.6626 | 0.0010 | 0.8945 | 0.0019 | 0.0010 |
| SMIYC | RaOD-ERAS final | **0.0010** | **0.6540** | **0.0155** | **0.7492** | **0.0304** | **0.0155** |
| StreetHazards | DINO | 0.0348 | 0.5106 | 0.0044 | **0.9948** | 0.0087 | 0.0044 |
| StreetHazards | RaOD-ERAS final | **0.0502** | **0.4269** | **0.0150** | 0.5486 | **0.0292** | **0.0148** |

The method improves AP, FPR95, precision, F1, and IoU over the same-threshold DINO baseline on every source in this disjoint validation. Threshold 0.70 was selected only on the 10-sample calibration split. Recall decreases on RoadAnomaly21 and StreetHazards, so the paper must report the full precision-recall tradeoff rather than claim universal metric improvement. Small-component cleanup contributes only marginal gains and must not be presented as a major contribution.

The unified aggregate FPR95 is not a primary paper metric because per-image min-max normalization makes score scales incomparable across datasets. Report each public benchmark separately.

## Reproduction Commands

```powershell
python scripts\run_research_experiment.py --dataset unified --max-samples 10 --sample-offset 0 --calibration-sweep --out outputs\calibration_10
python scripts\run_research_experiment.py --dataset unified --max-samples 20 --sample-offset 10 --risk-fusion-alpha 0.20 --out outputs\final_validation_20
```

## Next Paper Work

1. Run the three source datasets separately with frozen parameters.
2. Add confidence intervals and runtime/memory reporting.
3. Reproduce at least one external baseline under the same valid-mask protocol.
4. Use the 20-sample run for development evidence only; use full source splits for final tables.
