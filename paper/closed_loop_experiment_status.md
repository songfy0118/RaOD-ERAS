# Closed-Loop Experiment Status

## Implemented

- single-scale and multi-scale DINOv2 road-prototype heatmaps;
- local road-contrast and road/lane/near-field risk fusion;
- ERAS spatial refinement;
- seeded object candidate segmentation with boundary, area, road, lane, and confidence filtering;
- mask-to-heatmap feedback;
- fixed-threshold masks and warning events without per-image GT thresholds;
- component F1, boundary F1, fixed F1/IoU, AP, and FPR95 outputs;
- image-plane scoring of keep-lane, left-shift, right-shift, and brake candidates.

## Five-Sample Engineering Check

The five-sample run is a stability check, not a paper result.

| Method | AP | Fixed F1 | Fixed IoU | Component F1 | FPR95 |
|---|---:|---:|---:|---:|---:|
| DINO road prototype | 0.1462 | 0.1898 | 0.1072 | 0.1208 | 0.6721 |
| DINO multi-scale | 0.1402 | 0.1846 | 0.1041 | 0.1209 | 0.6999 |
| Risk-guided heatmap | 0.3493 | 0.3141 | 0.1912 | 0.0084 | 0.4453 |
| Risk heatmap + ERAS light | 0.3964 | 0.4489 | 0.3056 | 0.0092 | 0.4241 |
| Risk heatmap + object feedback | 0.3370 | 0.0780 | 0.0408 | 0.0000 | 0.4576 |

The risk-guided heatmap is promising. The current object-feedback module is not yet supported as an accuracy improvement and must remain an experimental ablation.

## Required Before Paper Claims

1. Freeze parameters without tuning on test GT.
2. Run all 189 samples and report each source dataset separately.
3. Add dataset-level aggregate pixel metrics, not only per-image means.
4. Compare against cited published/open-source baselines under the same split.
5. Report runtime, memory, failure cases, and confidence intervals or bootstrap variation.
6. Validate component matching thresholds and risk rules on a held-out calibration subset.
7. Treat trajectory output as image-plane rule validation, not vehicle control.
