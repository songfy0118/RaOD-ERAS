# Baseline Comparison Protocol

This document prevents invalid comparisons between published numbers obtained with different data, training, masks, or evaluation regions.

| Method | Official source | Method type | Fair comparison requirement | Current status |
|---|---|---|---|---|
| DINO road prototype | This repository | Training-free pixel score | Same images, GT, valid mask, and fixed threshold | Implemented |
| RaOD-ERAS ablations | This repository | Training-free score refinement | Shared DINO features and frozen parameters | Implemented; full run pending |
| [S2M](https://arxiv.org/abs/2311.16516) | CVPR 2024 paper/code | Converts anomaly scores to object masks with promptable segmentation | Feed the same anomaly maps to S2M and evaluate the same split with pixel and component metrics | Reproduction pending |
| [Mask2Anomaly](https://arxiv.org/abs/2307.13316) | ICCV 2023 paper/code | Trained mask-classification anomaly model | Run the official model on the exact local split and valid region | Reproduction pending |
| [DaCUP](https://github.com/vojirt/DaCUP) | WACV 2023 paper/code | Trained patch-based anomaly detector | Match training assumptions, road-only evaluation region, image split, and AP/FPR95 implementation | Reproduction pending |

Published table values must not be copied into the main quantitative table unless the protocol is identical. Otherwise they belong in a separately labeled literature-context table.

## Required Evaluation Rules

1. Freeze all RaOD-ERAS parameters before the final test run.
2. Use dataset-level pixel aggregation for AP, FPR95, precision, recall, F1, and IoU.
3. Report RoadAnomaly21, SMIYC RoadObstacle, and StreetHazards separately; the 189-image union is supplementary only.
4. Report the fixed deployment threshold separately from exploratory best-per-image F1.
5. Evaluate object proposal masks separately from final thresholded heatmaps.
6. Keep image-plane risk actions out of accuracy claims because these datasets provide no trajectory, depth, TTC, or risk ground truth.

## Paper-Valid Ablation

`DINO single-scale -> DINO multi-scale -> risk-guided heatmap -> ERAS refinement -> high-confidence object feedback`

The object-feedback stage may be claimed only if it improves the frozen aggregate protocol on at least two source datasets without a material regression on the third. Otherwise it must be reported as a negative ablation.
