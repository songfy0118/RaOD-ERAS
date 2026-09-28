# Archived public experiment results

These are the stored results of the public 189-image snapshot, not newly rerun benchmark scores.

## Controlled Ablation

Every variant uses the same images, DINOv2 front end, SAM-B, labels, and evaluation code.

| Variant | Road-aware prompts | Boundary tie-break | Mask feedback | Purpose |
|---|:---:|:---:|:---:|---|
| A: basic boxes | No | No | No | shared-front-end box baseline |
| B: road boxes | Yes | No | No | isolates road-aware prompting |
| C: boundary selection | Yes | Yes | No | tests candidate selection |
| D: full method | Yes | Yes | Yes | tests heatmap feedback |

C and D intentionally share the same binary mask. Their difference is evaluated only on continuous heatmap AP/FPR95.

## Data and Protocol

| Source | Image/GT pairs | Role |
|---|---:|---|
| RoadAnomaly | 10 | real-road cross-dataset validation |
| SMIYC RoadObstacle | 30 | primary real obstacle validation |
| StreetHazards partial | 149 | larger synthetic OOD validation |
| Total | 189 | unified controlled-ablation manifest |

Labels are standardized as `0=normal`, `1=anomaly`, and `255=ignore`. The 189-pair manifest is a convenience evaluation index over public subsets, not a new dataset or the full hidden test split of any benchmark.

Ten deterministic development samples were used to check the implementation, followed by a disjoint 20-sample stability check. Parameters were then frozen for all 189 pairs. Inference never reads GT.

## Main Results

Pixel-micro results on all 189 pairs:

| Variant | Precision | Recall | F1 | IoU | AP | FPR95↓ |
|---|---:|---:|---:|---:|---:|---:|
| A: basic boxes | 0.1282 | 0.1223 | 0.1252 | 0.0668 | 0.0492 | 0.8609 |
| B: road boxes | 0.1587 | 0.1834 | 0.1701 | 0.0930 | 0.0492 | 0.8609 |
| C: boundary selection | 0.1593 | 0.1836 | 0.1706 | 0.0932 | 0.0492 | 0.8609 |
| D: full method | **0.1593** | **0.1836** | **0.1706** | **0.0932** | **0.0810** | **0.8551** |

Road-aware prompting improves image-macro F1 by `+0.0615` (95% CI `[0.0488, 0.0748]`) and IoU by `+0.0363` (`[0.0264, 0.0475]`) relative to A. The C-vs-B effect is not significant. Feedback raises pixel-micro AP from `0.0492` to `0.0810` and reduces FPR95 from `0.8609` to `0.8551`.

Official SMIYC `ObstacleTrack-validation` evaluation:

| Method | AUPR↑ | FPR95↓ | GT-sIoU↑ | PPV↑ | mean F1↑ |
|---|---:|---:|---:|---:|---:|
| DINO base heatmap | 69.26 | **1.29** | 24.42 | 68.87 | 35.17 |
| RiskPrompt-SAM | **91.90** | 1.48 | **48.39** | **71.17** | **62.34** |

Feedback substantially improves AUPR and segment metrics, but FPR95 worsens by 0.18 percentage points. This trade-off must remain visible.
