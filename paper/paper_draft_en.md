# RaOD-ERAS: Road-Prototype Anomaly Heatmap with Ego-Lane Risk-Aware Refinement for Unexpected Road Obstacle Segmentation

## Abstract

Unexpected road obstacles are safety-critical for intelligent vehicles because they may not belong to the predefined classes used by standard semantic segmentation or object detection systems. This paper presents RaOD-ERAS, a lightweight training-free framework for road-obstacle anomaly segmentation. The method uses DINOv2 patch features to estimate a road-prototype representation from a coarse road-prior region and generates an anomaly heatmap by measuring feature dissimilarity to the road prototype. We further introduce Ego-Lane Risk-Aware Segmentation Refinement (ERAS), which incorporates road-region priors, ego-lane priors, near-field weighting, and connected-component risk scoring. The final output includes an anomaly heatmap, a binary obstacle mask, and structured warning events with bounding boxes and risk scores. Experiments on SMIYC RoadObstacle, RoadAnomaly21, and a StreetHazards partial subset show that ERAS improves the robustness of DINO-based anomaly heatmaps, especially under cross-dataset domain shift. The proposed framework requires no anomaly training data and is suitable for rapid deployment-oriented road safety research.

## 1. Introduction

Autonomous driving systems must respond not only to common objects such as vehicles, pedestrians, and traffic signs, but also to unexpected obstacles including animals, dropped cargo, unusual road objects, and rare hazards. These obstacles are difficult to handle with closed-set recognition systems because they may not be included in the model's label space.

Road anomaly segmentation aims to identify such unknown or unexpected regions at the pixel level. Existing approaches include uncertainty-based semantic segmentation, energy-based methods, mask-level reasoning, and distillation-based anomaly discovery. Many of these methods require training a dedicated model or rely on a complete semantic segmentation pipeline. For a fast conference paper and lightweight deployment setting, training a new model from scratch is not practical.

This work therefore focuses on a training-free alternative. We use pretrained DINOv2 visual features as semantic representations and construct an anomaly heatmap by comparing each image patch with a road-prototype feature. We then refine the initial heatmap using ERAS, a risk-aware post-processing module designed for road scenes.

The contributions are:

1. A training-free road anomaly segmentation pipeline based on DINOv2 road-prototype heatmaps.
2. ERAS, an ego-lane risk-aware refinement module using road priors, ego-lane priors, near-field weighting, and connected-component risk scoring.
3. A structured warning output format that converts binary anomaly masks into event IDs, bounding boxes, risk scores, and suggested system actions.
4. Experiments on SMIYC RoadObstacle, RoadAnomaly21, and a StreetHazards partial subset.

## 2. Related Work

SegmentMeIfYouCan provides a benchmark for anomaly and obstacle segmentation in driving scenes and motivates evaluating methods on RoadObstacle and RoadAnomaly-style data. PEBAL represents a strong training-based anomaly segmentation baseline using energy-biased learning. Maskomaly and Mask2Anomaly show that mask-level reasoning can improve anomaly segmentation beyond purely pixel-wise scoring. DiCNet uses distillation comparison to discover anomalous regions in semantic segmentation.

Compared with these works, RaOD-ERAS does not train a new anomaly model. It uses pretrained features and scene-specific refinement to produce interpretable heatmaps and warning events. The intended contribution is lightweight and deployment-oriented rather than state-of-the-art supervised performance.

## 3. Method

### 3.1 DINOv2 Road-Prototype Heatmap

Given an RGB road image, DINOv2 extracts patch-level semantic features:

```text
F = DINOv2(I)
```

A coarse road-prior region is estimated from the lower central area of the image. Features inside this prior are averaged to form a normal road prototype:

```text
P_road = mean(F(p), p in road-prior region)
```

Each patch receives an anomaly score based on cosine distance from the road prototype:

```text
A(p) = 1 - cosine_similarity(F(p), P_road)
```

The patch-level scores are resized to the original image resolution to obtain the initial anomaly heatmap.

### 3.2 ERAS Refinement

The initial anomaly heatmap may highlight background textures, shadows, or distant regions. ERAS refines it using four road-scene priors:

```text
road-region prior
ego-lane prior
near-field weighting
connected-component risk scoring
```

The refined heatmap increases scores in safety-critical regions and calibrates connected components according to their size, location, and lane overlap.

### 3.3 Warning Event Output

The refined heatmap is thresholded into a binary mask. Connected components are converted into warning events:

```json
{
  "event_id": "validation0000_dino_eras_balanced_best",
  "bbox_xyxy": [x0, y0, x1, y1],
  "risk_score": 0.692,
  "system_action": "warn_or_slow_down"
}
```

This format allows the perception output to be connected to a downstream driving system or safety monitor.

## 4. Experiments

### 4.1 Datasets

| Dataset | Public GT Used | Role |
|---|---:|---|
| SMIYC RoadObstacle | 30 | Main experiment |
| RoadAnomaly21 | 10 | Cross-dataset generalization |
| StreetHazards partial | 149 available pairs extracted from an interrupted test archive | Difficult synthetic-domain supplement |

### 4.2 Metrics

We report AP, F1, IoU, Precision, Recall, FPR95, and runtime. AP evaluates heatmap ranking quality. F1 and IoU evaluate binary mask quality. FPR95 measures false positives when recall is forced to 95%, which is important for safety-critical anomaly detection.

### 4.3 Quantitative Results

#### SMIYC RoadObstacle

| Method | AP | F1 | IoU | Precision | Recall | FPR95 |
|---|---:|---:|---:|---:|---:|---:|
| RoadContrast | 0.1169 | 0.1739 | 0.1108 | 0.1428 | 0.3690 | 0.4540 |
| RoadContrast + ERAS recall | 0.2180 | 0.2751 | 0.2068 | 0.3456 | 0.3981 | 0.4536 |
| DINO road prototype | 0.5203 | 0.5228 | **0.3998** | **0.4781** | 0.6864 | 0.0268 |
| DINO + ERAS light | **0.5271** | **0.5232** | 0.3989 | 0.4693 | 0.7524 | **0.0256** |

#### RoadAnomaly21

| Method | AP | F1 | IoU | Precision | Recall | FPR95 |
|---|---:|---:|---:|---:|---:|---:|
| RoadContrast | 0.3698 | 0.4473 | 0.3059 | 0.4022 | 0.6188 | 0.6409 |
| RoadContrast + ERAS balanced | **0.3733** | 0.4534 | 0.3099 | **0.4230** | 0.6137 | 0.6191 |
| DINO road prototype | 0.1324 | 0.3119 | 0.1907 | 0.1924 | **0.9314** | 0.6749 |
| DINO + ERAS balanced | 0.3227 | **0.4782** | **0.3289** | 0.3640 | 0.8369 | **0.5592** |

#### StreetHazards Partial

| Method | AP | F1 | IoU | Precision | Recall | FPR95 |
|---|---:|---:|---:|---:|---:|---:|
| RoadContrast | 0.0295 | 0.0634 | 0.0343 | 0.0421 | 0.4158 | 0.6656 |
| DINO road prototype | **0.1133** | **0.1588** | **0.0975** | **0.1424** | 0.4781 | 0.3671 |
| DINO + ERAS light | 0.0691 | 0.1344 | 0.0753 | 0.1077 | 0.5177 | **0.2688** |
| DINO + ERAS balanced | 0.0626 | 0.1293 | 0.0722 | 0.1013 | 0.5432 | 0.2769 |

### 4.4 Discussion

On SMIYC RoadObstacle, DINO road-prototype heatmaps provide a strong training-free signal, and ERAS light slightly improves AP and FPR95 while increasing recall. On RoadAnomaly21, raw DINO heatmaps suffer from domain shift, while ERAS balanced improves F1, IoU, and FPR95. On the 149-sample StreetHazards partial subset, raw DINO obtains the best AP and F1, while ERAS light reduces FPR95. This shows that the current risk prior is useful for false-positive control, but it can over-regularize heatmaps under a strong synthetic-domain shift.

## 5. Limitations

The public GT subsets of SMIYC and RoadAnomaly21 are small. StreetHazards was only partially downloaded in the current experiment, and its 149-sample result is reported as a partial subset rather than an official benchmark number. The method also relies on geometric road priors, which may fail in unusual camera viewpoints or non-road scenes. Finally, RaOD-ERAS does not replace supervised anomaly segmentation models; it is designed as a lightweight training-free baseline and warning module.

## 6. Conclusion

This paper introduced RaOD-ERAS, a training-free framework for unexpected road obstacle segmentation. By combining DINOv2 road-prototype anomaly heatmaps with ego-lane risk-aware refinement, the method produces interpretable heatmaps, binary masks, and structured warning events. Experiments across three datasets show that ERAS improves robustness under several settings, while also revealing clear limitations under synthetic-domain shift. Future work will extend evaluation to full StreetHazards and official benchmark submissions.

## References

- Chan et al., "SegmentMeIfYouCan: A Benchmark for Anomaly Segmentation", NeurIPS Datasets and Benchmarks, 2021.
- Hendrycks et al., "Scaling Out-of-Distribution Detection for Real-World Settings", ICML Workshop, 2019.
- Tian et al., "Unsupervised Road Anomaly Detection with Language Anchors", ICRA, 2022.
- Tian et al., "Pixel-wise Energy-biased Abstention Learning for Anomaly Segmentation on Complex Urban Driving Scenes", ECCV, 2022.
- Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", arXiv, 2023.
