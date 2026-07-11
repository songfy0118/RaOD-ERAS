# Paper Outline

## Title

RaOD-ERAS: Road-Prototype Anomaly Heatmap with Ego-Lane Risk-Aware Refinement for Unexpected Road Obstacle Segmentation

## Abstract Draft

Unexpected road obstacles pose a critical risk to intelligent vehicles because they may not belong to predefined semantic categories. This paper presents RaOD-ERAS, a lightweight training-free framework for road-obstacle anomaly segmentation. Instead of training a new segmentation network, RaOD-ERAS uses DINOv2 features to build a road-prototype anomaly heatmap, where pixels that deviate from the normal road representation receive higher anomaly scores. We further introduce Ego-Lane Risk-Aware Segmentation Refinement (ERAS), which incorporates road-region priors, ego-lane priors, near-field weighting, and connected-component risk scoring. The framework outputs both pixel-level anomaly heatmaps and structured warning events with bounding boxes and risk scores. Experiments on SMIYC RoadObstacle, RoadAnomaly21, and a StreetHazards partial subset show that ERAS improves the robustness of DINO-based anomaly heatmaps under cross-dataset domain shifts. The method requires no anomaly training data and is suitable for rapid deployment and preliminary safety-warning research.

## Method Sections

### Road-Prototype Heatmap

```text
Input RGB image -> DINOv2 patch features -> road prototype -> cosine distance heatmap
```

### ERAS Refinement

```text
Initial heatmap -> road prior -> ego-lane prior -> near-field weighting -> component risk scoring
```

### Warning Event Output

```text
Binary mask -> connected components -> event_id, bbox, risk_score, system_action
```

## Experiment Sections

### Datasets

```text
SMIYC RoadObstacle: 30 public GT images
RoadAnomaly21: 10 public GT images
StreetHazards partial: 20 evaluated images from 149 available image/GT pairs
```

### Metrics

```text
AP, F1, IoU, Precision, Recall, FPR95, runtime
```

### Main Finding

```text
On SMIYC, DINO + ERAS light achieves the best AP and FPR95.
On RoadAnomaly21, DINO + ERAS balanced improves DINO under domain shift.
On StreetHazards partial, ERAS improves DINO but absolute performance remains low.
```

## Limitation

The method relies on geometric road priors and pretrained visual features. It does not replace fully supervised anomaly segmentation methods and should not be claimed as state-of-the-art. Public GT availability in SMIYC and RoadAnomaly21 is limited, and StreetHazards results are currently based on a partial subset due to interrupted dataset download.
