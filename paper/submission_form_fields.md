# Submission Form Fields

## Title

```text
RaOD-ERAS: Road-Prototype Anomaly Heatmap with Ego-Lane Risk-Aware Refinement for Unexpected Road Obstacle Segmentation
```

## Short Title

```text
RaOD-ERAS for Road Anomaly Segmentation
```

## Abstract

```text
Unexpected road obstacles such as animals, debris, and other rare objects pose safety risks for intelligent vehicles because they are often under-represented in closed-set perception training data. This paper presents RaOD-ERAS, a lightweight training-free framework for unexpected road obstacle segmentation. The method uses pretrained DINOv2 visual features to construct a road-prototype anomaly heatmap and then applies Ego-lane Risk-Aware Segmentation (ERAS), which combines road-region, ego-lane, near-field, and connected-component risk priors. The final output includes an interpretable anomaly heatmap, a binary anomaly mask, and structured warning events that can be consumed by a downstream safety module. Experiments on SMIYC RoadObstacle, RoadAnomaly21, and a 149-sample StreetHazards partial subset show that the proposed refinement improves AP and FPR95 on SMIYC, improves F1/IoU/FPR95 under RoadAnomaly21 domain shift, and reduces false positives on the difficult StreetHazards partial subset. The results indicate that road-prototype features and risk-aware refinement are useful for fast conference-level road anomaly studies, while also revealing limitations under strong synthetic-domain shift.
```

## Keywords

```text
Road anomaly segmentation; Intelligent vehicles; DINOv2
```

## Main Contribution

```text
This paper proposes a training-free road anomaly segmentation framework that combines DINOv2 road-prototype heatmaps with ego-lane risk-aware refinement and exports heatmaps, binary masks, and structured warning events.
```

## Claim Boundary

```text
The method is not claimed as a supervised state-of-the-art model. It is positioned as a lightweight, training-free, reproducible baseline and warning-output module for unexpected road obstacle segmentation.
```

## Dataset Statement

```text
Experiments use SMIYC RoadObstacle with 30 public GT images, RoadAnomaly21 with 10 public GT images, and a 149-sample partial StreetHazards subset extracted from the locally available interrupted test archive. The StreetHazards result is reported as a partial subset rather than an official benchmark result.
```

## Code Availability

```text
The reproducible code and paper artifacts are prepared for GitHub release. Large datasets, model weights, and generated heatmap/mask outputs are excluded from the repository package.
```
