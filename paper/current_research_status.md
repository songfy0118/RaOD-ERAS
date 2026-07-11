# Current Research Status

## What is ready

- A standardized evaluation set has been assembled at `data/unified_road_anomaly_eval`.
- It contains 189 image/GT pairs from three public sources:
  - 30 SMIYC RoadObstacle samples.
  - 10 RoadAnomaly21 samples.
  - 149 StreetHazards partial samples.
- Every sample in the unified set has a binary ground-truth mask.
- The current pipeline produces:
  - anomaly heatmaps,
  - binary masks,
  - risk-filtered warning masks,
  - warning boxes/events,
  - quantitative tables.

## Main method

The submission method should be written as RaOD-ERAS:

1. Extract dense DINOv2 features from the RGB image.
2. Estimate a road prototype from the lower ego-lane region.
3. Score each region by feature distance from the road prototype.
4. Refine the anomaly heatmap using road/lane/near-field priors.
5. Threshold the heatmap to obtain a binary segmentation mask.
6. Convert connected components into warning events.

This is a training-free visual-foundation-model framework. The paper should not claim a newly trained segmentation backbone.

## Current quantitative picture

The strongest honest claim is not "all metrics improve everywhere".

- On SMIYC RoadObstacle, `dino_eras_light` improves AP, F1, recall, and FPR95 over raw DINOv2, with a small IoU/precision trade-off.
- On RoadAnomaly21, `dino_eras_balanced` strongly improves AP, F1, IoU, precision, and FPR95 over raw DINOv2.
- On StreetHazards partial, raw DINOv2 keeps better AP/F1, while ERAS improves recall and FPR95. This should be written as a risk-control trade-off, not as a universal SOTA result.

## Figures generated

- Main qualitative sheet:
  - `paper/figures/main_qualitative_figure.png`
- Per-sample publication panels:
  - `paper/figures/publication_panels/`
- Figure index:
  - `paper/figures/publication_panels/figure_index.csv`

For the final paper, manually choose 6-8 visually clean panels from the index. Do not blindly use every automatically selected example.

## Tables generated

- `paper/tables/quantitative_digest.md`
- `paper/tables/quantitative_digest.csv`

These are the current paper-ready metric tables.

## What still blocks a real submission

1. Author names, affiliations, and email must be filled.
2. The qualitative figure needs manual curation from the generated panel pool.
3. If the target venue expects a stronger "improvement over prior papers" claim, more work is needed:
   - run official benchmark submission, or
   - reproduce one or two stronger baselines locally, or
   - narrow the claim to lightweight training-free road-obstacle warning.

## Recommended claim

"RaOD-ERAS provides a lightweight, training-free road anomaly heatmap and warning-event pipeline built on DINOv2 features and ego-lane risk refinement. It improves risk-oriented metrics on SMIYC RoadObstacle and RoadAnomaly21 while revealing a precision/recall trade-off on StreetHazards."

