# RaOD-ERAS

Road-Prototype Anomaly Heatmap with Ego-Lane Risk-Aware Refinement for Unexpected Road Obstacle Segmentation.

This project is the cleaned CIVS conference-paper workspace. The model/framework is already built. It is a training-free road anomaly segmentation pipeline:

```text
RGB image
  -> DINOv2 feature extractor
  -> road-prototype anomaly heatmap
  -> ERAS ego-lane risk refinement
  -> binary anomaly mask
  -> warning event JSONL
```

It does not train a new backbone from scratch. It uses pretrained DINOv2 and our road-prototype plus ERAS refinement logic.

## Final Datasets

The final experiments use these local datasets:

| Dataset | Local folder | Samples with GT | Role |
|---|---|---:|---|
| SMIYC RoadObstacle | `data/smiyc_road_obstacle` | 30 | Main road-obstacle benchmark |
| RoadAnomaly21 | `data/road_anomaly` | 10 | Cross-dataset anomaly validation |
| StreetHazards partial | `data/street_hazards` | 149 | Larger partial OOD validation |
| Unified index | `data/unified_road_anomaly_eval` | 189 | Standardized combined evaluation set |

Fishyscapes-only mask data and old reference code were moved to `_archive_unused/`.

## Quick Run

Open this folder in PyCharm:

```text
C:\Users\93785\Desktop\CIVS
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run one image first:

```powershell
python scripts\run_research_experiment.py --dataset road_anomaly --max-samples 1 --out outputs\test_one
```

Run the three final experiments:

```powershell
python scripts\run_research_experiment.py --dataset smiyc
python scripts\run_research_experiment.py --dataset road_anomaly
python scripts\run_research_experiment.py --dataset street_hazards --out outputs\research_experiment_street_hazards_149
```

Generate paper tables and figures:

```powershell
python scripts\make_metric_digest.py
python scripts\make_ablation_and_objective_tables.py
python scripts\make_publication_figures.py
```

## Final Outputs

Each experiment writes:

```text
outputs/research_experiment_<dataset>/
  metrics.json              final averaged metrics
  comparison_table.csv      per-image metrics
  warning_events.jsonl      warning boxes and risk scores
  heatmaps/                 anomaly heatmaps
  binary_masks/             thresholded black-white masks
  reports/result_table.md   readable result table
```

Current formal output folders:

```text
outputs/research_experiment_smiyc
outputs/research_experiment_road_anomaly
outputs/research_experiment_street_hazards_149
```

Paper-ready materials:

```text
paper/figures/main_qualitative_figure.png
paper/figures/framework_pipeline.png
paper/figures/warning_event_example.png
paper/tables/quantitative_digest.md
paper/tables/ablation_objective.md
paper/RaOD-ERAS_CCIS_draft.pdf
```

## Current Result Summary

| Dataset | Best honest claim |
|---|---|
| SMIYC RoadObstacle | `dino_eras_light` slightly improves AP/F1/Recall/FPR95 over raw DINOv2 |
| RoadAnomaly21 | `dino_eras_balanced` improves AP/F1/IoU/FPR95 under domain shift |
| StreetHazards partial | ERAS improves Recall/FPR95, but raw DINOv2 keeps better AP/F1 |

So the paper should not claim universal SOTA. The correct claim is a lightweight, training-free, risk-aware road anomaly warning framework.

## More Details

Read:

```text
PROJECT_STRUCTURE.md
REPRODUCE.md
paper/current_research_status.md
paper/submission_form_fields.md
```

Before final submission, fill:

```text
paper/author_metadata_template.json
```

Then run:

```powershell
python scripts\prepare_final_submission.py --metadata paper\author_metadata_template.json
```
