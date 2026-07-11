# RaOD-ERAS

Road-Prototype Anomaly Heatmap with Ego-Lane Risk-Aware Refinement for Unexpected Road Obstacle Segmentation.

This repository contains a lightweight training-free pipeline for road anomaly segmentation in intelligent driving scenes. It uses pretrained DINOv2 features to generate road-prototype anomaly heatmaps and refines them with ERAS, an ego-lane risk-aware component refinement module.

## Method

```text
RGB road image
  -> DINOv2 patch features
  -> road-prototype anomaly heatmap
  -> ERAS risk-aware refinement
  -> binary anomaly mask
  -> warning_events.jsonl
```

The final output is not only a heatmap. Each experiment also exports binary masks and structured warning events with bounding boxes, risk scores, and a suggested downstream action.

## Datasets

Current local datasets:

```text
data/smiyc_road_obstacle/      SMIYC RoadObstacle, 30 public GT images
data/road_anomaly/             RoadAnomaly21, 10 public GT images
data/street_hazards/           StreetHazards partial subset, 149 image/GT pairs available
```

Large datasets and model caches should not be committed to GitHub.

## Quick Start

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run one sample smoke test:

```powershell
python scripts\run_research_experiment.py --dataset road_anomaly --max-samples 1 --out outputs\research_experiment_road_anomaly_smoke
```

Run SMIYC:

```powershell
python scripts\run_research_experiment.py --dataset smiyc
```

Run RoadAnomaly21:

```powershell
python scripts\run_research_experiment.py --dataset road_anomaly
```

Run StreetHazards partial 149:

```powershell
python scripts\run_research_experiment.py --dataset street_hazards --out outputs\research_experiment_street_hazards_149
```

Generate paper figures:

```powershell
python scripts\make_paper_assets.py
```

For a more detailed reproduction guide, see `REPRODUCE.md`.

Package a clean release:

```powershell
python scripts\package_release.py
```

Compile the CCIS PDF:

```powershell
python scripts\build_ccis_pdf.py
```

Prepare final submission artifacts after editing `paper/author_metadata_template.json`:

```powershell
python scripts\prepare_final_submission.py --metadata paper\author_metadata_template.json
```

Replace author metadata before final submission:

```powershell
python scripts\set_paper_metadata.py --authors "First Author\inst{1} \and Second Author\inst{1}" --authorrunning "F. Author et al." --institute "Institution Name, City, Country" --email "author@example.com"
```

## Outputs

Each experiment writes:

```text
outputs/research_experiment_<dataset>/
  metrics.json
  comparison_table.csv
  warning_events.jsonl
  heatmaps/
  binary_masks/
  reports/result_table.md
  reports/method_grid.png
```

Paper figures:

```text
paper/figures/framework_pipeline.png
paper/figures/warning_event_example.png
```

## Current Results

Summary documents:

```text
paper/三数据集实验汇总.md
paper/五轮自我纠错日志.md
paper/投稿冲刺清单.md
paper/submission_readiness_audit.md
paper/paper_outline_en.md
paper/paper_draft_en.md
paper/paper_ccis_latex.tex
paper/ccis_submission_notes.md
paper/references.bib
paper/release_checklist.md
paper/RaOD-ERAS_CCIS_draft.pdf
paper/final_submission_checklist.md
paper/submission_form_fields.md
paper/statements.md
paper/author_metadata_template.json
paper/author_metadata_examples.md
paper/最后提交怎么做.md
```

Main current finding:

```text
SMIYC: DINO + ERAS light gives the best AP/FPR95.
RoadAnomaly21: DINO + ERAS balanced improves DINO under domain shift.
StreetHazards partial 149: raw DINO gives better AP/F1, while ERAS light lowers FPR95.
```

## Claim Boundary

This is not a supervised SOTA model. The intended claim is a lightweight, training-free framework that combines pretrained semantic features, road-prototype anomaly scoring, and risk-aware warning output.
