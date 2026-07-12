# CIVS Project Structure

## Clean Folder Layout

```text
CIVS/
  data/                 final local datasets and unified index
  outputs/              final experiment outputs only
  paper/                draft paper, figures, tables, submission notes
  scripts/              runnable experiment / figure / packaging scripts
  src/raod_eras/        core method code
  dist/                 unified dataset archive and generated packages
  _archive_unused/      old reference/smoke/Fishyscapes material, not used in final paper
```

## Final Dataset Folders

```text
data/smiyc_road_obstacle/paper_subset/
  images/               RGB input images
  gt/                   GT masks

data/road_anomaly/paper_subset/
  images/               RGB input images
  gt/                   GT masks

data/street_hazards/paper_subset/
  images/               RGB input images
  gt/                   GT masks, anomaly label is 14

data/unified_road_anomaly_eval/
  images/               combined images copied from the three datasets
  gt_binary/            unified binary GT masks
  metadata/samples.csv  source, image path, GT path, size, anomaly pixels
```

## Core Python Files

### `src/raod_eras/`

| File | Meaning |
|---|---|
| `config.py` | Experiment, dataset, method, and output configuration dataclasses |
| `datasets.py` | Loads image/GT pairs and converts GT into binary anomaly masks |
| `dino_features.py` | Loads pretrained DINOv2 and extracts feature-based anomaly heatmaps |
| `baselines.py` | Lightweight road-contrast baseline heatmap |
| `priors.py` | Road trapezoid, ego-lane, near-field priors, score normalization |
| `refinement.py` | ERAS variants and connected-component risk refinement |
| `metrics.py` | AP, F1, IoU, Precision, Recall, FPR95 evaluation |
| `object_refinement.py` | Seeded object segmentation and mask-to-heatmap feedback |
| `io_utils.py` | Save JSON, CSV, heatmaps, binary masks |
| `reporting.py` | Result tables and method-grid image generation |
| `risk_planning.py` | Image-plane candidate trajectory and braking cost evaluation |
| `experiment.py` | Main experiment loop: run methods, save outputs, compute metrics |
| `__init__.py` | Package marker |

### `scripts/`

| File | Meaning |
|---|---|
| `run_research_experiment.py` | Runs SMIYC, RoadAnomaly, StreetHazards, or the unified archive |
| `build_unified_dataset.py` | Copies images, converts binary GT, and writes unified metadata |
| `make_metric_digest.py` | Builds paper metric tables from `metrics.json` |
| `make_ablation_and_objective_tables.py` | Builds ablation table and operating-point objective table |
| `make_publication_figures.py` | Builds per-sample panels and main qualitative figure |
| `make_paper_assets.py` | Builds framework and warning-example figures |
| `build_ccis_pdf.py` | Compiles the Springer/CCIS PDF from LaTeX |
| `set_paper_metadata.py` | Replaces placeholder author/institute/email metadata |
| `prepare_final_submission.py` | One-command final flow after author metadata is ready |
| `package_submission.py` | Creates the CCIS submission zip |
| `package_release.py` | Creates a lightweight GitHub release zip |
| `final_submission_check.py` | Checks PDF, submission zip, release zip, and author placeholders |

## How To Run

Download and extract the Git LFS dataset archive:

```powershell
git lfs pull
tar -xf dist\unified_road_anomaly_eval_189.zip
```

One-image smoke test:

```powershell
python scripts\run_research_experiment.py --dataset unified --max-samples 1 --out outputs\test_one
```

Full final experiments:

```powershell
python scripts\run_research_experiment.py --dataset smiyc
python scripts\run_research_experiment.py --dataset road_anomaly
python scripts\run_research_experiment.py --dataset street_hazards --out outputs\research_experiment_street_hazards_149
```

Regenerate paper assets:

```powershell
python scripts\make_metric_digest.py
python scripts\make_ablation_and_objective_tables.py
python scripts\make_publication_figures.py
python scripts\make_paper_assets.py
```

## What The Outputs Mean

| Output | Meaning |
|---|---|
| `heatmaps/<method>/*.png` | Model anomaly heatmap. Red/yellow means more anomalous. |
| `binary_masks/<method>/*.png` | Black-white mask from thresholding heatmap. White means predicted anomaly. |
| `warning_events.jsonl` | Connected components converted into warning events: bbox, risk score, action. |
| `risk_plans.jsonl` | Candidate trajectory costs and selected rule action. |
| `risk_summary.json` | Aggregate action counts and normalized clearance proxy. |
| `ablation_manifest.json` | Ordered method chain used for ablation reporting. |
| `metrics.json` | Average metrics over the dataset. |
| `comparison_table.csv` | Per-image metrics for every method. |
| `reports/result_table.md` | Human-readable result table. |

## How The Methods Are Compared

Compared methods:

```text
roadcontrast
roadcontrast_eras_light / balanced / recall
dino
dino_eras_light / balanced / recall
```

Metrics:

| Metric | Meaning | Direction |
|---|---|---|
| AP | Ranking quality of the heatmap against GT pixels | Higher better |
| F1 | Balance of precision and recall after thresholding | Higher better |
| IoU | Overlap between predicted binary mask and GT | Higher better |
| Precision | Predicted anomaly pixels that are truly anomaly | Higher better |
| Recall | GT anomaly pixels found by the method | Higher better |
| FPR95 | False positive rate when recall is near 95% | Lower better |

The ablation objective is:

```text
L = 0.40 * (1 - AP) + 0.35 * (1 - F1) + 0.25 * FPR95
```

This is not a neural-network loss. It is an exploratory validation criterion. Current F1/IoU values use per-image best-F1 thresholds; saved masks and warnings use the fixed inference threshold configured by `--output-threshold`.

## What Is Archived

`_archive_unused/` contains old material that is not part of the final paper pipeline:

```text
external__DiCNet
data__fishyscapes_laf
outputs__portable_root_smoke
outputs__research_experiment
outputs__research_experiment_road_anomaly_guarded
outputs__research_experiment_road_anomaly_smoke
outputs__research_experiment_street_hazards
outputs__research_experiment_street_hazards_20
outputs__research_experiment_street_hazards_smoke
```
