# Reproduce RaOD-ERAS Experiments

## Environment

```powershell
python -m pip install -r requirements.txt
```

DINOv2 weights are downloaded automatically by `torch.hub` on the first run. The warning `xFormers is not available` is acceptable.

## Downloadable Unified Dataset

The repository provides one Git LFS archive containing 189 standardized image/GT pairs:

```powershell
git lfs pull
tar -xf dist\unified_road_anomaly_eval_189.zip
```

This creates:

```text
data/unified_road_anomaly_eval/images
data/unified_road_anomaly_eval/gt_binary
data/unified_road_anomaly_eval/metadata
```

The original source-specific layout, required only by `--dataset smiyc`, `road_anomaly`, or `street_hazards`, is:

```text
data/smiyc_road_obstacle/paper_subset/images
data/smiyc_road_obstacle/paper_subset/gt
data/road_anomaly/paper_subset/images
data/road_anomaly/paper_subset/gt
data/street_hazards/paper_subset/images
data/street_hazards/paper_subset/gt
```

The current local experiments used:

```text
SMIYC RoadObstacle: 30 public GT images
RoadAnomaly21: 10 public GT images
StreetHazards partial: 149 available image/GT pairs
```

## Smoke Test

```powershell
python scripts\run_research_experiment.py --dataset unified --max-samples 1 --out outputs\test_one
```

## Full Commands

Run the downloadable unified evaluation set:

```powershell
python scripts\run_research_experiment.py --dataset unified --out outputs\research_experiment_unified
```

Source-specific commands require the original `paper_subset` folders:

```powershell
python scripts\run_research_experiment.py --dataset smiyc
python scripts\run_research_experiment.py --dataset road_anomaly
python scripts\run_research_experiment.py --dataset street_hazards --out outputs\research_experiment_street_hazards_149
python scripts\make_paper_assets.py
```

## Formal Output Directories

```text
outputs/research_experiment_smiyc
outputs/research_experiment_road_anomaly
outputs/research_experiment_street_hazards_149
paper/figures
```

Older exploratory output directories may exist locally, but they should not be used for the paper.

## Current Main Numbers

These are exploratory per-image results, not official benchmark submissions. Binary masks and warning events use a fixed inference threshold; GT-selected thresholds are used only in the current exploratory metric report.

| Dataset | Main observation |
|---|---|
| SMIYC RoadObstacle | DINO + ERAS light gives the best AP/FPR95 |
| RoadAnomaly21 | DINO + ERAS balanced improves F1/IoU/FPR95 under domain shift |
| StreetHazards partial 149 | Raw DINO gives better AP/F1; ERAS light lowers FPR95 |

## Compile the CCIS Draft

The official Springer LaTeX2e proceedings template has been prepared under:

```text
paper/ccis_build/
```

Compile with:

```powershell
python scripts\build_ccis_pdf.py
```

The current compiled draft PDF is:

```text
paper/RaOD-ERAS_CCIS_draft.pdf
```

To replace placeholder author metadata:

```powershell
python scripts\set_paper_metadata.py --authors "First Author\inst{1} \and Second Author\inst{1}" --authorrunning "F. Author et al." --institute "Institution Name, City, Country" --email "author@example.com"
python scripts\build_ccis_pdf.py
```

## Package a Clean Release

```powershell
python scripts\package_release.py
```

This creates:

```text
dist/raod_eras_release.zip
release_manifest.md
```

## Package CCIS Submission Files

```powershell
python scripts\package_submission.py
```

This creates:

```text
dist/raod_eras_ccis_submission_package.zip
```

## Final Submission Check

```powershell
python scripts\final_submission_check.py
```

This check is expected to fail until placeholder author metadata is replaced.

## One-Command Final Build

After editing `paper/author_metadata_template.json`, run:

```powershell
python scripts\prepare_final_submission.py --metadata paper\author_metadata_template.json
```

For draft artifacts with placeholder metadata:

```powershell
python scripts\prepare_final_submission.py --metadata paper\author_metadata_template.json --allow-placeholders
```
