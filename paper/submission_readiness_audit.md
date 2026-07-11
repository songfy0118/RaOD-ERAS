# Submission Readiness Audit

## Current Paper Version

Title:

```text
RaOD-ERAS: Road-Prototype Anomaly Heatmap with Ego-Lane Risk-Aware Refinement for Unexpected Road Obstacle Segmentation
```

Current claim:

```text
Training-free road anomaly heatmap generation using DINOv2 road prototypes, plus ERAS refinement for binary masks and structured warning events.
```

Do not claim:

```text
State of the art.
Fully supervised new model.
Complete StreetHazards benchmark result.
Official benchmark submission.
```

## Comparison Position

| Paper / System | What it does | Our relation |
|---|---|---|
| SegmentMeIfYouCan | Benchmark for road anomaly segmentation | Main benchmark motivation and SMIYC/RoadAnomaly source |
| DiCNet | Strong road anomaly method with semantic/anchor reasoning | Our method is much lighter and training-free; compare conceptually, not claim better |
| PEBAL | Supervised anomaly segmentation with energy-biased learning | Our method avoids training but is weaker; cite as strong supervised direction |
| DINOv2 | Self-supervised foundation visual features | Used as feature backbone for road-prototype heatmaps |
| StreetHazards | Synthetic driving OOD benchmark | Used only as partial difficult-domain supplement |

## Current Evidence

| Dataset | Samples evaluated | Best useful result | Interpretation |
|---|---:|---|---|
| SMIYC RoadObstacle | 30 | DINO+ERAS light AP 0.5271, F1 0.5232, FPR95 0.0256 | Main result |
| RoadAnomaly21 | 10 | DINO+ERAS balanced F1 0.4782, IoU 0.3289, FPR95 0.5592 | Cross-dataset support |
| StreetHazards partial | 149 available pairs | Raw DINO AP 0.1133/F1 0.1588; DINO+ERAS light FPR95 0.2688 | Difficult-domain supplement; ERAS helps false-positive control but not AP/F1 |

## Still Needed Before Submission

1. Fill real author names, affiliations, emails, acknowledgements, and funding.
2. Check final page limit and figure/table placement after author metadata is filled.
3. Create a clean GitHub repository with code, README, requirements, paper draft, figures, result summaries, `.gitignore`, and `REPRODUCE.md`.
4. Optional but useful: finish the full StreetHazards download and report the complete benchmark split separately.

## Latest Reproducibility Fix

```text
Scripts no longer hard-code C:\Users\93785\Desktop\CIVS.
The project root is inferred from the script location.
requirements.txt now includes scipy.
Portable smoke test passed on 2026-07-11.
Official Springer LaTeX2e template downloaded and compiled.
Current compiled draft: paper/RaOD-ERAS_CCIS_draft.pdf
Clean release zip generated: dist/raod_eras_release.zip
Automatic PDF build script verified: scripts/build_ccis_pdf.py
Author metadata replacement script verified with placeholder metadata: scripts/set_paper_metadata.py
PDF visual render check passed after fixing keyword wrap and float placement.
CCIS submission package script added: scripts/package_submission.py
Final submission checker added: scripts/final_submission_check.py
Current final check status: only author metadata replacement fails; PDF/page count/submission zip/release zip pass.
Submission form fields prepared: paper/submission_form_fields.md
Data/code/ethics/conflict/funding/limitation statements prepared: paper/statements.md
Author metadata JSON workflow verified: paper/author_metadata_template.json -> set_paper_metadata.py -> build_ccis_pdf.py.
One-command final build wrapper added and tested: scripts/prepare_final_submission.py.
Chinese final operation guide added: paper/最后提交怎么做.md
Author metadata examples added: paper/author_metadata_examples.md
```
