# GitHub Release Plan

## Release structure

Keep this repository lightweight. Do not upload raw datasets or dense experiment outputs.

Recommended public files:

- `src/raod_eras/`
- `scripts/`
- `configs/`
- `README.md`
- `REPRODUCE.md`
- `requirements.txt`
- `paper/paper_ccis_latex.tex`
- `paper/references.bib`
- `paper/tables/quantitative_digest.md`
- `paper/figures/framework_pipeline.png`
- `paper/figures/main_qualitative_figure.png`
- `paper/figures/warning_event_example.png`

Do not upload:

- `data/`
- `outputs/`
- downloaded model weights,
- large archives,
- full per-sample heatmap folders unless using Git LFS.

## GitHub description

Training-free road anomaly heatmap and warning-event generation with DINOv2 road-prototype scoring and ego-lane risk-aware refinement.

## Suggested repository name

`raod-eras-road-anomaly`

## Before pushing

Run:

```powershell
python scripts\package_release.py
python scripts\final_submission_check.py
```

Then upload either:

- the cleaned repository folder, or
- `dist/raod_eras_release.zip`.

## Missing user decision

I cannot push to GitHub until the target GitHub account/repository is known.

