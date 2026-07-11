# Release Checklist

## Ready

```text
Code is portable: scripts infer project root from their own location.
Requirements include scipy for fast connected components.
README explains method, datasets, commands, outputs, and claim boundary.
REPRODUCE.md gives smoke test and full experiment commands.
Paper draft, CCIS LaTeX draft, references, figures, and result summaries exist.
SMIYC, RoadAnomaly21, and StreetHazards partial 149 have been run.
Official Springer LaTeX template has been downloaded.
A 5-page CCIS draft PDF has been compiled.
build_ccis_pdf.py compiles the paper automatically.
set_paper_metadata.py updates placeholder authors/institute/email.
PDF render check passed: 5 pages, readable tables and figures.
package_submission.py creates a CCIS submission zip.
final_submission_check.py verifies author placeholders, PDF pages, submission zip contents, and release zip exclusions.
```

## Do Not Upload

```text
data/
external/
large outputs/heatmaps/
large outputs/binary_masks/
DINOv2 weights or torch cache
partial StreetHazards tar/zip files
```

## Still Needed for Actual Submission

```text
Fill real author names, affiliations, emails, acknowledgements, and funding.
Check page limit and figure/table placement.
Replace placeholder author metadata.
Export the final camera-ready PDF after author metadata is filled.
```

## Current Compiled Draft

```text
paper/RaOD-ERAS_CCIS_draft.pdf
paper/ccis_build/main.tex
paper/ccis_build/main.pdf
dist/raod_eras_ccis_submission_package.zip
paper/final_submission_checklist.md
paper/author_metadata_template.json
```
