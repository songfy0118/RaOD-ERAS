from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
DIST = ROOT / "dist"
PDF = PAPER / "RaOD-ERAS_CCIS_draft.pdf"
TEX = PAPER / "paper_ccis_latex.tex"
SUBMISSION_ZIP = DIST / "raod_eras_ccis_submission_package.zip"
RELEASE_ZIP = DIST / "raod_eras_release.zip"
METADATA_TEMPLATE = PAPER / "author_metadata_template.json"

PLACEHOLDERS = [
    "First Author",
    "Second Author",
    "Corresponding Author",
    "Institution Name",
    "author@example.com",
]

BAD_RELEASE_PREFIXES = ("data/", "external/", "outputs/")
BAD_RELEASE_PARTS = ("heatmaps/", "binary_masks/")
BAD_RELEASE_SUFFIXES = (".pth", ".pt", ".tar", ".zip")

EXPECTED_SUBMISSION = {
    "RaOD-ERAS_CCIS_draft.pdf",
    "README_SUBMISSION.txt",
    "figures/framework_pipeline.png",
    "figures/main_qualitative_figure.png",
    "figures/warning_event_example.png",
    "llncs.cls",
    "main.tex",
    "references.bib",
    "splncs04.bst",
    "tables/ablation_objective.md",
    "tables/quantitative_digest.md",
    "current_research_status.md",
}


def pdf_pages() -> int | None:
    pdfinfo = Path(r"C:\Users\93785\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdfinfo.exe")
    if not pdfinfo.exists() or not PDF.exists():
        return None
    result = subprocess.run([str(pdfinfo), str(PDF)], check=True, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    return None


def zip_names(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with zipfile.ZipFile(path) as zf:
        return set(zf.namelist())


def main() -> None:
    checks: list[dict[str, object]] = []

    tex = TEX.read_text(encoding="utf-8") if TEX.exists() else ""
    metadata = json.loads(METADATA_TEMPLATE.read_text(encoding="utf-8")) if METADATA_TEMPLATE.exists() else {}
    found_placeholders = [item for item in PLACEHOLDERS if item in tex or item in json.dumps(metadata)]
    checks.append(
        {
            "name": "author_metadata_replaced",
            "pass": not found_placeholders,
            "detail": "placeholder metadata remains: " + ", ".join(found_placeholders) if found_placeholders else "ok",
        }
    )

    pages = pdf_pages()
    checks.append(
        {
            "name": "pdf_exists_and_page_count",
            "pass": PDF.exists() and pages is not None and pages <= 6,
            "detail": f"pages={pages}, path={PDF}",
        }
    )

    submission_names = zip_names(SUBMISSION_ZIP)
    missing_submission = sorted(EXPECTED_SUBMISSION - submission_names)
    extra_submission = sorted(submission_names - EXPECTED_SUBMISSION)
    checks.append(
        {
            "name": "submission_zip_contents",
            "pass": SUBMISSION_ZIP.exists() and not missing_submission and not extra_submission,
            "detail": f"missing={missing_submission}; extra={extra_submission}",
        }
    )

    release_names = zip_names(RELEASE_ZIP)
    bad_release = sorted(
        name
        for name in release_names
        if name.startswith(BAD_RELEASE_PREFIXES)
        or any(part in name for part in BAD_RELEASE_PARTS)
        or name.endswith(BAD_RELEASE_SUFFIXES)
    )
    checks.append(
        {
            "name": "release_zip_excludes_large_artifacts",
            "pass": RELEASE_ZIP.exists() and not bad_release,
            "detail": f"bad_entries={bad_release[:20]}",
        }
    )

    all_passed = all(bool(item["pass"]) for item in checks)
    print(json.dumps({"pass": all_passed, "checks": checks}, indent=2, ensure_ascii=False))
    raise SystemExit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
