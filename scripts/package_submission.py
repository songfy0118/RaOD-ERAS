from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
BUILD = PAPER / "ccis_build"
OUT_DIR = ROOT / "dist"
PACKAGE_DIR = OUT_DIR / "ccis_submission_package"


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    if PACKAGE_DIR.exists():
        shutil.rmtree(PACKAGE_DIR)
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)

    files = {
        PAPER / "RaOD-ERAS_CCIS_draft.pdf": PACKAGE_DIR / "RaOD-ERAS_CCIS_draft.pdf",
        BUILD / "main.tex": PACKAGE_DIR / "main.tex",
        BUILD / "references.bib": PACKAGE_DIR / "references.bib",
        BUILD / "llncs.cls": PACKAGE_DIR / "llncs.cls",
        BUILD / "splncs04.bst": PACKAGE_DIR / "splncs04.bst",
        BUILD / "figures" / "framework_pipeline.png": PACKAGE_DIR / "figures" / "framework_pipeline.png",
        BUILD / "figures" / "warning_event_example.png": PACKAGE_DIR / "figures" / "warning_event_example.png",
        PAPER / "figures" / "main_qualitative_figure.png": PACKAGE_DIR / "figures" / "main_qualitative_figure.png",
        PAPER / "tables" / "quantitative_digest.md": PACKAGE_DIR / "tables" / "quantitative_digest.md",
        PAPER / "tables" / "ablation_objective.md": PACKAGE_DIR / "tables" / "ablation_objective.md",
        PAPER / "current_research_status.md": PACKAGE_DIR / "current_research_status.md",
    }
    for src, dst in files.items():
        copy_file(src, dst)

    readme = PACKAGE_DIR / "README_SUBMISSION.txt"
    readme.write_text(
        "\n".join(
            [
                "RaOD-ERAS CCIS submission package",
                "",
                "Compile with:",
                "pdflatex -interaction=nonstopmode main.tex",
                "bibtex main",
                "pdflatex -interaction=nonstopmode main.tex",
                "pdflatex -interaction=nonstopmode main.tex",
                "",
                "Before final submission, replace placeholder author names, affiliation, and email.",
                "",
                "Additional qualitative and ablation materials are included under figures/ and tables/.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    zip_path = OUT_DIR / "raod_eras_ccis_submission_package.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(PACKAGE_DIR.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(PACKAGE_DIR).as_posix())
    print(zip_path)


if __name__ == "__main__":
    main()
