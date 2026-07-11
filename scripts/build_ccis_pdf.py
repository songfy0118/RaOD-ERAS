from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
BUILD = PAPER / "ccis_build"
FIGURES = BUILD / "figures"
TEMPLATE = PAPER / "template" / "springer_latex2e_proceedings_template"


def copy_required_files() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    required = {
        PAPER / "paper_ccis_latex.tex": BUILD / "main.tex",
        PAPER / "references.bib": BUILD / "references.bib",
        TEMPLATE / "llncs.cls": BUILD / "llncs.cls",
        TEMPLATE / "splncs04.bst": BUILD / "splncs04.bst",
    }
    for src, dst in required.items():
        if not src.exists():
            raise FileNotFoundError(f"Missing required file: {src}")
        shutil.copy2(src, dst)

    for name in ["framework_pipeline.png", "warning_event_example.png"]:
        src = PAPER / "figures" / name
        if not src.exists():
            src = BUILD / "figures" / name
        if not src.exists():
            raise FileNotFoundError(f"Missing figure: {name}")
        dst = FIGURES / name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=BUILD, check=True)


def main() -> None:
    copy_required_files()
    run(["pdflatex", "-interaction=nonstopmode", "main.tex"])
    run(["bibtex", "main"])
    run(["pdflatex", "-interaction=nonstopmode", "main.tex"])
    run(["pdflatex", "-interaction=nonstopmode", "main.tex"])
    shutil.copy2(BUILD / "main.pdf", PAPER / "RaOD-ERAS_CCIS_draft.pdf")
    print(PAPER / "RaOD-ERAS_CCIS_draft.pdf")


if __name__ == "__main__":
    main()
