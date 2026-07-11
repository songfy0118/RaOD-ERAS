from __future__ import annotations

import argparse
import fnmatch
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    ".idea",
    ".vscode",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "data",
    "external",
    "outputs",
}

EXCLUDE_PATTERNS = [
    "*.pyc",
    "*.pyo",
    "*.pth",
    "*.pt",
    "*.tar",
    "*.zip",
    "paper/template/*",
    "paper/ccis_build/*.aux",
    "paper/ccis_build/*.blg",
    "paper/ccis_build/*.log",
    "paper/ccis_build/*.out",
    "paper/figures/publication_panels/*",
    "paper/figures/publication_panels/**/*",
]


def should_skip(path: Path) -> bool:
    rel_path = path.relative_to(ROOT)
    rel = rel_path.as_posix()
    if any(part in EXCLUDE_DIRS for part in rel_path.parts):
        return True
    return any(fnmatch.fnmatch(rel, pattern) for pattern in EXCLUDE_PATTERNS)


def iter_release_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if path.is_file() and not should_skip(path):
            files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def write_manifest(files: list[Path], manifest_path: Path) -> None:
    lines = ["# Release Manifest", ""]
    lines.extend(f"- `{path.relative_to(ROOT).as_posix()}`" for path in files)
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Package a clean RaOD-ERAS release zip.")
    parser.add_argument("--out", type=Path, default=ROOT / "dist" / "raod_eras_release.zip")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    files = iter_release_files()
    manifest_path = ROOT / "release_manifest.md"
    write_manifest(files, manifest_path)
    files = iter_release_files()

    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(ROOT).as_posix())

    print(f"Wrote {args.out}")
    print(f"Files: {len(files)}")


if __name__ == "__main__":
    main()
