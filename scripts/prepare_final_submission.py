from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = ROOT / "paper" / "author_metadata_template.json"
PLACEHOLDERS = {
    "First Author",
    "Second Author",
    "Corresponding Author",
    "Institution Name",
    "author@example.com",
}


def run(command: list[str], allow_failure: bool = False) -> int:
    print("+ " + " ".join(command))
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode and not allow_failure:
        raise SystemExit(result.returncode)
    return result.returncode


def metadata_has_placeholders(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    text = json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False)
    return sorted(item for item in PLACEHOLDERS if item in text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare final RaOD-ERAS submission artifacts.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="Build draft artifacts even if author metadata still contains placeholders.",
    )
    args = parser.parse_args()

    placeholders = metadata_has_placeholders(args.metadata)
    if placeholders and not args.allow_placeholders:
        print("Author metadata still contains placeholders:")
        for item in placeholders:
            print(f"- {item}")
        print("\nEdit paper/author_metadata_template.json, then rerun this command.")
        raise SystemExit(2)

    run([sys.executable, "scripts/set_paper_metadata.py", "--metadata", str(args.metadata)])
    run([sys.executable, "scripts/build_ccis_pdf.py"])
    run([sys.executable, "scripts/package_submission.py"])
    run([sys.executable, "scripts/package_release.py"])
    code = run([sys.executable, "scripts/final_submission_check.py"], allow_failure=True)
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
