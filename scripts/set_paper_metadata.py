from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper" / "paper_ccis_latex.tex"
DEFAULT_METADATA = ROOT / "paper" / "author_metadata_template.json"


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(ch, ch) for ch in value)


def load_metadata(path: Path) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8"))


def replace_marked_block(text: str, start: str, end: str, value: str) -> str:
    if start not in text or end not in text:
        raise ValueError(f"Missing markers: {start} / {end}")
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    return before + value + after


def main() -> None:
    parser = argparse.ArgumentParser(description="Set author metadata in paper_ccis_latex.tex.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA, help="JSON metadata file.")
    parser.add_argument("--authors", default=None, help=r"Example: Alice Zhang\inst{1} \and Bob Li\inst{1}")
    parser.add_argument("--authorrunning", default=None, help="Example: A. Zhang et al.")
    parser.add_argument("--institute", default=None, help=r"Example: School, University, City, Country")
    parser.add_argument("--email", default=None, help="Example: author@example.com")
    parser.add_argument("--acknowledgements", default=None, help="Optional acknowledgements/funding text.")
    args = parser.parse_args()

    metadata = load_metadata(args.metadata) if args.metadata.exists() else {}
    authors = args.authors or metadata.get("authors_latex")
    authorrunning = args.authorrunning or metadata.get("authorrunning")
    institute = args.institute or metadata.get("institute")
    email = args.email or metadata.get("email")
    acknowledgements = args.acknowledgements if args.acknowledgements is not None else metadata.get("acknowledgements", "")
    missing = [
        name
        for name, value in {
            "authors_latex": authors,
            "authorrunning": authorrunning,
            "institute": institute,
            "email": email,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"Missing required metadata fields: {', '.join(missing)}")

    text = TEX.read_text(encoding="utf-8")
    start = "% METADATA_START"
    end = "% METADATA_END"
    metadata = "\n".join(
        [
            start,
            f"\\author{{{authors}}}",
            f"\\authorrunning{{{latex_escape(authorrunning)}}}",
            f"\\institute{{{latex_escape(institute)}\\\\",
            f"\\email{{{latex_escape(email)}}}}}",
            end,
        ]
    )
    text = replace_marked_block(text, start, end, metadata)

    ack_start = "% ACKNOWLEDGEMENTS_START"
    ack_end = "% ACKNOWLEDGEMENTS_END"
    ack_body = ""
    if acknowledgements.strip():
        ack_body = "\n".join(
            [
                ack_start,
                "\\subsubsection*{Acknowledgements}",
                latex_escape(acknowledgements.strip()),
                ack_end,
            ]
        )
    else:
        ack_body = "\n".join([ack_start, ack_end])
    text = replace_marked_block(text, ack_start, ack_end, ack_body)
    TEX.write_text(text, encoding="utf-8")
    print(TEX)


if __name__ == "__main__":
    main()
