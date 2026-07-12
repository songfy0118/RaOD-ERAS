from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "unified_road_anomaly_eval"
OUTPUT = ROOT / "dist" / "unified_road_anomaly_eval_189.zip"


def main() -> None:
    required = ("images", "gt_labels", "metadata")
    for name in required:
        if not (DATASET_ROOT / name).is_dir():
            raise FileNotFoundError(f"Missing dataset directory: {DATASET_ROOT / name}")

    files = sorted(
        path
        for name in required
        for path in (DATASET_ROOT / name).rglob("*")
        if path.is_file()
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.relative_to(DATASET_ROOT).as_posix())

    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    manifest = {
        "archive": OUTPUT.name,
        "num_files": len(files),
        "size_bytes": OUTPUT.stat().st_size,
        "sha256": digest,
        "label_encoding": {"normal": 0, "anomaly": 1, "ignore": 255},
    }
    manifest_path = OUTPUT.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
