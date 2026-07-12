from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, obj: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def save_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_heatmap(path: Path, score: np.ndarray, cmap: str = "RdYlBu_r") -> None:
    ensure_dir(path.parent)
    plt.imsave(path, score, cmap=cmap, vmin=0, vmax=1)


def save_binary(path: Path, score: np.ndarray, threshold: float) -> None:
    ensure_dir(path.parent)
    Image.fromarray(((score >= threshold) * 255).astype(np.uint8)).save(path)


def save_mask(path: Path, mask: np.ndarray) -> None:
    ensure_dir(path.parent)
    Image.fromarray((mask.astype(bool) * 255).astype(np.uint8)).save(path)
