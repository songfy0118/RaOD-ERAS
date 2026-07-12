from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_research_experiment import dataset_config
from src.raod_eras.datasets import SMIYCRoadObstacleDataset


def source_of(sample_id: str) -> str:
    return sample_id.split("__", maxsplit=1)[0]


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the main RiskPrompt-SAM qualitative figure.")
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, default=Path("paper/figures/riskprompt_qualitative.png"))
    args = parser.parse_args()

    result = json.loads(args.results.read_text(encoding="utf-8"))
    rows = result["per_image"]
    selected: list[dict[str, object]] = []
    for source in ("road_anomaly", "smiyc", "street_hazards"):
        source_rows = [row for row in rows if source_of(str(row["id"])) == source]
        selected.append(max(source_rows, key=lambda row: float(row["riskprompt_fixed_f1"]) - float(row["s2m_style_fixed_f1"])))
    smiyc_rows = [row for row in rows if source_of(str(row["id"])) == "smiyc"]
    selected.append(min(smiyc_rows, key=lambda row: float(row["riskprompt_fixed_f1"]) - float(row["s2m_style_fixed_f1"])))

    dataset = SMIYCRoadObstacleDataset(dataset_config("unified"))
    samples = {sample.sample_id: sample for sample in dataset.list_samples()}
    experiment_dir = args.results.parent
    columns = ("Input", "Ground truth", "Base score", "S2M-style", "UGainS-style score", "RiskPrompt score", "RiskPrompt mask")
    fig, axes = plt.subplots(len(selected), len(columns), figsize=(15.4, 8.8), constrained_layout=True)
    for row_index, row in enumerate(selected):
        sample_id = str(row["id"])
        loaded = dataset.load(samples[sample_id])
        cache = np.load(experiment_dir / "cache" / f"{sample_id}.npz")
        panels = (
            np.asarray(loaded.image.resize((loaded.gt.shape[1], loaded.gt.shape[0]), Image.Resampling.BILINEAR)),
            loaded.gt,
            cache["score"],
            load_mask(experiment_dir / "masks" / "s2m_style" / f"{sample_id}.png"),
            cache["ugains_score"],
            cache["riskprompt_score"],
            load_mask(experiment_dir / "masks" / "riskprompt" / f"{sample_id}.png"),
        )
        for column_index, panel in enumerate(panels):
            axis = axes[row_index, column_index]
            if column_index == 0:
                axis.imshow(panel)
            elif column_index in (2, 4, 5):
                axis.imshow(panel, cmap="magma", vmin=0, vmax=1)
            else:
                axis.imshow(panel, cmap="gray", vmin=0, vmax=1)
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(columns[column_index], fontsize=10)
            if column_index == 0:
                delta = float(row["riskprompt_fixed_f1"]) - float(row["s2m_style_fixed_f1"])
                axis.set_ylabel(f"{source_of(sample_id)}\nDelta F1={delta:+.3f}", fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    selection = [
        {
            "id": row["id"],
            "s2m_f1": row["s2m_style_fixed_f1"],
            "riskprompt_f1": row["riskprompt_fixed_f1"],
        }
        for row in selected
    ]
    args.output.with_suffix(".json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
