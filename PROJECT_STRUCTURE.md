# Repository map

| Path | Purpose |
|---|---|
| `scripts/predict_image.py` | Standalone inference without labels |
| `src/raod_eras/inference.py` | Shared inference interface |
| `src/raod_eras/dino_features.py` | Frozen DINOv2 features |
| `src/raod_eras/score_to_mask.py` | SAM prompts, candidate selection, feedback |
| `src/raod_eras/metrics.py` | Binary and continuous evaluation metrics |
| `scripts/run_s2m_comparison.py` | Shared-front-end benchmark cache |
| `scripts/run_prompt_ablation.py` | A–D module ablation |
| `scripts/analyze_ablation_results.py` | Paired statistics and report |
| `scripts/evaluate_smiyc_official.py` | External official evaluation |
| `tests/` | CPU unit and CLI checks |
| `dist/` | LFS archive pointer and dataset manifest |
| `outputs/riskprompt_ablation_full_189_v2/` | Tracked historical summary/report files |
| `outputs/smiyc_official_protocol/` | Tracked official-protocol summaries |
| `paper/` | Manuscript development artifacts |

For the unified benchmark, extract the archive under:

```text
data/unified_road_anomaly_eval/
  images/
  gt_labels/
  metadata/
```

The active loader reads `gt_labels`, where 0 means normal, 1 anomaly and 255 ignore. Other local archive folders are not needed by the quick start. Model weights, image datasets and generated inference outputs are ignored by Git.
