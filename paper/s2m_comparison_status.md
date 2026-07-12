# S2M Comparison Status

## Reference

Zhao et al., "Segment Every Out-of-Distribution Object," CVPR 2024. Official paper and code:

- https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_Segment_Every_Out-of-Distribution_Object_CVPR_2024_paper.pdf
- https://github.com/WenjieZhao1/S2M

S2M converts anomaly scores into box prompts and uses a promptable segmentation model to obtain complete OoD object masks. It is therefore the closest published basis for the mask stage of this project.

## Current Reproduction Boundary

The official repository and SAM-B checkpoint are available locally. The author's Google Drive detector checkpoint could not yet be downloaded, and the official code depends on an older Detectron2 environment. Current results compare two implementations under identical inputs:

1. `s2m_style`: fixed score threshold, connected-component boxes, SAM-B mask selection by SAM confidence.
2. `risk_s2m`: road-aware adaptive seeds, risk-ranked boxes, local-peak fallback, SAM multimask selection by SAM confidence plus heatmap boundary contrast.

Both methods use the same DINOv2 risk heatmap, images, SAM-B checkpoint, GT masks, and evaluator. GT is used only for final evaluation. These numbers must be labeled **S2M-style**, not official S2M results.

## Disjoint 20-Sample Validation

| Dataset | Method | Precision | Recall | F1 | IoU |
|---|---|---:|---:|---:|---:|
| RoadAnomaly21 | S2M-style | 0.4331 | 0.1595 | 0.2332 | 0.1320 |
| RoadAnomaly21 | Risk-S2M | **0.6174** | **0.1805** | **0.2793** | **0.1623** |
| SMIYC | S2M-style | 0.0191 | **0.8393** | 0.0374 | 0.0191 |
| SMIYC | Risk-S2M | **0.0306** | 0.8377 | **0.0590** | **0.0304** |
| StreetHazards | S2M-style | 0.0010 | 0.0028 | 0.0014 | 0.0007 |
| StreetHazards | Risk-S2M | **0.0340** | **0.2251** | **0.0591** | **0.0305** |
| All 20 | S2M-style | 0.1930 | 0.1635 | 0.1770 | 0.0971 |
| All 20 | Risk-S2M | **0.2787** | **0.1878** | **0.2244** | **0.1264** |

The development result supports the risk-aware prompting contribution on a disjoint subset. It does not yet establish superiority over the official S2M model or published state of the art.

## Next Required Experiment

## Full 189-Pair Result

| Dataset | Method | Precision | Recall | F1 | IoU |
|---|---|---:|---:|---:|---:|
| RoadAnomaly21 (10) | S2M-style | 0.2738 | 0.1944 | 0.2274 | 0.1283 |
| RoadAnomaly21 (10) | Risk-S2M | **0.6520** | **0.2196** | **0.3285** | **0.1965** |
| SMIYC (30) | S2M-style | 0.0759 | 0.9023 | 0.1400 | 0.0753 |
| SMIYC (30) | Risk-S2M | **0.0772** | **0.9037** | **0.1423** | **0.0766** |
| StreetHazards (149) | S2M-style | 0.0030 | 0.0034 | 0.0032 | 0.0016 |
| StreetHazards (149) | Risk-S2M | **0.0550** | **0.1004** | **0.0711** | **0.0369** |
| All 189 | S2M-style | 0.0865 | 0.1223 | 0.1013 | 0.0534 |
| All 189 | Risk-S2M | **0.1151** | **0.1848** | **0.1418** | **0.0763** |

The full same-input comparison preserves the improvement on every source dataset. Total runtime was approximately 321 seconds on an RTX 4060 Laptop GPU. The large StreetHazards gain comes from the road-weighted local-peak fallback when a coarse score region is too large to form a valid box prompt.

## Remaining Required Experiment

1. Report component F1, boundary F1, and per-image confidence intervals from the saved full results; do not report AP/FPR95 for binary masks.
2. Obtain and run the official S2M detector checkpoint or use official published values only in a separately labeled literature table.
3. Add DaCUP-DINOv2 as the strong heatmap front end when its official checkpoints are available.
