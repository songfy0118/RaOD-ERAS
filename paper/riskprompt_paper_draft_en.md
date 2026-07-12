# RiskPrompt-SAM: Road-Aware Prompt Selection and Mask Feedback for Training-Free Unexpected Obstacle Segmentation

## Abstract

Unexpected road obstacles are difficult to enumerate during training but must be localized as complete objects before an autonomous-driving system can issue a useful warning. Pixel-wise anomaly maps provide a useful signal, yet fixed thresholding often produces fragmented masks and road-texture false positives. Promptable segmentation models can recover object boundaries, but their performance depends strongly on how anomaly scores are converted into prompts and how candidate masks are selected. We present RiskPrompt-SAM, a training-free score-to-mask framework that keeps DINOv2 and SAM-B frozen. A multi-scale road-prototype anomaly map is converted into adaptive road-aware box prompts. For each prompt, the method chooses among SAM masks using both SAM confidence and anomaly contrast across the candidate boundary. Accepted instance masks are fed back into the continuous anomaly map to improve anomaly ranking. We conduct a controlled comparison in which thresholding, S2M-style box prompting, UGainS-style farthest-point prompting, and RiskPrompt-SAM share the same DINOv2 score map and SAM-B. The evaluation contains 189 public image/ground-truth pairs from partial public subsets of RoadAnomaly, SMIYC RoadObstacle, and StreetHazards, with source void pixels excluded. RiskPrompt-SAM improves pixel-micro F1 from 0.1252 to 0.1715 and IoU from 0.0668 to 0.0938 over the S2M-style baseline, while improving AP from 0.0492 to 0.0805. Paired image-level bootstrap intervals confirm positive overall differences in F1 and IoU. On SMIYC, anomaly ranking improves but binary F1 is slightly lower than S2M-style, exposing a trade-off of fixed geometric road priors.

## 1. Introduction

Autonomous vehicles must respond to objects that are rare or absent from the semantic training taxonomy, including animals, debris, temporary barriers, and unusual cargo. Road anomaly segmentation addresses this open-world failure mode by assigning an anomaly score to each image pixel. The score map is useful for ranking pixels, but it is not yet an object representation. A single threshold can split one obstacle into fragments, merge nearby anomalies, or activate broad road-texture regions.

Recent work has therefore shifted from pixel-only scoring toward mask-level reasoning. UGainS samples uncertain pixels and uses the Segment Anything Model (SAM) to obtain anomaly instances. S2M learns a detector that converts anomaly scores to box prompts and lets SAM recover complete objects. Maskomaly and Mask2Anomaly likewise show that mask-level representations improve contextual and boundary reasoning. These methods establish the value of object masks, but two practical questions remain important for a rapid, training-free deployment: where should prompts be placed when no prompt detector can be trained, and which SAM candidate should be trusted when several masks fit the same prompt?

RiskPrompt-SAM addresses these questions with a deterministic road-aware prompt generator and anomaly-boundary mask selection. The method is deliberately lightweight: it does not train a new detector, tune DINOv2, or fine-tune SAM. This makes it appropriate for a small conference study focused on prompt design rather than a new anomaly backbone.

The contributions are:

1. A road-aware adaptive score-to-box generator that ranks connected anomaly regions by anomaly strength, drivable-road overlap, ego-lane overlap, and near-field relevance, with local-peak fallback for missed small objects.
2. A multi-mask selection rule that combines SAM confidence with anomaly contrast between a mask interior and a narrow exterior ring, followed by instance-to-score feedback.
3. A controlled, reproducible comparison against thresholding, S2M-style box prompting, and UGainS-style farthest-point prompting using one shared DINOv2/SAM-B front end on 189 public image/GT pairs.

The claim boundary is important. We compare prompt mechanisms under a shared weak front end; we do not claim to outperform the official end-to-end S2M or UGainS systems, which use trained RPL/RbA-based anomaly models and different benchmark protocols.

## 2. Related Work

### 2.1 Road anomaly heatmaps

DaCUP models anomalous regions as image-consistent yet unpredictable patches and uses image-conditioned distances and inpainting to reduce false positives. Maskomaly obtains zero-shot anomaly maps from raw mask predictions of a pretrained semantic model. Mask2Anomaly introduces global masked attention, mask contrastive learning, and mask refinement. These approaches use stronger road-semantic training than our frozen DINOv2 prototype and therefore provide literature context rather than numerically identical baselines.

### 2.2 Score-to-mask anomaly segmentation

UGainS combines RbA uncertainty with farthest-point sampling (FPS) and SAM. It reports 88.98% pixel AP on RoadAnomaly with outlier-exposure training and demonstrates that instance masks can refine a continuous uncertainty map. S2M trains a Faster R-CNN prompt generator on anomaly scores and uses SAM-B to segment complete objects. Its official paper reports 58.49% IoU and 61.66% mean F1 on RoadAnomaly under its trained RPL+CoroCL protocol. These absolute values are not directly comparable to ours; they motivate the two controlled mechanisms reproduced in this work: FPS point prompts and score-derived box prompts.

## 3. Method

### 3.1 Shared anomaly front end

Given an RGB image $I$, frozen DINOv2 ViT-S/14 produces patch features $F$. A road prototype $p_r$ is estimated from lower-image road patches. The base anomaly value of patch $i$ is the normalized feature distance

$$a_i = \operatorname{norm}(1-\cos(F_i,p_r)).$$

We average single-scale and multi-scale prototype distances with a lightweight local road-contrast map $c$ and spatial priors for the road $R$, ego lane $L$, and near field $N$:

$$A = \operatorname{norm}\left(0.8 A_{dino} + 0.2\operatorname{norm}[(0.78A_{ms}+0.22c)(0.62+0.20R+0.18LN)]\right).$$

All four compared methods receive exactly this $A$.

### 3.2 Controlled baselines

The threshold baseline predicts $M_t=[A\geq0.70]$ and removes tiny connected components. The S2M-style baseline converts connected high-score regions into boxes and uses the highest-confidence SAM mask for each box. It reproduces the score-box-SAM mechanism but not S2M's trained Faster R-CNN. The UGainS-style baseline selects pixels above 0.70, performs deterministic FPS for 50 points, prompts SAM, ranks masks by mean anomaly score, applies mask NMS at IoU 0.70, feeds retained masks back into the score map, and uses a 0.60 output threshold fixed on the 10-image calibration split.

### 3.3 Road-aware adaptive prompts

RiskPrompt-SAM computes an adaptive seed threshold from the 98.5th percentile inside the road prior. Connected regions smaller than $5\times10^{-5}$ or larger than 0.12 of the image are rejected. Candidate $k$ is ranked by

$$q_k=\bar A_k(0.70+0.20\bar R_k+0.10\bar L_k).$$

Regions with insufficient road overlap are removed. If fewer than four prompts survive, non-maximum local peaks of $A(0.35+0.65R)$ produce fallback boxes. At most 12 boxes are retained.

### 3.4 Boundary-consistent mask selection

SAM-B returns three candidate masks for each risk box. For candidate $m$, let $s_{sam}$ be SAM confidence, $\mu_{in}$ the mean anomaly inside $m$, and $\mu_{ring}$ the mean anomaly in a three-pixel exterior ring. We select

$$m^*=\arg\max_m [0.55s_{sam}+0.45(\mu_{in}-\mu_{ring})].$$

This rule rejects geometrically plausible masks whose interiors are not more anomalous than their immediate surroundings. Mask NMS removes duplicate instances.

### 3.5 Mask-to-score feedback and warning output

Each retained instance contributes a confidence-weighted support map. The refined score suppresses unsupported responses and reinforces accepted masks:

$$A' = \operatorname{norm}(A\cdot w_s + 0.35F_m),$$

where $w_s=1$ inside supported masks and 0.82 elsewhere. The system outputs $A'$, the binary union of selected masks, instance bounding boxes, and image-plane warning attributes. Trajectory suggestions are qualitative only because the benchmarks do not provide depth, motion, TTC, or control ground truth.

## 4. Experiments

### 4.1 Data and protocol

We use all locally available public image/GT pairs: 10 from RoadAnomaly, 30 from SMIYC RoadObstacle, and 149 from StreetHazards, for 189 total. These are partial public validation subsets, not the complete private benchmark test sets, and their union is an evaluation manifest rather than a new dataset. Labels are standardized as 0 normal, 1 anomaly, and 255 ignore. StreetHazards anomaly class 14 is converted to label 1; RoadAnomaly and SMIYC retain their provided void regions as label 255 and those pixels are excluded from every metric.

Ten deterministic stratified samples form the calibration split. Twenty samples starting at offset 10 form a disjoint validation split. The final 189-image run uses frozen parameters. No GT is read by the inference functions. DINOv2 ViT-S/14 and SAM ViT-B are frozen. Experiments run on an NVIDIA RTX 4060 Laptop GPU. Median end-to-end time is 8.70 s/image and the 90th percentile is 14.99 s/image in the unoptimized Python implementation.

### 4.2 Metrics

Continuous heatmaps are evaluated by pixel average precision (AP) and FPR at 95% true-positive rate (FPR95). Binary masks are evaluated by precision, recall, F1, IoU, component F1, and boundary F1. Dataset tables use pixel-micro aggregation. Stability is assessed with 5,000 paired image-level bootstrap resamples and a fixed seed of 42.

### 4.3 Main results

| Method | Precision | Recall | F1 | IoU | AP | FPR95 |
|---|---:|---:|---:|---:|---:|---:|
| Threshold | 0.0312 | **0.5568** | 0.0590 | 0.0304 | 0.0492 | 0.8609 |
| S2M-style | 0.1282 | 0.1223 | 0.1252 | 0.0668 | 0.0492 | 0.8609 |
| UGainS-style | 0.0190 | 0.2889 | 0.0356 | 0.0181 | 0.0458 | 0.8657 |
| **RiskPrompt-SAM** | **0.1601** | 0.1846 | **0.1715** | **0.0938** | **0.0805** | **0.8551** |

RiskPrompt-SAM improves F1 by 0.0463 and IoU by 0.0270 over S2M-style in pixel-micro aggregation. Relative to the base score, mask feedback improves AP from 0.0492 to 0.0805 and slightly reduces FPR95. The paired macro difference against S2M-style is +0.0624 F1 (95% CI [0.0494, 0.0758]) and +0.0365 IoU ([0.0264, 0.0479]). Boundary F1 improves by +0.0664 ([0.0532, 0.0787]), supporting the claim that the method improves object boundaries rather than only pixel counts.

### 4.4 Per-source results

| Source | Method | F1 | IoU | AP | FPR95 |
|---|---|---:|---:|---:|---:|
| RoadAnomaly | S2M-style | 0.2277 | 0.1285 | 0.1110 | **0.9815** |
| RoadAnomaly | **RiskPrompt-SAM** | **0.3286** | **0.1966** | **0.2688** | 0.9826 |
| SMIYC | **S2M-style** | **0.9349** | **0.8777** | 0.6894 | **0.0135** |
| SMIYC | RiskPrompt-SAM | 0.9281 | 0.8658 | **0.9159** | 0.0157 |
| StreetHazards | S2M-style | 0.0032 | 0.0016 | **0.0646** | 0.5630 |
| StreetHazards | **RiskPrompt-SAM** | **0.0710** | **0.0368** | 0.0517 | **0.5409** |

RoadAnomaly shows the strongest gains. StreetHazards obtains markedly better binary segmentation and FPR95 but lower AP than the unmodified base map. On SMIYC, RiskPrompt raises AP from 0.6894 to 0.9159 but reduces F1 from 0.9349 to 0.9281 and IoU from 0.8777 to 0.8658 relative to S2M-style. The paired image-macro differences confirm that this small segmentation reduction is systematic. Small obstacles and scenes whose road geometry violates the fixed trapezoid prior remain difficult.

### 4.5 Mechanism ablation

The ordered comparison provides the following mechanism ablation. Thresholding tests the raw score only. S2M-style adds object completion through box-prompted SAM. UGainS-style replaces boxes with FPS points and mask feedback. RiskPrompt-SAM adds road-aware adaptive boxes, peak fallback, anomaly-boundary candidate selection, and feedback. A rejected hybrid variant directly unioned 50 point-prompt masks with box masks; on the 10-image calibration set it produced excessive recall and lower F1 than box prompting. This negative result motivates using points as a separate baseline rather than unconditionally merging all proposals.

## 5. Discussion and Limitations

The experiment supports a narrow conclusion: with a common training-free DINOv2 front end and common SAM-B, road-aware box prompting and boundary-consistent selection produce better anomaly masks than fixed thresholding, simple score boxes, or FPS points. It does not show that the complete method exceeds official S2M, UGainS, Maskomaly, or Mask2Anomaly, whose stronger trained front ends achieve much higher absolute scores.

Three limitations matter. First, the road prior is image-plane geometry rather than a learned drivable-area model, so curved roads and unusual camera poses can suppress true anomalies. Second, DINOv2 road prototypes produce weak absolute anomaly ranking on several scenes, especially at the 95% recall operating point. Third, single frames cannot validate metric distance, TTC, safe steering, or closed-loop control. Future work should replace the weak front end with a strong road-semantic anomaly model, estimate uncertainty over time, and evaluate risk decisions in video or simulation.

## 6. Conclusion

RiskPrompt-SAM turns a weak training-free anomaly map into more coherent road-obstacle instances through adaptive road-aware prompts, boundary-consistent SAM mask selection, and mask feedback. Across 189 public image/GT pairs it improves overall segmentation F1, IoU, component F1, boundary F1, and heatmap AP under a controlled shared-front-end protocol. The method is suitable as a concise conference contribution and a reproducible foundation for a later temporal, uncertainty-aware driving-risk study.

## Reproducibility artifacts

- Full result: `outputs/riskprompt_full_189/results.json`
- Statistics: `outputs/riskprompt_full_189/reports/paper_statistics.md`
- Main figure: `paper/figures/riskprompt_qualitative.png`
- Inference: `scripts/run_s2m_comparison.py`
- Statistics: `scripts/analyze_prompt_results.py`
