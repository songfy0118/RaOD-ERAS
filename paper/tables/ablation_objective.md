# Ablation and Operating-Point Selection

The validation objective is lower-is-better:

`L = 0.40 * (1 - AP) + 0.35 * (1 - F1) + 0.25 * FPR95`

This is not a neural-network training loss. It is the paper's operating-point selection criterion for a training-free pipeline.

## SMIYC RoadObstacle

| Rank | Method | Objective | AP | F1 | IoU | Recall | FPR95 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | dino_eras_light | 0.3624 | 0.5271 | 0.5232 | 0.3989 | 0.7524 | 0.0256 |
| 2 | dino | 0.3656 | 0.5203 | 0.5228 | 0.3998 | 0.6864 | 0.0268 |
| 3 | dino_eras_balanced | 0.3812 | 0.4975 | 0.5039 | 0.3825 | 0.8062 | 0.0263 |
| 4 | dino_eras_recall | 0.4149 | 0.4520 | 0.4634 | 0.3501 | 0.8548 | 0.0316 |
| 5 | roadcontrast_eras_recall | 0.6799 | 0.2180 | 0.2751 | 0.2068 | 0.3981 | 0.4536 |
| 6 | roadcontrast_eras_balanced | 0.7068 | 0.1852 | 0.2359 | 0.1690 | 0.3796 | 0.4538 |

## RoadAnomaly21

| Rank | Method | Objective | AP | F1 | IoU | Recall | FPR95 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | dino_eras_balanced | 0.5933 | 0.3227 | 0.4782 | 0.3289 | 0.8369 | 0.5592 |
| 2 | roadcontrast_eras_recall | 0.5958 | 0.3730 | 0.4528 | 0.3092 | 0.6279 | 0.6140 |
| 3 | roadcontrast_eras_balanced | 0.5968 | 0.3733 | 0.4534 | 0.3099 | 0.6137 | 0.6191 |
| 4 | roadcontrast_eras_light | 0.5991 | 0.3717 | 0.4529 | 0.3095 | 0.6142 | 0.6252 |
| 5 | dino_eras_recall | 0.6052 | 0.3122 | 0.4741 | 0.3239 | 0.7594 | 0.5844 |
| 6 | roadcontrast | 0.6058 | 0.3698 | 0.4473 | 0.3059 | 0.6188 | 0.6409 |

## StreetHazards partial

| Rank | Method | Objective | AP | F1 | IoU | Recall | FPR95 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | dino | 0.7409 | 0.1133 | 0.1588 | 0.0975 | 0.4781 | 0.3671 |
| 2 | dino_eras_light | 0.7425 | 0.0691 | 0.1344 | 0.0753 | 0.5177 | 0.2688 |
| 3 | dino_eras_balanced | 0.7489 | 0.0626 | 0.1293 | 0.0722 | 0.5432 | 0.2769 |
| 4 | dino_eras_recall | 0.7711 | 0.0465 | 0.1107 | 0.0610 | 0.5562 | 0.3136 |
| 5 | roadcontrast | 0.8824 | 0.0295 | 0.0634 | 0.0343 | 0.4158 | 0.6656 |
| 6 | roadcontrast_eras_light | 0.8878 | 0.0279 | 0.0610 | 0.0330 | 0.4026 | 0.6814 |
