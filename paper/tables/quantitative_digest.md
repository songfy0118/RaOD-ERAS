# Quantitative Digest

Lower is better for FPR95; higher is better for all other metrics.

## SMIYC RoadObstacle

| Method | AP | F1 | IoU | Precision | Recall | FPR95 |
|---|---:|---:|---:|---:|---:|---:|
| roadcontrast | 0.1169 | 0.1739 | 0.1108 | 0.1428 | 0.3690 | 0.4540 |
| roadcontrast_eras_light | 0.1457 | 0.1921 | 0.1306 | 0.2115 | 0.3724 | 0.4541 |
| dino | 0.5203 | 0.5228 | 0.3998 | 0.4781 | 0.6864 | 0.0268 |
| dino_eras_light | 0.5271 | 0.5232 | 0.3989 | 0.4693 | 0.7524 | 0.0256 |
| dino_eras_balanced | 0.4975 | 0.5039 | 0.3825 | 0.4311 | 0.8062 | 0.0263 |

## RoadAnomaly21

| Method | AP | F1 | IoU | Precision | Recall | FPR95 |
|---|---:|---:|---:|---:|---:|---:|
| roadcontrast | 0.3698 | 0.4473 | 0.3059 | 0.4022 | 0.6188 | 0.6409 |
| roadcontrast_eras_light | 0.3717 | 0.4529 | 0.3095 | 0.4114 | 0.6142 | 0.6252 |
| dino | 0.1324 | 0.3119 | 0.1907 | 0.1924 | 0.9314 | 0.6749 |
| dino_eras_light | 0.2907 | 0.4348 | 0.2865 | 0.3497 | 0.7624 | 0.5889 |
| dino_eras_balanced | 0.3227 | 0.4782 | 0.3289 | 0.3640 | 0.8369 | 0.5592 |

## StreetHazards partial

| Method | AP | F1 | IoU | Precision | Recall | FPR95 |
|---|---:|---:|---:|---:|---:|---:|
| roadcontrast | 0.0295 | 0.0634 | 0.0343 | 0.0421 | 0.4158 | 0.6656 |
| roadcontrast_eras_light | 0.0279 | 0.0610 | 0.0330 | 0.0437 | 0.4026 | 0.6814 |
| dino | 0.1133 | 0.1588 | 0.0975 | 0.1424 | 0.4781 | 0.3671 |
| dino_eras_light | 0.0691 | 0.1344 | 0.0753 | 0.1077 | 0.5177 | 0.2688 |
| dino_eras_balanced | 0.0626 | 0.1293 | 0.0722 | 0.1013 | 0.5432 | 0.2769 |
