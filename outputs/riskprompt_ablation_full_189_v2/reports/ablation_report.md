# RiskPrompt-SAM sequential ablation

Samples: 189.

C and D intentionally share the same binary mask. Their difference is evaluated on continuous heatmap AP/FPR95.

## Pixel-micro main table

| Variant | Precision | Recall | F1 | IoU | AP | FPR95 |
|---|---:|---:|---:|---:|---:|---:|
| A_basic_box | 0.1282 | 0.1223 | 0.1252 | 0.0668 | 0.0492 | 0.8609 |
| B_road_box | 0.1587 | 0.1834 | 0.1701 | 0.0930 | 0.0492 | 0.8609 |
| C_boundary_select | 0.1593 | 0.1836 | 0.1706 | 0.0932 | 0.0492 | 0.8609 |
| D_full_feedback | 0.1593 | 0.1836 | 0.1706 | 0.0932 | 0.0810 | 0.8551 |

## Sequential paired image-macro differences

| Scope | Module | Metric | Mean difference | 95% CI | P(positive) |
|---|---|---|---:|---:|---:|
| aggregate | road-aware boxes | fixed_f1 | 0.0615 | [0.0488, 0.0748] | 1.000 |
| aggregate | road-aware boxes | fixed_iou | 0.0363 | [0.0264, 0.0475] | 1.000 |
| aggregate | road-aware boxes | component_f1 | 0.0186 | [0.0085, 0.0278] | 1.000 |
| aggregate | road-aware boxes | boundary_f1 | 0.0681 | [0.0555, 0.0803] | 1.000 |
| aggregate | boundary-consistent selection | fixed_f1 | -0.0001 | [-0.0011, 0.0008] | 0.421 |
| aggregate | boundary-consistent selection | fixed_iou | -0.0005 | [-0.0018, 0.0004] | 0.194 |
| aggregate | boundary-consistent selection | component_f1 | -0.0020 | [-0.0056, 0.0003] | 0.076 |
| aggregate | boundary-consistent selection | boundary_f1 | -0.0016 | [-0.0043, 0.0002] | 0.062 |
| aggregate | mask feedback | fixed_f1 | 0.0000 | [0.0000, 0.0000] | 0.000 |
| aggregate | mask feedback | fixed_iou | 0.0000 | [0.0000, 0.0000] | 0.000 |
| aggregate | mask feedback | component_f1 | 0.0000 | [0.0000, 0.0000] | 0.000 |
| aggregate | mask feedback | boundary_f1 | 0.0000 | [0.0000, 0.0000] | 0.000 |
| road_anomaly | road-aware boxes | fixed_f1 | 0.1400 | [0.0244, 0.2767] | 0.995 |
| road_anomaly | road-aware boxes | fixed_iou | 0.1462 | [0.0217, 0.2968] | 0.993 |
| road_anomaly | road-aware boxes | component_f1 | 0.0849 | [-0.0043, 0.1750] | 0.970 |
| road_anomaly | road-aware boxes | boundary_f1 | 0.0925 | [0.0070, 0.2005] | 0.987 |
| road_anomaly | boundary-consistent selection | fixed_f1 | -0.0005 | [-0.0014, 0.0000] | 0.000 |
| road_anomaly | boundary-consistent selection | fixed_iou | -0.0008 | [-0.0025, 0.0000] | 0.000 |
| road_anomaly | boundary-consistent selection | component_f1 | -0.0067 | [-0.0200, 0.0000] | 0.000 |
| road_anomaly | boundary-consistent selection | boundary_f1 | -0.0022 | [-0.0060, 0.0000] | 0.000 |
| road_anomaly | mask feedback | fixed_f1 | 0.0000 | [0.0000, 0.0000] | 0.000 |
| road_anomaly | mask feedback | fixed_iou | 0.0000 | [0.0000, 0.0000] | 0.000 |
| road_anomaly | mask feedback | component_f1 | 0.0000 | [0.0000, 0.0000] | 0.000 |
| road_anomaly | mask feedback | boundary_f1 | 0.0000 | [0.0000, 0.0000] | 0.000 |
| smiyc | road-aware boxes | fixed_f1 | -0.0059 | [-0.0121, -0.0012] | 0.002 |
| smiyc | road-aware boxes | fixed_iou | -0.0101 | [-0.0207, -0.0019] | 0.004 |
| smiyc | road-aware boxes | component_f1 | -0.0333 | [-0.0778, 0.0000] | 0.000 |
| smiyc | road-aware boxes | boundary_f1 | -0.0261 | [-0.0494, -0.0065] | 0.000 |
| smiyc | boundary-consistent selection | fixed_f1 | -0.0028 | [-0.0079, 0.0000] | 0.000 |
| smiyc | boundary-consistent selection | fixed_iou | -0.0043 | [-0.0120, 0.0000] | 0.000 |
| smiyc | boundary-consistent selection | component_f1 | -0.0100 | [-0.0300, 0.0000] | 0.000 |
| smiyc | boundary-consistent selection | boundary_f1 | -0.0099 | [-0.0263, 0.0000] | 0.000 |
| smiyc | mask feedback | fixed_f1 | 0.0000 | [0.0000, 0.0000] | 0.000 |
| smiyc | mask feedback | fixed_iou | 0.0000 | [0.0000, 0.0000] | 0.000 |
| smiyc | mask feedback | component_f1 | 0.0000 | [0.0000, 0.0000] | 0.000 |
| smiyc | mask feedback | boundary_f1 | 0.0000 | [0.0000, 0.0000] | 0.000 |
| street_hazards | road-aware boxes | fixed_f1 | 0.0697 | [0.0573, 0.0835] | 1.000 |
| street_hazards | road-aware boxes | fixed_iou | 0.0382 | [0.0310, 0.0464] | 1.000 |
| street_hazards | road-aware boxes | component_f1 | 0.0247 | [0.0186, 0.0312] | 1.000 |
| street_hazards | road-aware boxes | boundary_f1 | 0.0855 | [0.0733, 0.0978] | 1.000 |
| street_hazards | boundary-consistent selection | fixed_f1 | 0.0005 | [-0.0002, 0.0014] | 0.869 |
| street_hazards | boundary-consistent selection | fixed_iou | 0.0003 | [-0.0001, 0.0008] | 0.876 |
| street_hazards | boundary-consistent selection | component_f1 | -0.0000 | [-0.0011, 0.0008] | 0.489 |
| street_hazards | boundary-consistent selection | boundary_f1 | 0.0001 | [-0.0005, 0.0010] | 0.567 |
| street_hazards | mask feedback | fixed_f1 | 0.0000 | [0.0000, 0.0000] | 0.000 |
| street_hazards | mask feedback | fixed_iou | 0.0000 | [0.0000, 0.0000] | 0.000 |
| street_hazards | mask feedback | component_f1 | 0.0000 | [0.0000, 0.0000] | 0.000 |
| street_hazards | mask feedback | boundary_f1 | 0.0000 | [0.0000, 0.0000] | 0.000 |

## Feedback heatmap differences

| Scope | Metric | Mean improvement | 95% CI | P(positive) |
|---|---|---:|---:|---:|
| aggregate | AP (pixel-micro, no image-bootstrap) | 0.0318 | n/a | n/a |
| aggregate | FPR95 reduction (pixel-micro) | 0.0058 | n/a | n/a |
