# RiskPrompt-SAM experiment statistics

Samples: 189; source counts: {'road_anomaly': 10, 'smiyc': 30, 'street_hazards': 149}.

Dataset-level values below are pixel-micro aggregates. Bootstrap intervals are paired image-macro differences.

### aggregate

| Method | Precision | Recall | F1 | IoU | AP | FPR95 |
|---|---:|---:|---:|---:|---:|---:|
| threshold | 0.0312 | 0.5568 | 0.0590 | 0.0304 | 0.0492 | 0.8609 |
| s2m_style | 0.1282 | 0.1223 | 0.1252 | 0.0668 | 0.0492 | 0.8609 |
| ugains_style | 0.0190 | 0.2889 | 0.0356 | 0.0181 | 0.0458 | 0.8657 |
| riskprompt | 0.1601 | 0.1846 | 0.1715 | 0.0938 | 0.0805 | 0.8551 |

### road_anomaly

| Method | Precision | Recall | F1 | IoU | AP | FPR95 |
|---|---:|---:|---:|---:|---:|---:|
| threshold | 0.1074 | 0.1951 | 0.1386 | 0.0744 | 0.1110 | 0.9815 |
| s2m_style | 0.2747 | 0.1944 | 0.2277 | 0.1285 | 0.1110 | 0.9815 |
| ugains_style | 0.0847 | 0.1991 | 0.1188 | 0.0632 | 0.1030 | 0.9825 |
| riskprompt | 0.6529 | 0.2196 | 0.3286 | 0.1966 | 0.2688 | 0.9826 |

### smiyc

| Method | Precision | Recall | F1 | IoU | AP | FPR95 |
|---|---:|---:|---:|---:|---:|---:|
| threshold | 0.4576 | 0.9142 | 0.6099 | 0.4388 | 0.6894 | 0.0135 |
| s2m_style | 0.9699 | 0.9023 | 0.9349 | 0.8777 | 0.6894 | 0.0135 |
| ugains_style | 0.9967 | 0.0882 | 0.1621 | 0.0882 | 0.7111 | 0.0136 |
| riskprompt | 0.9539 | 0.9036 | 0.9281 | 0.8658 | 0.9159 | 0.0157 |

### street_hazards

| Method | Precision | Recall | F1 | IoU | AP | FPR95 |
|---|---:|---:|---:|---:|---:|---:|
| threshold | 0.0256 | 0.8176 | 0.0496 | 0.0255 | 0.0646 | 0.5630 |
| s2m_style | 0.0030 | 0.0034 | 0.0032 | 0.0016 | 0.0646 | 0.5630 |
| ugains_style | 0.0141 | 0.3765 | 0.0272 | 0.0138 | 0.0636 | 0.5847 |
| riskprompt | 0.0550 | 0.1001 | 0.0710 | 0.0368 | 0.0517 | 0.5409 |

## Paired bootstrap: RiskPrompt minus baseline

| Scope | Baseline | Metric | Mean difference | 95% CI | P(diff>0) |
|---|---|---|---:|---:|---:|
| aggregate | s2m_style | fixed_f1 | 0.0624 | [0.0494, 0.0758] | 1.000 |
| aggregate | s2m_style | fixed_iou | 0.0365 | [0.0264, 0.0479] | 1.000 |
| aggregate | s2m_style | component_f1 | 0.0166 | [0.0062, 0.0261] | 0.999 |
| aggregate | s2m_style | boundary_f1 | 0.0664 | [0.0532, 0.0787] | 1.000 |
| aggregate | ugains_style | fixed_f1 | 0.1482 | [0.1049, 0.1940] | 1.000 |
| aggregate | ugains_style | fixed_iou | 0.1258 | [0.0861, 0.1684] | 1.000 |
| aggregate | ugains_style | component_f1 | 0.1141 | [0.0751, 0.1555] | 1.000 |
| aggregate | ugains_style | boundary_f1 | 0.1318 | [0.0942, 0.1719] | 1.000 |
| road_anomaly | s2m_style | fixed_f1 | 0.1397 | [0.0253, 0.2756] | 0.995 |
| road_anomaly | s2m_style | fixed_iou | 0.1459 | [0.0223, 0.2947] | 0.993 |
| road_anomaly | s2m_style | component_f1 | 0.0824 | [-0.0031, 0.1667] | 0.969 |
| road_anomaly | s2m_style | boundary_f1 | 0.0904 | [0.0058, 0.1952] | 0.986 |
| road_anomaly | ugains_style | fixed_f1 | 0.3105 | [0.0910, 0.5379] | 0.998 |
| road_anomaly | ugains_style | fixed_iou | 0.2942 | [0.0966, 0.5045] | 0.999 |
| road_anomaly | ugains_style | component_f1 | 0.1542 | [-0.0076, 0.3113] | 0.968 |
| road_anomaly | ugains_style | boundary_f1 | 0.1748 | [0.0092, 0.3747] | 0.984 |
| smiyc | s2m_style | fixed_f1 | -0.0080 | [-0.0164, -0.0012] | 0.008 |
| smiyc | s2m_style | fixed_iou | -0.0135 | [-0.0269, -0.0024] | 0.005 |
| smiyc | s2m_style | component_f1 | -0.0433 | [-0.0867, -0.0100] | 0.000 |
| smiyc | s2m_style | boundary_f1 | -0.0329 | [-0.0601, -0.0095] | 0.001 |
| smiyc | ugains_style | fixed_f1 | 0.6412 | [0.4841, 0.7868] | 1.000 |
| smiyc | ugains_style | fixed_iou | 0.6042 | [0.4545, 0.7425] | 1.000 |
| smiyc | ugains_style | component_f1 | 0.5852 | [0.4286, 0.7371] | 1.000 |
| smiyc | ugains_style | boundary_f1 | 0.5804 | [0.4369, 0.7161] | 1.000 |
| street_hazards | s2m_style | fixed_f1 | 0.0714 | [0.0586, 0.0855] | 1.000 |
| street_hazards | s2m_style | fixed_iou | 0.0392 | [0.0318, 0.0475] | 1.000 |
| street_hazards | s2m_style | component_f1 | 0.0242 | [0.0181, 0.0308] | 1.000 |
| street_hazards | s2m_style | boundary_f1 | 0.0848 | [0.0726, 0.0973] | 1.000 |
| street_hazards | ugains_style | fixed_f1 | 0.0381 | [0.0164, 0.0579] | 0.999 |
| street_hazards | ugains_style | fixed_iou | 0.0182 | [0.0021, 0.0319] | 0.984 |
| street_hazards | ugains_style | component_f1 | 0.0165 | [0.0094, 0.0235] | 1.000 |
| street_hazards | ugains_style | boundary_f1 | 0.0385 | [0.0244, 0.0528] | 1.000 |

## Runtime

Median 0.10s/image; P90 0.19s/image; total 21.1s.
