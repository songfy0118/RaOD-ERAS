# SMIYC官方协议结果记录

## 协议

- 官方仓库：SegmentMeIfYouCan `road-anomaly-benchmark`
- 数据集：`ObstacleTrack-validation`
- 样本：30张，1080×1920
- 像素指标：`PixBinaryClass`
- 组件指标：`SegEval-ObstacleTrack`
- 输出：官方Evaluation API生成的HDF5异常分数文件

## 结果

| 方法 | AUPR | AUROC | FPR95 | best pixel F1 | GT-sIoU | Pred-sIoU | PPV | mean component F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DINOBase-v2 | 0.692558 | 0.996686 | 0.012914 | 0.670601 | 0.244227 | 0.477836 | 0.688663 | 0.351735 |
| RiskPromptSAM-v2 | 0.918974 | 0.998300 | 0.014753 | 0.934355 | 0.483874 | 0.680448 | 0.711718 | 0.623377 |

## 文件

- `outputs/smiyc_official_protocol/protocol_run_summary_DINOBase-v2.json`
- `outputs/smiyc_official_protocol/protocol_run_summary_RiskPromptSAM-v2.json`
- `outputs/smiyc_official_protocol/benchmark_outputs/PixBinaryClass/`
- `outputs/smiyc_official_protocol/benchmark_outputs/SegEval-ObstacleTrack/`
