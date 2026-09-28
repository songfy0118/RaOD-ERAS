# RPSAN / RiskPrompt-SAM

[English](README.md) · [复现步骤](REPRODUCE.md) · [已有实验结果](docs/RESULTS.md)

本项目研究未知道路障碍分割：使用冻结的 DINOv2 提取特征，以道路位置等先验生成 SAM 提示，再用接受的掩码反馈改善异常热力图。推理不读取真值标注，不进行异常专用模型训练。

仓库地址沿用 `RaOD-ERAS`，公开代码和实验文件沿用 `RiskPrompt-SAM` 命名。本次整理针对这一公开实验版本，不代表已完整复现后续 RPSAN 稿件的全部实验。

## 单张图片运行

Python 3.10+；在仓库根目录运行。先安装适合设备的 PyTorch/torchvision，以及项目依赖：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/predict_image.py --image path/to/road.jpg --sam-checkpoint path/to/sam_vit_b_01ec64.pth --out outputs/example
```

SAM ViT-B 权重从 [SAM 官方仓库](https://github.com/facebookresearch/segment-anything#model-checkpoints)获取。DINOv2 首次运行通过官方 Torch Hub 下载；后续使用缓存。单图运行不需要数据集或真值标注。

输出包括二值掩码、叠加图、异常热力图、原始分数数组和运行信息。默认最长边缩放至 1024 像素；`--max-side 0` 保留原尺寸。结果坐标对应缩放后的图片。

## 结果怎么理解

公开快照保留了 189 对图片的受控消融与官方 SMIYC 验证集评估。README 链接中的表格是已有实验记录，不是本次新跑出的完整基准成绩。单图入口采用 D 组完整反馈方法，能执行不等于证明精度或具备实车部署条件。

当前道路先验是图像平面启发式规则，输出不表示真实距离、碰撞时间或车辆控制指令。更多依赖、数据准备与限制见 [REPRODUCE.md](REPRODUCE.md)。
