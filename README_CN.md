# RiskPrompt-SAM

[English](README.md) | [简体中文](README_CN.md)

面向自动驾驶未知道路障碍的训练免费异常分割方法。

## 一句话说明

输入一张道路RGB图像，DINOv2先生成像素级异常热力图；RiskPrompt-SAM再根据道路位置生成提示框，调用SAM补全障碍物轮廓，并把可靠掩码反馈给热力图，最终输出异常热力图和黑白障碍物分割图。

## 实验逻辑

现有方法的主要问题是：

1. 直接阈值化热力图容易产生碎片和道路纹理误报。
2. S2M-style框提示没有显式利用道路空间信息。
3. UGainS-style点提示覆盖范围大，但容易生成重复或过大的SAM候选掩码。

因此，本实验固定DINOv2和SAM-B，只改变“如何从热力图生成提示、如何选择SAM掩码、如何反馈热力图”，从而单独验证提示策略是否有效。

## 方法框架

```text
道路RGB图像
    ↓
冻结的DINOv2提取多尺度特征
    ↓
道路原型距离 + 局部对比度
    ↓
基础异常热力图
    ↓
道路区域 / 本车道 / 近场位置约束
    ↓
自适应异常提示框 + 局部峰值补充
    ↓
冻结的SAM-B生成多个候选掩码
    ↓
SAM置信度 + 掩码内外异常对比
    ↓
选择完整障碍物掩码
    ↓
掩码反向增强热力图并抑制零散响应
    ↓
最终热力图 + 黑白分割图 + 障碍物实例
```

## 具体改进

### 1. 道路感知提示框

普通方法直接在高分区域生成框。本方法额外考虑：

- 是否位于道路区域；
- 是否靠近本车行驶走廊；
- 是否位于车辆近场；
- 异常区域内部平均分数。

这一步主要减少树木、天空、建筑和路外纹理误报。

### 2. 小障碍峰值补充

当连通域过大或没有生成足够提示框时，从道路热力图的局部高峰补充小目标提示，降低小障碍漏检概率。

### 3. SAM多候选选择

SAM对每个提示框输出多个掩码。我们不只选择SAM置信度最高的掩码，还计算：

```text
候选质量 = 0.55 × SAM置信度
         + 0.45 ×（掩码内部异常均值 - 掩码外部边界异常均值）
```

只有内部明显比外部异常的掩码才更可信。

### 4. 掩码反馈热力图

可信障碍物内部的异常分数被增强，未被实例支持的零散响应被抑制。这样同时保留：

- 连续热力图：用于AP和FPR95评价；
- 二值掩码：用于Precision、Recall、F1和IoU评价。

## 对比方法

四种方法使用相同的189张图像、相同DINOv2热力图、相同SAM-B和相同评价代码：

| 方法 | 说明 |
|---|---|
| Threshold | 热力图直接固定阈值二值化 |
| S2M-style | 异常连通域转提示框，再调用SAM |
| UGainS-style | 从高异常像素最远点采样50个点，再调用SAM |
| RiskPrompt-SAM | 道路约束框、峰值补充、边界一致性选择和掩码反馈 |

这里的S2M-style和UGainS-style是核心提示机制复现，不是原论文训练完成的官方模型。

## 数据

实验使用三个公开数据源的本地公开子集：

| 数据源 | 数量 | 类型 |
|---|---:|---|
| SMIYC RoadObstacle | 30 | 真实道路障碍 |
| RoadAnomaly | 10 | 真实道路异常 |
| StreetHazards partial | 149 | 合成道路异常 |
| 合计 | 189 | 统一受控评测 |

统一标签定义：

```text
0   正常
1   异常/障碍物
255 忽略区域
```

这189张是统一评测包，不是新数据集，也不是三个官方完整测试集。详细来源见[DATASET.md](DATASET.md)。

## 最终结果

修正并保留原始void区域后的189张像素微平均结果：

| 方法 | Precision↑ | Recall↑ | F1↑ | IoU↑ | AP↑ | FPR95↓ |
|---|---:|---:|---:|---:|---:|---:|
| Threshold | 0.0312 | **0.5568** | 0.0590 | 0.0304 | 0.0492 | 0.8609 |
| S2M-style | 0.1282 | 0.1223 | 0.1252 | 0.0668 | 0.0492 | 0.8609 |
| UGainS-style | 0.0190 | 0.2889 | 0.0356 | 0.0181 | 0.0458 | 0.8657 |
| **RiskPrompt-SAM** | **0.1601** | 0.1846 | **0.1715** | **0.0938** | **0.0805** | **0.8551** |

相对S2M-style：

- F1：`0.1252 → 0.1715`
- IoU：`0.0668 → 0.0938`
- AP：`0.0492 → 0.0805`
- FPR95：`0.8609 → 0.8551`

配对bootstrap结果显示，总体F1差值为`+0.0624`，95%置信区间为`[0.0494, 0.0758]`。

## 结果边界

可以得出的结论：

> 在统一受控协议下，RiskPrompt-SAM优于固定阈值、复现的S2M-style框提示和UGainS-style点提示。

不能得出的结论：

> RiskPrompt-SAM超过S2M或UGainS论文的官方完整模型。

原因是官方模型使用了训练后的RPL、RbA或检测器前端，数据划分和评价指标也不完全相同。

分数据源结果也存在权衡：

- RoadAnomaly：RiskPrompt分割提升明显；
- StreetHazards：F1、IoU和FPR95提升，但AP下降；
- SMIYC：AP从`0.6894`提高到`0.9159`，但F1从`0.9349`略降到`0.9281`。

## 运行

安装依赖：

```powershell
python -m pip install -r requirements.txt
git lfs pull
tar -xf dist\unified_road_anomaly_eval_189.zip
```

单张测试：

```powershell
conda run -n Test2 python scripts\run_s2m_comparison.py --max-samples 1 --out outputs\riskprompt_smoke
```

完整189张实验：

```powershell
conda run -n Test2 python scripts\run_s2m_comparison.py --max-samples 189 --ugains-threshold 0.60 --out outputs\riskprompt_full_189
```

生成统计和论文图：

```powershell
conda run -n Test2 python scripts\analyze_prompt_results.py outputs\riskprompt_full_189\results.json
conda run -n Test2 python scripts\make_riskprompt_figure.py outputs\riskprompt_full_189\results.json
```

## 主要文件

| 文件 | 作用 |
|---|---|
| `src/raod_eras/score_to_mask.py` | S2M-style、UGainS-style和RiskPrompt提示/掩码算法 |
| `scripts/run_s2m_comparison.py` | 四方法统一对比实验 |
| `scripts/analyze_prompt_results.py` | 指标汇总与bootstrap统计 |
| `scripts/build_unified_dataset.py` | 统一标签和元数据构建 |
| `paper/riskprompt_paper_draft_en.md` | 当前英文论文初稿 |
| `outputs/riskprompt_full_189/results_summary.json` | 精简最终结果 |
