[English](README.md) | [中文](README-zh.md)

---

# *Reason Only When Needed: Efficient Generative Reward Modeling via Model-Internal Uncertainty*

*必要时才推理：基于模型内部不确定性的高效生成式奖励建模*

[Chao Xue](https://aclanthology.org/people/chao-xue-7974/)
· [Yao Wang](https://aclanthology.org/people/yao-wang/)
· [Mengqiao Liu](https://aclanthology.org/people/mengqiao-liu/)
· [Di Liang](https://aclanthology.org/people/di-liang/)
· [Xingsheng Han](https://aclanthology.org/people/xingsheng-han/)
· [Peiyang Liu](https://aclanthology.org/people/peiyang-liu/)
· [Xianjie Wu](https://aclanthology.org/people/xianjie-wu/)
· [Chenyao Lu](https://aclanthology.org/people/chenyao-lu/)
· [Lei Jiang](https://aclanthology.org/people/lei-jiang-3052/)
· [Yu Lu](https://aclanthology.org/people/yu-lu-7040/)
· [Haibo Shi](https://aclanthology.org/people/haibo-shi/)
· [Shuang Liang](https://aclanthology.org/people/shuang-liang/)
· [Minlong Peng](https://aclanthology.org/people/minlong-peng/)
· [Flora D. Salim](https://aclanthology.org/people/flora-d-salim/)

*Findings of ACL 2026* · [arXiv](https://arxiv.org/abs/2604.10072) · [ACL Anthology](https://aclanthology.org/2026.findings-acl.1167/)

**摘要。** 最近，生成式奖励模型（GRM）的进展展示了通过思维链（CoT）提示
增强 LLM 推理能力的潜力。然而，现有 GRM 实现存在两个关键局限。第一，CoT
提示不加区分地应用于所有输入，无论其固有复杂度如何，为适合快速直接推理的
任务引入不必要的计算开销。第二，现有方法主要依赖投票机制评估 CoT 输出，
这在评估推理质量时往往缺乏细粒度和精确性。本文提出 E-GRM，一种基于模型
内部不确定性的高效生成式奖励建模框架。E-GRM 利用并行模型生成的收敛行为
来估计不确定性，仅在需要时选择性触发 CoT 推理，无需手工设计特征或任务相关
信号。为提升奖励保真度，我们引入一个轻量级判别评分器，以混合回归—排序
目标函数训练，提供对推理路径的细粒度评估。

## 框架总览

```
并行解码（Parallel Decode）→ 不确定性估计（Uncertainty）→ 路由（Route）→ 评分（Score）
```

| 阶段 | 模块 | 说明 |
|------|------|------|
| **不确定性** | `egrm_uncertainty.py` | M 次并行解码 → 一致性 → 触发信号 |
| **评分器** | `egrm_scorer.py` | 混合 Huber + Hinge 损失判别评分器 |
| **路由器** | `egrm_router.py` | 直答 / CoT 路由 + SFT + GRPO 训练 |
| 编排 | `egrm_pipeline.py` | 端到端流程编排器 |

## 关键机制

### 1. 动态 CoT 触发

- 执行 **M=5** 次并行解码，采样参数各异（T∈{0, 0.3, 0.7, 1.0}）
- 计算 **Consensus(x)** = max_y Count(y) / M
- ≥ τ (0.8)：直接输出答案（约 58% 样本）——无需 CoT 开销
- < τ：触发完整 CoT 生成 + 判别评分

### 2. 判别评分器

- **混合损失**：L = (1−α) · Huber_回归 + α · Hinge_排序
- α = 0.3 平衡绝对奖励值和相对偏好排序

### 3. 训练流程

```
SFT（混合短答 + 长 CoT）→ GRPO（成对偏好优化）
```

| 阶段 | 数据 | 目标 |
|------|------|------|
| SFT | 50% 直接 / 50% CoT | 学习两种推理模式 |
| GRPO | 成对路径对比 | 用混合损失微调评分器 |

## 核心结果

- **延迟**：相比全 CoT 基线降低 62%（RM-Bench）
- **准确率**：RM-Bench +3.2%，RMB +2.8%，RewardBench +2.1%
- **触发率**：约 42% 样本路由到 CoT（τ=0.8）
- **一致性信号**：跨所有基准有效，无需手工特征

## 环境配置

### 依赖安装

```bash
pip install numpy torch transformers accelerate sentencepiece protobuf datasets
```

### Dry-run（无需 GPU）

```bash
python experiments.py --benchmark rm-bench --max-samples 200 --dry-run
```

### 完整推理（需 GPU + 模型）

```bash
python experiments.py --benchmark rm-bench --model Qwen/Qwen2.5-7B --max-samples 500
```

### 含训练流程

```bash
python experiments.py --benchmark rm-bench --max-samples 200 --dry-run --train
```

### 编程方式调用

```python
from egrm_pipeline import EGRMPipeline
from egrm_uncertainty import UncertaintyEstimator

pipeline = EGRMPipeline(consensus_threshold=0.8, num_parallel_runs=5)
report = pipeline.run(
    prompts=["15 * 7 等于多少？"],
    direct_model_fn=你的直答函数,
    cot_model_fn=你的CoT函数,
)
print(f"直接应答率: {report.direct_ratio:.1%}")
print(f"平均延迟:   {report.avg_latency_ms:.1f} ms")
```

## 引用

```bibtex
@inproceedings{xue-etal-2026-reason,
  title     = {Reason Only When Needed: Efficient Generative
               Reward Modeling via Model-Internal Uncertainty},
  author    = {Chao Xue and Yao Wang and Mengqiao Liu and Di Liang and
               Xingsheng Han and Peiyang Liu and Xianjie Wu and Chenyao Lu and
               Lei Jiang and Yu Lu and Haibo Shi and Shuang Liang and
               Minlong Peng and Flora D. Salim},
  booktitle = {Findings of the Association for Computational
               Linguistics: ACL 2026},
  pages     = {23302--23319},
  year      = {2026},
  address   = {San Diego, California, United States},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/2026.findings-acl.1167},
  url       = {https://aclanthology.org/2026.findings-acl.1167/},
}
```

## 复现说明

代码提供了 E-GRM 的完整算法框架。要完全复现论文中的数值结果，需要以下条件：

| 组件 | 状态 | 缺少什么 |
|------|------|----------|
| 并行解码 + 一致性路由 | 完整实现 | — |
| 混合 (Huber + Hinge) 损失 | 完整实现 | — |
| SFT 混合数据准备 | 完整实现 | — |
| GRPO 成对偏好流程 | 完整实现 | — |
| 真实模型推理 | HF 加载已就绪 | 需 GPU + Qwen2.5-7B 权重（~40 GB 显存） |
| 评分器训练（真实反向传播） | 模拟梯度更新 | 需 GPU + RM-Bench / RMB / RewardBench 数据 |
| SFT / GRPO 训练损失 | 模拟损失值 | 需真实 `model.generate()` 输出 |
| 基准数据集 | 合成模板 | 需 RM-Bench、RMB、RewardBench 真实 JSONL |
| 基线对比（Table 2–4） | 未实现 | GRM、GRPO-vanilla、AdaCoT |
| 消融扫描（τ, M, α） | 未自动化 | 需手动或脚本遍历 |

**要复现精确数值：** 需要一台搭载 NVIDIA A100（80 GB）或同等规格 GPU 的 Linux 服务器，
以及 Qwen2.5-7B 模型权重和真实基准数据集。
`--dry-run` 模式仅用于验证流程正确性，使用合成 prompt 和 Mock 模型调用，
不应期待从中获得真实结果。

## 许可证

MIT
