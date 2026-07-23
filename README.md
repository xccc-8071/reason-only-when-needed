[English](README.md) | [中文](README-zh.md)

---

# *Reason Only When Needed: Efficient Generative Reward Modeling via Model-Internal Uncertainty*

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

**Abstract.** Recent advancements in the Generative Reward Model (GRM) have
demonstrated its potential to enhance the reasoning abilities of LLMs through
Chain-of-Thought (CoT) prompting. Despite these gains, existing implementations
of GRM suffer from two critical limitations. First, CoT prompting is applied
indiscriminately to all inputs regardless of their inherent complexity. This
introduces unnecessary computational costs for tasks amenable to fast, direct
inference. Second, existing approaches primarily rely on voting-based mechanisms
to evaluate CoT outputs, which often lack granularity and precision in assessing
reasoning quality. In this paper, we propose E-GRM, an efficient generative
reward modeling framework grounded in model-internal uncertainty. E-GRM
leverages the convergence behavior of parallel model generations to estimate
uncertainty and selectively trigger CoT reasoning only when needed, without
relying on handcrafted features or task-dependent signals. To improve reward
fidelity, we introduce a lightweight discriminative scorer trained with a hybrid
regression--ranking objective to provide fine-grained evaluation of reasoning
paths.

## Framework Overview

```
Parallel Decode → Uncertainty → Route → Score
```

| Phase | Module | Description |
|-------|--------|-------------|
| **Uncertainty** | `egrm_uncertainty.py` | M parallel decodings → consensus → trigger signal |
| **Scorer** | `egrm_scorer.py` | Hybrid Huber + Hinge loss discriminative scorer |
| **Router** | `egrm_router.py` | Direct / CoT routing + SFT + GRPO training |
| Pipeline | `egrm_pipeline.py` | End-to-end orchestrator |

## Key Mechanism

### 1. Dynamic CoT Triggering

- Run **M=5** parallel decodings with varied sampling params (T∈{0, 0.3, 0.7, 1.0})
- Compute **Consensus(x)** = max_y Count(y) / M
- If ≥ τ (0.8): direct answer (~58% of samples) — no CoT cost
- If < τ: full CoT generation + discriminative scoring

### 2. Discriminative Scorer

- **Hybrid loss**: L = (1−α) · Huber_regression + α · Hinge_ranking
- α = 0.3 balances absolute reward fidelity with pairwise preference

### 3. Training Pipeline

```
SFT (mixed short + long CoT) → GRPO (pairwise preference optimization)
```

| Stage | Data | Objective |
|-------|------|-----------|
| SFT | 50% direct / 50% CoT | Teach both reasoning modes |
| GRPO | Pairwise path comparisons | Fine-tune scorer with hybrid loss |

## Key Results

- **Latency**: −62% vs. full CoT baseline (RM-Bench)
- **Accuracy**: +3.2% on RM-Bench, +2.8% on RMB, +2.1% on RewardBench
- **Trigger rate**: ~42% of samples routed to CoT (τ=0.8)
- **Consensus signal**: effective across all benchmarks without handcrafted features

## Environment Setup

### Requirements

```bash
pip install numpy torch transformers accelerate sentencepiece protobuf datasets
```

### Dry-run (no GPU)

```bash
python experiments.py --benchmark rm-bench --max-samples 200 --dry-run
```

### Full inference (GPU + model)

```bash
python experiments.py --benchmark rm-bench --model Qwen/Qwen2.5-7B --max-samples 500
```

### With training

```bash
python experiments.py --benchmark rm-bench --max-samples 200 --dry-run --train
```

### Programmatic use

```python
from egrm_pipeline import EGRMPipeline
from egrm_uncertainty import UncertaintyEstimator

pipeline = EGRMPipeline(consensus_threshold=0.8, num_parallel_runs=5)
report = pipeline.run(
    prompts=["What is 15 * 7?"],
    direct_model_fn=your_direct_fn,
    cot_model_fn=your_cot_fn,
)
print(f"Direct ratio: {report.direct_ratio:.1%}")
print(f"Avg latency:  {report.avg_latency_ms:.1f} ms")
```

## Citation

```bibtex
@inproceedings{xue2026reason,
   title={Reason only when needed: Efficient generative reward modeling via model-internal uncertainty},
   author={Xue, Chao and Wang, Yao and Liu, Mengqiao and Liang, Di and Han, Xingsheng and Liu, Peiyang and Wu, Xianjie and Lu, Chenyao and Jiang, Lei and Lu, Yu and others},
   booktitle={Findings of the Association for Computational Linguistics: ACL 2026},
   pages={23302--23319},
   year={2026}
}
```

## Reproducibility Notes

The code provides the full algorithmic skeleton of E-GRM. Exact numerical
reproduction of the paper's results requires the following setup:

| Component | Status | What is missing for exact reproduction |
|-----------|--------|----------------------------------------|
| Parallel decoding + consensus routing | Fully implemented | — |
| Hybrid (Huber + Hinge) loss | Fully implemented | — |
| SFT mixed-data preparation | Fully implemented | — |
| GRPO pairwise preference pipeline | Fully implemented | — |
| Real model inference | HF model loading ready | Requires GPU + Qwen2.5-7B weights (~40 GB VRAM) |
| Scorer training (real backprop) | Mock gradient update | Needs GPU + RM-Bench / RMB / RewardBench data |
| SFT / GRPO training loss | Simulated loss value | Needs real `model.generate()` outputs |
| Benchmark datasets | Synthetic templates | RM-Bench, RMB, RewardBench JSONL files required |
| Baseline comparisons (Table 2–4) | Not included | GRM, GRPO-vanilla, AdaCoT implementations |
| Ablation scans (τ, M, α) | Not automated | Manual sweep needed |

**To reproduce exact paper numbers:** a Linux server with an NVIDIA A100 (80 GB)
or equivalent, plus the Qwen2.5-7B model weights and real benchmark datasets.
The `--dry-run` mode uses synthetic prompts and mock model calls for
pipeline validation only — no realistic results should be expected from it.

## License

MIT
