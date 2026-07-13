"""
E-GRM Pipeline
--------------
Uncertainty → Route → Score.

End-to-end orchestration: training + inference + evaluation.

"""

from __future__ import annotations

import json
import time
import numpy as np

from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field

from egrm_uncertainty import ParallelDecoder, UncertaintyEstimator
from egrm_scorer import DiscriminativeScorer, HybridLoss, ReasoningPath, TrainingSample
from egrm_router import CoTRouter, RouterDecision, SFTExample, GRPOPair


@dataclass
class EGRMReport:
    """Full E-GRM analysis report."""
    n_samples: int
    direct_ratio: float              # fraction routed to direct (no CoT)
    avg_latency_ms: float
    avg_tokens_per_sample: float
    avg_consensus: float
    cot_trigger_rate: float
    total_tokens_saved: int
    decisions: List[RouterDecision] = field(default_factory=list)
    sft_loss: Optional[Dict] = None
    scorer_loss: Optional[Dict] = None


class EGRMPipeline:
    """
    E-GRM: Efficient Generative Reward Modeling.

    Inference: parallel decode → uncertainty → route (direct / CoT) → score.

    Training:
      1. SFT: mixed short + long CoT samples
      2. GRPO: pairwise preference optimization with hybrid loss
    """

    def __init__(
        self,
        consensus_threshold: float = 0.8,
        num_parallel_runs: int = 5,
        num_cot_paths: int = 4,
    ):
        self.threshold = consensus_threshold
        self.num_parallel = num_parallel_runs
        self.num_cot_paths = num_cot_paths
        self.scorer = DiscriminativeScorer()
        self.router = CoTRouter(
            consensus_threshold=consensus_threshold,
            num_parallel=num_parallel_runs,
            num_cot_paths=num_cot_paths,
            scorer=self.scorer,
        )

    # ---- Inference ----

    def run(
        self,
        prompts: List[str],
        direct_model_fn: Callable,
        cot_model_fn: Optional[Callable] = None,
        max_tokens: int = 128,
        verbose: bool = True,
    ) -> EGRMReport:
        """
        Run E-GRM inference on a batch of prompts.
        Returns: routing stats, latency, token savings, per-decision details.
        """
        def log(msg):
            if verbose:
                print(msg)

        log("=" * 60)
        log("E-GRM Pipeline: Parallel Decode → Route → Score")
        log(f"  Threshold τ = {self.threshold}  |  Parallel M = {self.num_parallel}")
        log("=" * 60)

        decisions = []
        total_latency = 0.0
        total_tokens = 0
        n_direct = 0
        n_cot = 0
        tokens_saved = 0

        for i, prompt in enumerate(prompts):
            decision = self.router.decide(
                prompt, direct_model_fn, cot_model_fn, max_tokens,
            )
            decisions.append(decision)
            total_latency += decision.latency_ms
            total_tokens += decision.tokens_used
            if decision.route == "direct":
                n_direct += 1
                tokens_saved += 200 * self.num_parallel  # estimated CoT saving
            else:
                n_cot += 1

            if verbose and (i + 1) % 20 == 0:
                log(f"  [{i + 1}/{len(prompts)}] "
                    f"direct={n_direct} cot={n_cot}")

        n = len(prompts)
        log(f"\n---")
        log(f"  Total: {n} prompts")
        log(f"  Direct (no CoT): {n_direct} ({n_direct / n * 100:.1f}%)")
        log(f"  CoT triggered:  {n_cot} ({n_cot / n * 100:.1f}%)")
        log(f"  Avg latency: {total_latency / n:.1f} ms/sample")
        log(f"  Avg tokens:  {total_tokens / n:.0f} tokens/sample")
        log(f"  Tokens saved: ~{tokens_saved}")
        log("=" * 60)

        return EGRMReport(
            n_samples=n,
            direct_ratio=round(n_direct / n, 4),
            avg_latency_ms=round(total_latency / n, 1),
            avg_tokens_per_sample=round(total_tokens / n, 1),
            avg_consensus=round(
                sum(d.consensus for d in decisions) / n, 4
            ),
            cot_trigger_rate=round(n_cot / n, 4),
            total_tokens_saved=tokens_saved,
            decisions=decisions,
        )

    # ---- Training ----

    def train_sft(
        self,
        samples: List[SFTExample],
        mix_ratio: float = 0.5,
        epochs: int = 1,
        lr: float = 0.01,
        verbose: bool = True,
    ) -> Dict:
        """SFT stage: mixed short + long CoT."""
        data = self.router.prepare_sft_data(samples, mix_ratio)
        loss_hist = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for sample in data:
                # Proxy loss when running without real model outputs
                epoch_loss += abs(len(sample["completion"]) - 50) / 1000.0
            loss_hist.append(round(epoch_loss / len(data), 6))
            if verbose:
                print(f"  SFT epoch {epoch + 1}/{epochs} — loss: {loss_hist[-1]:.4f}")
        return {"sft_loss": loss_hist, "n_samples": len(data)}

    def train_grpo(
        self,
        prompts: List[str],
        generate_paths_fn: Callable[..., List[ReasoningPath]],
        iterations: int = 3,
        lr: float = 0.01,
        verbose: bool = True,
    ) -> Dict:
        """GRPO stage: pairwise preference optimization."""
        pairs = self.router.prepare_grpo_pairs(prompts, generate_paths_fn)
        if not pairs:
            return {"grpo_pairs": 0, "loss_history": []}

        loss_hist = []
        for it in range(iterations):
            pred_scores = []
            true_scores = []
            pairwise = []
            for k, pair in enumerate(pairs):
                sp = self.scorer.score(pair.chosen_path)
                sq = self.scorer.score(pair.rejected_path)
                pred_scores.extend([sp, sq])
                true_scores.extend([pair.chosen_score, pair.rejected_score])
                pairwise.append((2 * k, 2 * k + 1, True))
            loss_info = self.scorer.train_step(pred_scores, true_scores, pairwise, lr)
            loss_hist.append(loss_info)
            if verbose:
                print(f"  GRPO iter {it + 1}/{iterations} — "
                      f"huber={loss_info['huber']:.4f} hinge={loss_info['hinge']:.4f}")
        return {"grpo_pairs": len(pairs), "loss_history": loss_hist}

    # ---- Full training routine ----

    def train(
        self,
        sft_samples: List[SFTExample],
        grpo_prompts: List[str],
        generate_paths_fn: Callable,
        verbose: bool = True,
    ) -> Dict:
        if verbose:
            print("=" * 50)
            print("Training E-GRM: SFT → GRPO")
            print("=" * 50)
            print("\n[Stage 1] SFT (mixed short + long CoT)")
        sft_result = self.train_sft(sft_samples, mix_ratio=0.5, epochs=1, verbose=verbose)
        if verbose:
            print("\n[Stage 2] GRPO (pairwise preference)")
        grpo_result = self.train_grpo(grpo_prompts, generate_paths_fn, iterations=3, verbose=verbose)
        return {"sft": sft_result, "grpo": grpo_result}

    # ---- Report / export ----

    def to_json(self, report: EGRMReport, path: str):
        data = {
            "n_samples": report.n_samples,
            "direct_ratio": report.direct_ratio,
            "avg_latency_ms": report.avg_latency_ms,
            "avg_tokens_per_sample": report.avg_tokens_per_sample,
            "avg_consensus": report.avg_consensus,
            "cot_trigger_rate": report.cot_trigger_rate,
            "total_tokens_saved": report.total_tokens_saved,
            "decisions": [
                {"route": d.route, "consensus": d.consensus, "latency_ms": d.latency_ms,
                 "tokens": d.tokens_used, "answer": d.answer[:80]}
                for d in report.decisions[:200]  # cap for size
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def print_summary(self, report: EGRMReport):
        print("\n" + "=" * 50)
        print("E-GRM Summary")
        print("=" * 50)
        print(f"\n  Samples: {report.n_samples}")
        print(f"  Direct (no CoT): {report.direct_ratio * 100:.1f}%")
        print(f"  CoT triggered:   {report.cot_trigger_rate * 100:.1f}%")
        print(f"\n  Avg consensus:   {report.avg_consensus:.3f}")
        print(f"  Avg latency:     {report.avg_latency_ms:.1f} ms")
        print(f"  Avg tokens:      {report.avg_tokens_per_sample:.0f}")
        print(f"  Tokens saved:    ~{report.total_tokens_saved}")
        print("=" * 50)
