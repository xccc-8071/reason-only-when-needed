"""
Uncertainty Estimation via Parallel Decoding Convergence
---------------------------------------------------------
Core mechanism of E-GRM: M parallel decoding runs with varied sampling
hyperparameters (temperatures, nucleus p). High consensus → simple input
(direct answer). Low consensus → high uncertainty → trigger CoT reasoning.

Key metric: Consensus(x) = max_y Count(y) / M
  τ = 0.8  →  ~58% of samples skip CoT
  τ = 0.6  →  ~72% of samples skip CoT (more aggressive)

"""

from __future__ import annotations

import random
import numpy as np

from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class DecodingResult:
    answer: str
    confidence: float       # logit-based or generation probability
    temperature: float
    tokens_generated: int   # for cost tracking


@dataclass
class UncertaintyReport:
    consensus: float         # max agreement ratio
    dominant_answer: str
    answer_distribution: Dict[str, int]
    num_runs: int
    entropy: float           # normalized entropy of answer distribution
    requires_cot: bool
    tokens_saved: int         # estimated tokens saved by skipping CoT


class ParallelDecoder:
    """Run M parallel decodings with varied sampling params."""

    DEFAULT_TEMPERATURES = [0.0, 0.3, 0.7, 1.0]
    DEFAULT_NUCLEUS = [0.9, 1.0]
    DEFAULT_NUM_RUNS = 5          # M in the paper

    def __init__(
        self,
        num_runs: int = DEFAULT_NUM_RUNS,
        temperatures: Optional[List[float]] = None,
        nucleus_values: Optional[List[float]] = None,
    ):
        self.num_runs = num_runs
        self.temperatures = temperatures or self.DEFAULT_TEMPERATURES
        self.nucleus_values = nucleus_values or self.DEFAULT_NUCLEUS

    def decode(
        self,
        prompt: str,
        model_fn: Callable,   # (prompt, temperature, top_p) -> str
        max_tokens: int = 128,
    ) -> List[DecodingResult]:
        """Run parallel decodings and return raw results."""
        results = []
        rng = random.Random(42)
        for i in range(self.num_runs):
            temp = self.temperatures[i % len(self.temperatures)]
            top_p = self.nucleus_values[i % len(self.nucleus_values)]
            answer = model_fn(prompt, temperature=temp, top_p=top_p, max_tokens=max_tokens)
            results.append(DecodingResult(
                answer=answer.strip(),
                confidence=0.5 + 0.4 * rng.random(),  # reads logits in real inference
                temperature=temp,
                tokens_generated=len(answer.split()),
            ))
        return results


class UncertaintyEstimator:
    """
    Estimate uncertainty from parallel decoding convergence.

    High consensus → likely simple, route direct.
    Low consensus → complex, trigger CoT.
    """

    def __init__(self, consensus_threshold: float = 0.8):
        self.threshold = consensus_threshold

    def _normalize(self, answer: str) -> str:
        """Strip whitespace, lowercase, remove trailing punctuation."""
        return answer.strip().lower().rstrip(".,;:\"'")

    def estimate(self, results: List[DecodingResult]) -> UncertaintyReport:
        """Analyze parallel decoding results and decide CoT routing."""
        normalized = [self._normalize(r.answer) for r in results]
        counter = Counter(normalized)
        m = len(results)
        dominant, dominant_count = counter.most_common(1)[0]
        consensus = dominant_count / m

        # Normalized entropy: 0 = all agree, 1 = all disagree
        probs = [c / m for c in counter.values()]
        max_entropy = np.log(m)
        entropy = -sum(p * np.log(p) for p in probs if p > 0) / max_entropy if max_entropy > 0 else 0.0

        requires_cot = consensus < self.threshold

        # Estimated token savings: if we skip CoT (~200 tokens/runs)
        cot_cost = 200  # avg CoT tokens per run
        tokens_saved = 0
        if not requires_cot:
            tokens_saved = cot_cost * m

        return UncertaintyReport(
            consensus=round(consensus, 4),
            dominant_answer=dominant,
            answer_distribution=dict(counter),
            num_runs=m,
            entropy=round(entropy, 4),
            requires_cot=requires_cot,
            tokens_saved=tokens_saved,
        )


def calibrate_threshold(
    results_by_sample: List[Tuple[bool, float]],  # (is_complex, consensus)
    target_recall: float = 0.95,
) -> float:
    """
    Calibrate consensus threshold τ to achieve target recall
    on a held-out set where ground-truth complexity is known.
    """
    thresholds = np.linspace(0.0, 1.0, 101)
    best_threshold = 0.8
    best_recall = 0.0
    for t in thresholds:
        # recall = how many complex samples correctly trigger CoT
        recall = sum(1 for is_c, c in results_by_sample if is_c and c < t)
        total_complex = sum(1 for is_c, _ in results_by_sample if is_c)
        if total_complex == 0:
            continue
        r = recall / total_complex
        if r >= target_recall and t > best_threshold:
            best_threshold = t
            best_recall = r
    return round(best_threshold, 2)
