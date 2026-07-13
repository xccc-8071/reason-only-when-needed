"""
Dynamic CoT Router
------------------
Three routing paths:
  Route 1 -- Consensus high → direct output, no CoT
  Route 2 -- Consensus low → CoT + discriminative scoring
  Route 3 -- Very low consensus → multi-step CoT + self-consistency

Training: SFT (mixed short / long CoT) → GRPO pairwise preference.
"""

from __future__ import annotations

import random
import time
import numpy as np

from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field

from egrm_uncertainty import (
    UncertaintyEstimator, UncertaintyReport, ParallelDecoder, DecodingResult,
)
from egrm_scorer import DiscriminativeScorer, ReasoningPath, ScoredPath


@dataclass
class RouterDecision:
    route: str                                    # "direct" | "cot" | "cot_multi"
    consensus: float
    uncertainty: float
    latency_ms: float
    tokens_used: int
    answer: str
    score: Optional[float] = None
    reasoning_paths: List[ScoredPath] = field(default_factory=list)


@dataclass
class SFTExample:
    """Mixed-mode training sample: short answer + long CoT."""
    prompt: str
    direct_answer: str        # short-form, used for Route 1 style
    cot_answer: str           # full reasoning chain
    final_answer: str         # ground-truth answer
    category: str = "general"


@dataclass
class GRPOPair:
    """Pairwise preference data for GRPO training."""
    prompt: str
    chosen_path: ReasoningPath
    rejected_path: ReasoningPath
    chosen_score: float
    rejected_score: float


class CoTRouter:
    """
    Main router: decision → generation → scoring.

    Training pipeline: SFT → GRPO preference optimization.
    Inference: uncertainty → route → (direct | CoT+score).
    """

    def __init__(
        self,
        consensus_threshold: float = 0.8,
        num_parallel: int = 5,
        num_cot_paths: int = 4,
        scorer: Optional[DiscriminativeScorer] = None,
    ):
        self.threshold = consensus_threshold
        self.num_parallel = num_parallel
        self.num_cot_paths = num_cot_paths
        self.decoder = ParallelDecoder(num_runs=num_parallel)
        self.estimator = UncertaintyEstimator(consensus_threshold=consensus_threshold)
        self.scorer = scorer or DiscriminativeScorer()

    def decide(
        self,
        prompt: str,
        direct_model_fn: Callable,   # (prompt, temp, top_p, max_tokens) -> str
        cot_model_fn: Optional[Callable] = None,  # (prompt) -> str (with CoT prompt)
        max_tokens: int = 128,
    ) -> RouterDecision:
        """
        Full inference: probe uncertainty → route → generate → score.
        """
        t0 = time.time()

        # Phase 1: parallel decoding → uncertainty estimate
        decodings = self.decoder.decode(prompt, direct_model_fn, max_tokens)
        report = self.estimator.estimate(decodings)

        if not report.requires_cot:
            # Route 1: consensus → direct answer
            elapsed = (time.time() - t0) * 1000
            return RouterDecision(
                route="direct",
                consensus=report.consensus,
                uncertainty=report.entropy,
                latency_ms=round(elapsed, 1),
                tokens_used=sum(d.tokens_generated for d in decodings),
                answer=report.dominant_answer,
            )

        # Route 2: CoT generation + scoring
        if cot_model_fn is None:
            cot_model_fn = direct_model_fn  # fallback

        paths = self._generate_cot_paths(prompt, cot_model_fn)
        scored = self.scorer.score_paths(paths)
        best = scored[0] if scored else None
        elapsed = (time.time() - t0) * 1000

        total_tokens = sum(d.tokens_generated for d in decodings)
        total_tokens += sum(p.tokens_generated for p in paths)

        return RouterDecision(
            route="cot",
            consensus=report.consensus,
            uncertainty=report.entropy,
            latency_ms=round(elapsed, 1),
            tokens_used=total_tokens,
            answer=best.path.final_answer if best else "",
            score=best.score if best else None,
            reasoning_paths=scored,
        )

    def _generate_cot_paths(self, prompt: str, model_fn: Callable) -> List[ReasoningPath]:
        """Generate multiple CoT reasoning paths."""
        cot_prompt = prompt + "\nLet's think step by step."
        paths = []
        rng = random.Random(42)
        for i in range(self.num_cot_paths):
            temp = 0.3 + 0.4 * rng.random()
            cot_text = model_fn(cot_prompt, temperature=temp, top_p=0.95, max_tokens=256)
            # Parse final answer after the "Answer:" marker
            lines = cot_text.split("\n")
            final = lines[-1] if lines else cot_text
            if "Answer:" in final:
                final = final.split("Answer:")[-1].strip()
            paths.append(ReasoningPath(
                path_id=f"cot_{i}",
                cot_text=cot_text,
                final_answer=final,
                tokens_generated=len(cot_text.split()),
            ))
        return paths

    # ---- Training helpers ----

    def prepare_sft_data(
        self,
        samples: List[SFTExample],
        mix_ratio: float = 0.5,
    ) -> List[Dict]:
        """
        Mixed training set: ratio * short + (1-ratio) * long-CoT.
        """
        rng = random.Random(42)
        data = []
        for s in samples:
            if rng.random() < mix_ratio:
                data.append({
                    "prompt": s.prompt,
                    "completion": s.direct_answer,
                    "mode": "short",
                })
            else:
                data.append({
                    "prompt": s.prompt + "\nLet's think step by step.",
                    "completion": s.cot_answer,
                    "mode": "cot",
                })
        return data

    def prepare_grpo_pairs(
        self,
        prompts: List[str],
        generate_paths_fn: Callable[..., List[ReasoningPath]],
    ) -> List[GRPOPair]:
        """Generate pairwise preference data for GRPO training."""
        pairs = []
        for prompt in prompts:
            paths = generate_paths_fn(prompt)
            if len(paths) < 2:
                continue
            scored = self.scorer.score_paths(paths)
            for i in range(len(scored) - 1):
                for j in range(i + 1, len(scored)):
                    if scored[i].score > scored[j].score:
                        pairs.append(GRPOPair(
                            prompt=prompt,
                            chosen_path=scored[i].path,
                            rejected_path=scored[j].path,
                            chosen_score=scored[i].score,
                            rejected_score=scored[j].score,
                        ))
        return pairs
