"""
Discriminative Scoring Module
------------------------------
Lightweight scorer trained with a hybrid regression--ranking objective.
Replaces naive voting-based aggregation with fine-grained path evaluation.

Hybrid loss:
  L_hybrid = L_huber(y_true, y_pred) + λ * L_hinge(pairs)
    - Huber loss: robust regression for absolute reward values
    - Hinge loss: pairwise ranking for relative quality ordering

"""

from __future__ import annotations

import random
import numpy as np

from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class ReasoningPath:
    """A single CoT reasoning trajectory."""
    path_id: str
    cot_text: str            # full reasoning chain
    final_answer: str
    tokens_generated: int
    features: Optional[List[float]] = None   # extracted feature vector


@dataclass
class ScoredPath:
    path: ReasoningPath
    score: float              # predicted reward score
    rank: int                 # rank among all paths for this prompt


@dataclass(order=True)
class TrainingSample:
    """Pairwise training sample for the hybrid loss."""
    idx: int = field(compare=False)
    prompt: str = field(compare=False)
    path_a: ScoredPath = field(compare=False)
    path_b: ScoredPath = field(compare=False)
    label_a: float = field(compare=False)       # ground-truth score A
    label_b: float = field(compare=False)       # ground-truth score B
    # A > B by label
    is_preference: bool = field(compare=False)  # True for ranking, False for regression-only

    def __init__(self, idx: int, prompt: str, path_a: ScoredPath, path_b: ScoredPath,
                 label_a: float, label_b: float, is_preference: bool = True):
        self.idx = idx
        self.prompt = prompt
        self.path_a = path_a
        self.path_b = path_b
        self.label_a = label_a
        self.label_b = label_b
        self.is_preference = is_preference


class HybridLoss:
    """
    Huber loss + Hinge ranking loss.

    L = (1 - α) * Huber(y_pred_a, y_true_a) + (1 - α) * Huber(y_pred_b, y_true_b)
      + α * Hinge(margin − (y_pred_a − y_pred_b))

    Default: δ = 1.0 (Huber), margin = 0.1, α = 0.3
    """

    def __init__(
        self,
        huber_delta: float = 1.0,
        hinge_margin: float = 0.1,
        alpha: float = 0.3,     # weight between regression and ranking
    ):
        self.huber_delta = huber_delta
        self.hinge_margin = hinge_margin
        self.alpha = alpha

    def compute(
        self,
        pred_scores: List[float],
        true_scores: List[float],
        pairs: Optional[List[Tuple[int, int, bool]]] = None,
    ) -> Dict[str, float]:
        """
        pairs: list of (idx_a, idx_b, label_a > label_b) for pairwise ranking.
        """
        n = len(pred_scores)

        # Huber regression loss
        huber_total = 0.0
        for yp, yt in zip(pred_scores, true_scores):
            diff = abs(yp - yt)
            if diff <= self.huber_delta:
                huber_total += 0.5 * diff ** 2
            else:
                huber_total += self.huber_delta * (diff - 0.5 * self.huber_delta)
        huber_loss = huber_total / max(n, 1)

        # Hinge ranking loss
        hinge_total = 0.0
        if pairs:
            for i, j, a_better_than_b in pairs:
                if a_better_than_b:
                    hinge_total += max(0.0, self.hinge_margin - (pred_scores[i] - pred_scores[j]))
                else:
                    hinge_total += max(0.0, self.hinge_margin - (pred_scores[j] - pred_scores[i]))
            hinge_loss = hinge_total / len(pairs)
        else:
            hinge_loss = 0.0

        total = (1 - self.alpha) * huber_loss + self.alpha * hinge_loss
        return {
            "total": round(total, 6),
            "huber": round(huber_loss, 6),
            "hinge": round(hinge_loss, 6),
            "alpha": self.alpha,
        }


class DiscriminativeScorer:
    """
    Lightweight scorer for reasoning path evaluation.
    Uses hybrid loss: regression + ranking.

    The scorer estimates a reward score r in [0, 1] for each
    generated reasoning path based on extracted features.
    """

    def __init__(
        self,
        feature_dim: int = 16,
        hybrid_loss: Optional[HybridLoss] = None,
    ):
        # A small FFN (Linear → ReLU → Linear → Sigmoid) in production.
        # Dry-run uses random-projection + logistic proxy.
        self.feature_dim = feature_dim
        self.weights = np.random.randn(feature_dim) * 0.1
        self.bias = 0.0
        self.hybrid_loss = hybrid_loss or HybridLoss(alpha=0.3)

    def extract_features(self, path: ReasoningPath) -> np.ndarray:
        """Extract handcrafted or learned features from a reasoning path."""
        features = np.zeros(self.feature_dim)
        features[0] = len(path.cot_text) / 1000.0           # length
        features[1] = len(path.cot_text.split()) / 200.0    # token count
        features[2] = len(path.final_answer) / 100.0        # answer length
        features[3] = float(path.tokens_generated) / 500.0  # gen tokens
        # Structural features from the reasoning text
        features[4] = 0.5 if "step" in path.cot_text.lower() else -0.3
        features[5] = 0.3 if "therefore" in path.cot_text.lower() else -0.1
        features[6:] = np.random.randn(self.feature_dim - 6) * 0.05
        return features

    def score(self, path: ReasoningPath) -> float:
        """Predict reward score for a single reasoning path."""
        feats = self.extract_features(path)
        if path.features is not None and len(path.features) >= self.feature_dim:
            feats[:self.feature_dim] = path.features[:self.feature_dim]
        logit = float(np.dot(self.weights, feats) + self.bias)
        return 1.0 / (1.0 + np.exp(-logit))  # sigmoid

    def score_paths(self, paths: List[ReasoningPath]) -> List[ScoredPath]:
        """Score and rank multiple reasoning paths."""
        scores = [self.score(p) for p in paths]
        order = np.argsort(scores)[::-1]
        ranked: List[ScoredPath] = []
        for rank, idx in enumerate(order):
            ranked.append(ScoredPath(path=paths[idx], score=scores[idx], rank=rank + 1))
        return ranked

    def predict_ranks(self, paths: List[ReasoningPath]) -> List[int]:
        """Return predicted ranks (0 = best)."""
        scores = [self.score(p) for p in paths]
        order = np.argsort(scores)[::-1]
        ranks = [0] * len(paths)
        for i, idx in enumerate(order):
            ranks[idx] = i
        return ranks

    # ---- Dry-run training step ----

    def train_step(
        self,
        pred_scores: List[float],
        true_scores: List[float],
        pairs: Optional[List[Tuple[int, int, bool]]] = None,
        lr: float = 0.01,
    ) -> Dict[str, float]:
        """Single training step with hybrid loss. Returns loss components."""
        loss_info = self.hybrid_loss.compute(pred_scores, true_scores, pairs)
        # Per-parameter gradient step (proxy for real autograd)
        for i, (yp, yt) in enumerate(zip(pred_scores, true_scores)):
            err = yt - yp
            self.weights += lr * err * np.random.randn(self.feature_dim) * 0.1
            self.bias += lr * err * 0.01
        self.weights = np.clip(self.weights, -3.0, 3.0)
        return loss_info
