"""Small, dependency-free decoding helpers.

The original experiment scripts use PyTorch tensors and Hugging Face models.
These helpers keep the core decoding policies easy to test and explain.
"""

from __future__ import annotations

import math
import random
from typing import Mapping, MutableSequence, Sequence, Tuple


TokenScore = Tuple[str, float]
Distribution = Sequence[Tuple[str, float]]


def sorted_logits(logits: Mapping[str, float]) -> list[TokenScore]:
    """Return logits sorted from highest to lowest score."""
    if not logits:
        raise ValueError("logits must contain at least one token")
    return sorted(logits.items(), key=lambda item: item[1], reverse=True)


def softmax(scores: Sequence[TokenScore], temperature: float = 1.0) -> list[TokenScore]:
    """Convert token scores into a probability distribution."""
    if not scores:
        raise ValueError("scores must contain at least one token")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    scaled = [(token, score / temperature) for token, score in scores]
    max_score = max(score for _, score in scaled)
    exp_scores = [(token, math.exp(score - max_score)) for token, score in scaled]
    total = sum(score for _, score in exp_scores)
    return [(token, score / total) for token, score in exp_scores]


def greedy_token(logits: Mapping[str, float]) -> str:
    """Pick the highest-logit token."""
    return sorted_logits(logits)[0][0]


def top_k_candidates(logits: Mapping[str, float], k: int) -> list[TokenScore]:
    """Keep only the top-k tokens by logit score."""
    if k <= 0:
        raise ValueError("k must be positive")
    return sorted_logits(logits)[:k]


def nucleus_candidates(
    logits: Mapping[str, float],
    p: float,
    temperature: float = 1.0,
) -> list[TokenScore]:
    """Return the smallest high-probability set whose mass is at least p."""
    if not 0 < p <= 1:
        raise ValueError("p must be in the interval (0, 1]")

    distribution = softmax(sorted_logits(logits), temperature=temperature)
    selected: MutableSequence[TokenScore] = []
    cumulative = 0.0

    for token, probability in distribution:
        selected.append((token, probability))
        cumulative += probability
        if cumulative >= p:
            break

    return list(selected)


def sample_from_distribution(distribution: Distribution, rng: random.Random) -> str:
    """Sample one token from a normalized distribution."""
    if not distribution:
        raise ValueError("distribution must contain at least one token")

    threshold = rng.random()
    cumulative = 0.0
    for token, probability in distribution:
        if probability < 0:
            raise ValueError("probabilities must be non-negative")
        cumulative += probability
        if threshold <= cumulative:
            return token

    return distribution[-1][0]


def _renormalize(distribution: Distribution) -> list[TokenScore]:
    total = sum(probability for _, probability in distribution)
    if total <= 0:
        raise ValueError("distribution must have positive probability mass")
    return [(token, probability / total) for token, probability in distribution]


def select_token(
    logits: Mapping[str, float],
    strategy: str,
    rng: random.Random | None = None,
    temperature: float = 1.0,
    top_k: int = 10,
    top_p: float = 0.9,
) -> str:
    """Select a token according to a decoding strategy."""
    rng = rng or random.Random()

    if strategy == "greedy":
        return greedy_token(logits)

    if strategy == "random":
        return sample_from_distribution(softmax(sorted_logits(logits), temperature), rng)

    if strategy == "topk":
        candidates = top_k_candidates(logits, top_k)
        return sample_from_distribution(softmax(candidates, temperature), rng)

    if strategy == "nucleus":
        candidates = nucleus_candidates(logits, top_p, temperature)
        return sample_from_distribution(_renormalize(candidates), rng)

    raise ValueError(f"unknown strategy: {strategy}")
