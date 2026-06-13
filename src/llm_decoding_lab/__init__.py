"""Reusable decoding primitives for the LLM Decoding Lab."""

from .sampling import (
    Distribution,
    TokenScore,
    greedy_token,
    nucleus_candidates,
    sample_from_distribution,
    select_token,
    softmax,
    sorted_logits,
    top_k_candidates,
)
from .medusa import Candidate, build_medusa_candidates, truncate_at_eos
from .toy_lm import ToyLanguageModel, decode
from .trie import Trie

__all__ = [
    "Candidate",
    "Distribution",
    "TokenScore",
    "Trie",
    "ToyLanguageModel",
    "build_medusa_candidates",
    "decode",
    "greedy_token",
    "nucleus_candidates",
    "sample_from_distribution",
    "select_token",
    "softmax",
    "sorted_logits",
    "top_k_candidates",
    "truncate_at_eos",
]
