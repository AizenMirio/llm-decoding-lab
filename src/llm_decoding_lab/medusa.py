"""Dependency-free helpers for reasoning about Medusa-style candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .sampling import TokenScore


@dataclass(frozen=True)
class Candidate:
    """A candidate multi-token extension."""

    tokens: tuple[str, ...]
    score: float


def build_medusa_candidates(
    head_log_probs: Sequence[Sequence[TokenScore]],
    beam_width: int,
) -> list[Candidate]:
    """Build beam candidates from per-head token log probabilities.

    Each head proposes the token for one future position. The candidate state is
    only the proposed extension, not the already generated prefix. This avoids
    duplicating committed tokens when the extension is appended to the prompt.
    """
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    if not head_log_probs:
        raise ValueError("head_log_probs must contain at least one head")

    candidates = [Candidate(tokens=(), score=0.0)]

    for head in head_log_probs:
        if not head:
            raise ValueError("each head must contain at least one token score")

        ranked_head = sorted(head, key=lambda item: item[1], reverse=True)[:beam_width]
        expanded: list[Candidate] = []
        for candidate in candidates:
            for token, log_probability in ranked_head:
                expanded.append(
                    Candidate(
                        tokens=candidate.tokens + (token,),
                        score=candidate.score + log_probability,
                    )
                )

        candidates = sorted(expanded, key=lambda candidate: candidate.score, reverse=True)[:beam_width]

    return candidates


def truncate_at_eos(tokens: Sequence[str], eos_token: str) -> tuple[str, ...]:
    """Return tokens before EOS, dropping EOS and anything after it."""
    if eos_token not in tokens:
        return tuple(tokens)
    return tuple(tokens[: tokens.index(eos_token)])
