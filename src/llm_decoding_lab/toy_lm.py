"""A tiny toy language model used by the local decoding demo."""

from __future__ import annotations

import random
from typing import Mapping, Sequence

from .sampling import select_token


EOS_TOKEN = "<eos>"


class ToyLanguageModel:
    """A context-dependent next-token scorer for demonstration only."""

    def next_logits(self, context: Sequence[str]) -> Mapping[str, float]:
        last = context[-1] if context else ""
        default = {
            EOS_TOKEN: -2.0,
            "the": 1.7,
            "agent": 1.2,
            "model": 1.1,
            "learns": 0.4,
            "samples": 0.3,
            "optimizes": 0.2,
            "a": 0.7,
            "policy": 0.1,
            "with": -0.1,
            "rewards": -0.3,
            "constraints": -0.4,
            "diverse": -0.2,
            "actions": -0.5,
            "quickly": -0.6,
            "carefully": -0.7,
        }

        transitions = {
            "the": {"agent": 2.3, "model": 2.1, "policy": 1.1},
            "agent": {"learns": 2.0, "samples": 1.5, "optimizes": 1.4},
            "model": {"samples": 2.0, "learns": 1.4, "optimizes": 1.2},
            "learns": {"a": 1.8, "with": 1.2, "quickly": 0.8},
            "samples": {"diverse": 1.9, "actions": 1.5, "carefully": 0.9},
            "optimizes": {"a": 1.8, "rewards": 1.5, "constraints": 1.1},
            "a": {"policy": 2.2, "model": 0.8},
            "policy": {"with": 1.8, EOS_TOKEN: 1.0},
            "with": {"rewards": 1.8, "constraints": 1.6},
            "rewards": {EOS_TOKEN: 2.4},
            "constraints": {EOS_TOKEN: 2.3, "carefully": 1.0},
            "diverse": {"actions": 2.0, "samples": 0.5},
            "actions": {EOS_TOKEN: 2.2, "quickly": 0.9},
            "quickly": {EOS_TOKEN: 2.1},
            "carefully": {EOS_TOKEN: 2.1},
        }

        logits = default.copy()
        logits.update(transitions.get(last, {}))
        return logits


def decode(
    strategy: str,
    steps: int = 12,
    seed: int = 7,
    temperature: float = 0.8,
    top_k: int = 4,
    top_p: float = 0.85,
) -> list[str]:
    """Generate tokens from the toy model using a selected strategy."""
    if steps <= 0:
        raise ValueError("steps must be positive")

    rng = random.Random(seed)
    model = ToyLanguageModel()
    generated: list[str] = []

    for _ in range(steps):
        token = select_token(
            logits=model.next_logits(generated),
            strategy=strategy,
            rng=rng,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        generated.append(token)
        if token == EOS_TOKEN:
            break

    return generated
