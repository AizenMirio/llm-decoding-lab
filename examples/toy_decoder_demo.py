"""Small dependency-free demo of common LLM decoding strategies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_decoding_lab import decode  # noqa: E402


def strategies_from_arg(strategy: str) -> Iterable[str]:
    if strategy == "all":
        return ("greedy", "random", "topk", "nucleus")
    return (strategy,)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare toy decoding strategies.")
    parser.add_argument("--strategy", choices=["all", "greedy", "random", "topk", "nucleus"], default="all")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--top-p", type=float, default=0.85)
    args = parser.parse_args()

    for strategy in strategies_from_arg(args.strategy):
        tokens = decode(
            strategy=strategy,
            steps=args.steps,
            seed=args.seed,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print(f"{strategy:<8}: {' '.join(tokens)}")


if __name__ == "__main__":
    main()
