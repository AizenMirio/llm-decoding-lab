"""PyTorch decoding helpers for causal language models.

These classes assume a single prompt per call because the original benchmark
evaluated one translation example at a time.
"""

from __future__ import annotations

from typing import Callable

import torch
from jaxtyping import Int
from transformers import AutoModelForCausalLM


class TextGenerator:
    """Generate tokens from a causal LM with common decoding strategies."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        decoding_strategy: str,
        eos_id: int,
        max_output_len: int = 10,
        tau: float = 1.0,
        k: int = 10,
        p: float = 0.5,
    ) -> None:
        self.model = model
        self.decoding_strategy = decoding_strategy
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        self.tau = tau
        self.k = k
        self.p = p

        if max_output_len <= 0:
            raise ValueError("max_output_len must be positive")
        if tau <= 0:
            raise ValueError("tau must be positive")
        if k <= 0:
            raise ValueError("k must be positive")
        if not 0 < p <= 1:
            raise ValueError("p must be in the interval (0, 1]")

        generators: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
            "greedy": self.greedy_decoding,
            "random": self.random_sampling,
            "topk": self.topk_sampling,
            "nucleus": self.nucleus_sampling,
        }
        if decoding_strategy not in generators:
            raise ValueError(f"unknown decoding strategy: {decoding_strategy}")
        self.generator_func = generators[decoding_strategy]

    def __call__(
        self,
        input_ids: Int[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "out_seq_len"]:
        return self.generator_func(input_ids)

    def _validate_single_batch(self, input_ids: torch.Tensor) -> None:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("TextGenerator expects input_ids with shape (1, seq_len)")

    def _next_logits(self, current_input: torch.Tensor) -> torch.Tensor:
        output = self.model(current_input)
        return output.logits[:, -1, :]

    @torch.no_grad()
    def greedy_decoding(
        self,
        input_ids: Int[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "out_seq_len"]:
        self._validate_single_batch(input_ids)
        generated_tokens: list[int] = []
        current_input = input_ids.clone()

        for _ in range(self.max_output_len):
            next_token = torch.argmax(self._next_logits(current_input), dim=-1)
            token_id = int(next_token.item())
            if token_id == self.eos_token_id:
                break

            generated_tokens.append(token_id)
            current_input = torch.cat([current_input, next_token.unsqueeze(1)], dim=1)

        return torch.tensor(generated_tokens, dtype=torch.long, device=input_ids.device)

    @torch.no_grad()
    def random_sampling(
        self,
        input_ids: Int[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "out_seq_len"]:
        self._validate_single_batch(input_ids)
        generated_tokens: list[int] = []
        current_input = input_ids.clone()

        for _ in range(self.max_output_len):
            probabilities = torch.softmax(self._next_logits(current_input) / self.tau, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            token_id = int(next_token.item())
            if token_id == self.eos_token_id:
                break

            generated_tokens.append(token_id)
            current_input = torch.cat([current_input, next_token], dim=1)

        return torch.tensor(generated_tokens, dtype=torch.long, device=input_ids.device)

    @torch.no_grad()
    def topk_sampling(
        self,
        input_ids: Int[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "out_seq_len"]:
        self._validate_single_batch(input_ids)
        generated_tokens: list[int] = []
        current_input = input_ids.clone()

        for _ in range(self.max_output_len):
            next_token_logits = self._next_logits(current_input)
            effective_k = min(self.k, next_token_logits.shape[-1])
            top_k_values, top_k_indices = torch.topk(next_token_logits, effective_k, dim=-1)
            probabilities = torch.softmax(top_k_values / self.tau, dim=-1)
            next_token_idx = torch.multinomial(probabilities, num_samples=1)
            next_token = torch.gather(top_k_indices, dim=-1, index=next_token_idx)
            token_id = int(next_token.item())
            if token_id == self.eos_token_id:
                break

            generated_tokens.append(token_id)
            current_input = torch.cat([current_input, next_token], dim=1)

        return torch.tensor(generated_tokens, dtype=torch.long, device=input_ids.device)

    @torch.no_grad()
    def nucleus_sampling(
        self,
        input_ids: Int[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "out_seq_len"]:
        self._validate_single_batch(input_ids)
        generated_tokens: list[int] = []
        current_input = input_ids.clone()

        for _ in range(self.max_output_len):
            sorted_logits, sorted_indices = torch.sort(self._next_logits(current_input), descending=True, dim=-1)
            sorted_probabilities = torch.softmax(sorted_logits / self.tau, dim=-1)
            cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)

            remove_mask = cumulative_probabilities > self.p
            remove_mask[..., 1:] = remove_mask[..., :-1].clone()
            remove_mask[..., 0] = False

            filtered_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
            probabilities = torch.softmax(filtered_logits / self.tau, dim=-1)
            next_sorted_idx = torch.multinomial(probabilities, num_samples=1)
            next_token = sorted_indices.gather(dim=-1, index=next_sorted_idx)
            token_id = int(next_token.item())
            if token_id == self.eos_token_id:
                break

            generated_tokens.append(token_id)
            current_input = torch.cat([current_input, next_token], dim=1)

        return torch.tensor(generated_tokens, dtype=torch.long, device=input_ids.device)
