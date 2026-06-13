"""Medusa-style decoding helpers."""

from __future__ import annotations

from typing import Callable

import torch
from jaxtyping import Int
from transformers import AutoModelForCausalLM


class MedusaTextGenerator:
    """Single-head and multi-head decoding for Medusa-style models."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        decoding_strategy: str,
        eos_id: int,
        use_no_medusa_heads: int = 5,
        beam_width: int = 2,
        max_output_len: int = 10,
    ) -> None:
        if max_output_len <= 0:
            raise ValueError("max_output_len must be positive")
        if beam_width <= 0:
            raise ValueError("beam_width must be positive")
        if not 0 <= use_no_medusa_heads <= 5:
            raise ValueError("use_no_medusa_heads must be between 0 and 5")

        self.model = model
        self.decoding_strategy = decoding_strategy
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        self.beam_width = beam_width
        self.no_heads = use_no_medusa_heads + 1

        generators: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
            "single-head": self.single_head_decoding,
            "multi-head": self.multi_head_decoding,
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
            raise ValueError("MedusaTextGenerator expects input_ids with shape (1, seq_len)")

    @torch.no_grad()
    def single_head_decoding(
        self,
        input_ids: Int[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "out_seq_len"]:
        self._validate_single_batch(input_ids)
        current_input = input_ids.clone()
        generated_tokens: list[int] = []

        for _ in range(self.max_output_len):
            output = self.model(input_ids=current_input)
            next_token = torch.argmax(output.logits[:, -1, :], dim=-1)
            token_id = int(next_token.item())
            if token_id == self.eos_token_id:
                break

            generated_tokens.append(token_id)
            current_input = torch.cat([current_input, next_token.unsqueeze(0)], dim=1)

        return torch.tensor(generated_tokens, dtype=torch.long, device=input_ids.device)

    @torch.no_grad()
    def multi_head_decoding(
        self,
        input_ids: Int[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "out_seq_len"]:
        self._validate_single_batch(input_ids)
        current_input = input_ids.clone()
        generated_tokens: list[int] = []

        while len(generated_tokens) < self.max_output_len:
            outputs = self.model(input_ids=current_input)
            lm_logits = outputs.logits[:, -1, :]
            medusa_logits = outputs.medusa_logits

            logits_list = [lm_logits] + list(medusa_logits[: self.no_heads - 1])
            log_probs_list = [torch.log_softmax(logits.squeeze(0), dim=-1) for logits in logits_list]

            candidates: list[list[int]] = [[]]
            scores = [0.0]

            for log_probs in log_probs_list:
                new_candidates: list[list[int]] = []
                new_scores: list[float] = []
                topk = torch.topk(log_probs, min(self.beam_width, log_probs.shape[-1]))

                for current_seq, current_score in zip(candidates, scores):
                    for token_id, token_score in zip(topk.indices.tolist(), topk.values.tolist()):
                        new_candidates.append(current_seq + [int(token_id)])
                        new_scores.append(current_score + float(token_score))

                top_indices = torch.topk(
                    torch.tensor(new_scores, device=input_ids.device),
                    min(self.beam_width, len(new_scores)),
                ).indices.tolist()
                candidates = [new_candidates[i] for i in top_indices]
                scores = [new_scores[i] for i in top_indices]

            extension = candidates[0]
            if self.eos_token_id in extension:
                eos_idx = extension.index(self.eos_token_id)
                generated_tokens.extend(extension[:eos_idx])
                break

            remaining = self.max_output_len - len(generated_tokens)
            extension = extension[:remaining]
            generated_tokens.extend(extension)

            if not extension:
                break

            next_tokens = torch.tensor(extension, dtype=torch.long, device=input_ids.device).unsqueeze(0)
            current_input = torch.cat([current_input, next_tokens], dim=1)

        return torch.tensor(generated_tokens, dtype=torch.long, device=input_ids.device)
