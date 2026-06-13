"""Allowed-vocabulary constrained decoding for causal language models."""

from __future__ import annotations

from typing import List

import torch
from jaxtyping import Int
from transformers import AutoModelForCausalLM, AutoTokenizer


class TrieNode:
    """Trie node for tokenized constraint words."""

    def __init__(self) -> None:
        self.children: dict[int, "TrieNode"] = {}
        self.is_end = False


class Trie:
    """Prefix tree over tokenized allowed words."""

    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, tokenized_word: List[int]) -> None:
        if not tokenized_word:
            return

        node = self.root
        for token in tokenized_word:
            node = node.children.setdefault(token, TrieNode())
        node.is_end = True

    def get_valid_next_tokens(self, token_seq: List[int]) -> List[int]:
        node = self.root
        for token in token_seq:
            if token not in node.children:
                return []
            node = node.children[token]
        return list(node.children.keys())

    def is_complete_word(self, token_seq: List[int]) -> bool:
        node = self.root
        for token in token_seq:
            if token not in node.children:
                return False
            node = node.children[token]
        return node.is_end


class ConstrainedTextGenerator:
    """Generate only tokens that continue one of the supplied allowed words.

    This is stricter than "include these words somewhere." It is best framed as
    allowed-vocabulary constrained generation.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        eos_id: int,
        max_output_len: int = 10,
    ) -> None:
        if max_output_len <= 0:
            raise ValueError("max_output_len must be positive")

        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_id = eos_id
        self.max_output_len = max_output_len

    def __call__(
        self,
        input_ids: Int[torch.Tensor, "batch in_seq_len"],
        word_list: List[str],
    ) -> Int[torch.Tensor, "out_seq_len"]:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("ConstrainedTextGenerator expects input_ids with shape (1, seq_len)")

        device = input_ids.device
        trie = Trie()
        for word in word_list:
            tokenized_word = self.tokenizer.encode(word, add_special_tokens=False)
            trie.insert(tokenized_word)

        current_tokens = input_ids.clone()
        generated_tokens: list[int] = []
        partial_word: list[int] = []

        for _ in range(self.max_output_len):
            valid_tokens = trie.get_valid_next_tokens(partial_word)
            if not valid_tokens:
                break

            with torch.no_grad():
                output = self.model(current_tokens)
                logits = output["logits"] if isinstance(output, dict) else output.logits
                next_token_logits = logits[:, -1, :]
                masked_logits = torch.full_like(next_token_logits, float("-inf"))
                masked_logits[0, valid_tokens] = next_token_logits[0, valid_tokens]
                next_token = torch.argmax(masked_logits, dim=-1)

            token_id = int(next_token.item())
            if token_id == self.eos_token_id:
                break

            partial_word.append(token_id)
            generated_tokens.append(token_id)

            if trie.is_complete_word(partial_word):
                partial_word = []

            current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0)], dim=1)

        return torch.tensor(generated_tokens, dtype=torch.long, device=device)
