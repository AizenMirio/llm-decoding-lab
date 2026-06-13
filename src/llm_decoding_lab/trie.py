"""Trie utilities for constrained decoding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Iterable, Sequence


Token = Hashable


@dataclass
class _TrieNode:
    children: dict[Token, "_TrieNode"] = field(default_factory=dict)
    is_end: bool = False


class Trie:
    """A prefix tree over token sequences."""

    def __init__(self) -> None:
        self.root = _TrieNode()

    @classmethod
    def from_sequences(cls, sequences: Iterable[Sequence[Token]]) -> "Trie":
        trie = cls()
        for sequence in sequences:
            trie.insert(sequence)
        return trie

    def insert(self, sequence: Sequence[Token]) -> None:
        if not sequence:
            raise ValueError("cannot insert an empty sequence")

        node = self.root
        for token in sequence:
            node = node.children.setdefault(token, _TrieNode())
        node.is_end = True

    def valid_next_tokens(self, prefix: Sequence[Token]) -> list[Token]:
        node = self._walk(prefix)
        if node is None:
            return []
        return list(node.children.keys())

    def is_complete(self, sequence: Sequence[Token]) -> bool:
        node = self._walk(sequence)
        return bool(node and node.is_end)

    def is_prefix(self, sequence: Sequence[Token]) -> bool:
        return self._walk(sequence) is not None

    def _walk(self, sequence: Sequence[Token]) -> _TrieNode | None:
        node = self.root
        for token in sequence:
            if token not in node.children:
                return None
            node = node.children[token]
        return node
