import unittest

from llm_decoding_lab.trie import Trie


class TrieTests(unittest.TestCase):
    def test_valid_next_tokens_for_prefix(self):
        trie = Trie.from_sequences([[1, 2, 3], [1, 2, 4], [5]])
        self.assertEqual(set(trie.valid_next_tokens([1, 2])), {3, 4})

    def test_complete_sequence_detection(self):
        trie = Trie.from_sequences([["tool", "call"], ["tool", "result"]])
        self.assertTrue(trie.is_complete(["tool", "call"]))
        self.assertFalse(trie.is_complete(["tool"]))
        self.assertFalse(trie.is_complete(["tool", "missing"]))

    def test_prefix_detection(self):
        trie = Trie.from_sequences([["agent", "action"]])
        self.assertTrue(trie.is_prefix(["agent"]))
        self.assertFalse(trie.is_prefix(["model"]))

    def test_rejects_empty_sequence(self):
        trie = Trie()
        with self.assertRaises(ValueError):
            trie.insert([])


if __name__ == "__main__":
    unittest.main()
