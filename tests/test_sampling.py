import random
import unittest

from llm_decoding_lab.sampling import (
    greedy_token,
    nucleus_candidates,
    select_token,
    softmax,
    top_k_candidates,
)


class SamplingTests(unittest.TestCase):
    def test_softmax_returns_normalized_distribution(self):
        distribution = softmax([("a", 2.0), ("b", 1.0), ("c", 0.0)])
        self.assertAlmostEqual(sum(probability for _, probability in distribution), 1.0)
        self.assertGreater(distribution[0][1], distribution[1][1])

    def test_greedy_selects_largest_logit(self):
        self.assertEqual(greedy_token({"a": 0.0, "b": 3.0, "c": 1.0}), "b")

    def test_top_k_keeps_only_highest_k_tokens(self):
        candidates = top_k_candidates({"a": 0.0, "b": 3.0, "c": 1.0}, k=2)
        self.assertEqual([token for token, _ in candidates], ["b", "c"])

    def test_top_k_validates_k(self):
        with self.assertRaises(ValueError):
            top_k_candidates({"a": 0.0}, k=0)

    def test_nucleus_candidates_cover_requested_probability_mass(self):
        candidates = nucleus_candidates({"a": 3.0, "b": 2.0, "c": 1.0}, p=0.8)
        self.assertGreaterEqual(sum(probability for _, probability in candidates), 0.8)
        self.assertEqual([token for token, _ in candidates], ["a", "b"])

    def test_nucleus_candidates_p_one_keeps_full_vocabulary(self):
        candidates = nucleus_candidates({"a": 3.0, "b": 2.0, "c": 1.0}, p=1.0)
        self.assertEqual([token for token, _ in candidates], ["a", "b", "c"])

    def test_nucleus_candidates_validates_p(self):
        with self.assertRaises(ValueError):
            nucleus_candidates({"a": 1.0}, p=0.0)

    def test_select_token_topk_with_k_one_is_deterministic(self):
        token = select_token(
            {"a": 0.0, "b": 3.0, "c": 1.0},
            strategy="topk",
            rng=random.Random(123),
            top_k=1,
        )
        self.assertEqual(token, "b")


if __name__ == "__main__":
    unittest.main()
