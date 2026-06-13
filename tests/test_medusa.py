import unittest

from llm_decoding_lab.medusa import build_medusa_candidates, truncate_at_eos


class MedusaCandidateTests(unittest.TestCase):
    def test_candidate_builder_returns_extensions_not_prefix_duplicates(self):
        candidates = build_medusa_candidates(
            head_log_probs=[
                [("A", -0.1), ("B", -1.0)],
                [("C", -0.2), ("D", -0.5)],
            ],
            beam_width=2,
        )
        self.assertEqual(candidates[0].tokens, ("A", "C"))
        self.assertEqual(candidates[1].tokens, ("A", "D"))

    def test_candidate_builder_validates_inputs(self):
        with self.assertRaises(ValueError):
            build_medusa_candidates([], beam_width=2)
        with self.assertRaises(ValueError):
            build_medusa_candidates([[("A", -0.1)]], beam_width=0)

    def test_truncate_at_eos_drops_eos_and_following_tokens(self):
        self.assertEqual(truncate_at_eos(["A", "<eos>", "B"], "<eos>"), ("A",))
        self.assertEqual(truncate_at_eos(["A", "B"], "<eos>"), ("A", "B"))


if __name__ == "__main__":
    unittest.main()
