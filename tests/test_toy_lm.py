import unittest

from llm_decoding_lab.toy_lm import EOS_TOKEN, decode


class ToyLanguageModelTests(unittest.TestCase):
    def test_greedy_decode_reaches_eos(self):
        tokens = decode(strategy="greedy", steps=12)
        self.assertEqual(tokens[-1], EOS_TOKEN)
        self.assertEqual(tokens[:4], ["the", "agent", "learns", "a"])

    def test_decode_rejects_non_positive_steps(self):
        with self.assertRaises(ValueError):
            decode(strategy="greedy", steps=0)

    def test_unknown_strategy_is_rejected(self):
        with self.assertRaises(ValueError):
            decode(strategy="missing", steps=3)


if __name__ == "__main__":
    unittest.main()
