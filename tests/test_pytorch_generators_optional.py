"""Tests for PyTorch-backed reference generators."""

import importlib.util
import unittest


torch_available = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(torch_available, "torch is not installed")
class PyTorchGeneratorTests(unittest.TestCase):
    def test_text_generator_greedy_with_fake_model(self):
        import torch
        from experiments.generate import TextGenerator

        class Output:
            def __init__(self, logits):
                self.logits = logits

        class FakeModel:
            def __init__(self):
                self.tokens = [2, 3, 0]
                self.calls = 0

            def __call__(self, input_ids):
                vocab = 5
                logits = torch.zeros((1, input_ids.shape[1], vocab))
                logits[:, -1, self.tokens[self.calls]] = 10.0
                self.calls += 1
                return Output(logits)

        generator = TextGenerator(FakeModel(), "greedy", eos_id=0, max_output_len=5)
        generated = generator(torch.tensor([[1]], dtype=torch.long))
        self.assertEqual(generated.tolist(), [2, 3])

    def test_medusa_multi_head_commits_only_new_extension(self):
        import torch
        from experiments.generate_medusa import MedusaTextGenerator

        class Output:
            def __init__(self):
                self.logits = torch.tensor([[[0.0, 10.0, 0.0, 0.0]]])
                self.medusa_logits = torch.tensor([
                    [[0.0, 0.0, 10.0, 0.0]],
                    [[0.0, 0.0, 0.0, 10.0]],
                ])

        class FakeModel:
            def __call__(self, input_ids):
                return Output()

        generator = MedusaTextGenerator(
            FakeModel(),
            "multi-head",
            eos_id=0,
            use_no_medusa_heads=2,
            beam_width=1,
            max_output_len=3,
        )
        generated = generator(torch.tensor([[9]], dtype=torch.long))
        self.assertEqual(generated.tolist(), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
