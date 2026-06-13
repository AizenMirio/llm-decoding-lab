# Technical Overview

## Overview

Autoregressive language models generate text one token at a time. At each step, the model produces a probability distribution over the vocabulary, and a decoding rule selects the next token. Different decoding rules can produce very different tradeoffs in precision, diversity, controllability, and latency.

This repository studies that effect through standard sampling methods, constrained decoding, and Medusa-style multi-token decoding.

## Implemented Methods

### Greedy Decoding

Greedy decoding selects the most likely token at each step:

```text
next_token = argmax P(token | prompt, generated_so_far)
```

It is deterministic and often strong on precision-oriented tasks such as translation, but it can also be repetitive and locally myopic.

### Temperature Sampling

Temperature rescales logits before sampling:

```text
probabilities = softmax(logits / temperature)
```

Lower temperature sharpens the distribution. Higher temperature flattens it and increases diversity.

### Top-k Sampling

Top-k sampling keeps only the `k` highest-scoring tokens, renormalizes their probabilities, and samples from that reduced set. This limits the influence of very unlikely tokens while preserving some randomness.

### Nucleus Sampling

Nucleus sampling, or top-p sampling, keeps the smallest token set whose cumulative probability exceeds a threshold `p`. Unlike top-k, the candidate set size is adaptive.

### Allowed-Vocabulary Constrained Decoding

The constrained decoder builds a trie from tokenized allowed words. At each step, logits for invalid next tokens are masked out, so generation can only continue along valid trie prefixes.

This implementation is stricter than "include these words somewhere." It is better described as allowed-vocabulary constrained generation.

### Medusa-Style Decoding

Medusa-style decoding proposes multiple future tokens per forward pass by using auxiliary heads in addition to the main language-model head. The implementation in this repository is a compact multi-head decoding reference that captures the candidate-extension idea and the speed-quality tradeoff.

## Repository Structure

- `src/llm_decoding_lab/sampling.py`: dependency-free helpers for greedy, random, top-k, and nucleus selection.
- `src/llm_decoding_lab/trie.py`: generic trie implementation for constrained decoding.
- `src/llm_decoding_lab/medusa.py`: dependency-free helpers for Medusa-style candidate extensions.
- `src/llm_decoding_lab/toy_lm.py`: lightweight toy language model for local demonstrations.
- `experiments/generate.py`, `experiments/generate_constrained.py`, `experiments/generate_medusa.py`: PyTorch reference implementations used by the benchmark runners.
- `experiments/task0.py`, `experiments/task1.py`, `experiments/task2.py`: experiment runners for the original translation benchmark.

## Benchmark Interpretation

The original benchmark used Hindi-to-English translation with BLEU, ROUGE, and real-time factor.

The key takeaway is not that one decoding rule is universally best. The useful conclusion is that decoding acts as an inference-time policy, and the right choice depends on the objective:

- Translation rewards precision and lexical fidelity, so lower-diversity decoding performed better.
- Constrained decoding improved overlap metrics because the allowed vocabulary supplied strong lexical guidance.
- Medusa-style decoding exposed a speed-quality tradeoff that matters for interactive systems.

## Limitations

- The original translation benchmark depends on gated model and dataset access.
- The PyTorch experiment runners assume batch size 1 in practice.
- The Medusa implementation is a compact reference implementation.
- BLEU and ROUGE are useful but incomplete measures of generation quality.
