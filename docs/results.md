# Results Summary

The original course experiments evaluated decoding strategies on 50 Hindi-to-English translation examples from IN22-Gen using Llama-2 style causal language models.

The tables below summarize the reported benchmark results.

## Standard Decoding

| Method | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 0.3097 | 0.3538 | 0.1297 | 0.2704 |
| Temperature, tau=0.5 | 0.2856 | 0.2929 | 0.1113 | 0.2387 |
| Temperature, tau=0.9 | 0.1996 | 0.1791 | 0.0549 | 0.1477 |
| Top-k, k=5 | 0.2366 | 0.2267 | 0.0607 | 0.1738 |
| Top-k, k=10 | 0.2200 | 0.2204 | 0.0538 | 0.1683 |
| Nucleus, p=0.5 | 0.2825 | 0.3075 | 0.0999 | 0.2478 |
| Nucleus, p=0.9 | 0.1918 | 0.1908 | 0.0493 | 0.1540 |

## Constrained Decoding

| Method | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: | ---: |
| Trie allowed-vocabulary decoding | 0.3578 | 0.4369 | 0.2507 | 0.3770 |

The constrained decoder scored highest on overlap metrics because the allowed-word lists provided strong lexical guidance for the target translations.

## Medusa-Style Decoding

| Method | BLEU | ROUGE-1 | ROUGE-2 | RTF |
| --- | ---: | ---: | ---: | ---: |
| Single-head Medusa | 0.2921 | 0.3963 | 0.1483 | 0.1070 |
| Multi-head, W=2, S=2 | 0.2842 | 0.3820 | 0.1385 | 0.0730 |
| Multi-head, W=2, S=5 | 0.2801 | 0.3748 | 0.1340 | 0.0590 |
| Multi-head, W=5, S=2 | 0.2980 | 0.4025 | 0.1522 | 0.0910 |
| Multi-head, W=5, S=5 | 0.2925 | 0.3952 | 0.1478 | 0.0775 |
| Multi-head, W=10, S=2 | 0.3120 | 0.4201 | 0.1605 | 0.1250 |
| Multi-head, W=10, S=5 | 0.3055 | 0.4123 | 0.1559 | 0.1055 |

RTF is real-time factor. Lower values indicate faster generation.

## Main Interpretation

The project is best understood as a study of decoding as an inference-time policy:

- Greedy decoding worked well for translation because the task rewards lexical precision.
- Higher temperature and larger sampling sets increased diversity but reduced overlap metrics.
- Word constraints gave the strongest automatic metrics by injecting task-relevant target vocabulary.
- Medusa-style decoding introduced a speed-quality tradeoff that matters for interactive systems.

These findings should not be generalized into a single best decoding rule. The right decoding policy depends on the objective: translation, creative writing, tool calling, code generation, and agent action selection reward different behavior.
