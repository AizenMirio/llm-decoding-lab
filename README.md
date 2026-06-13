# LLM Decoding Lab

Decoding, constrained generation, and Medusa-style fast inference experiments
for language models.

It includes foundational implementations and experiments around decoding,
constrained generation, and fast autoregressive inference. Originally developed
as IIT Bombay coursework and later cleaned for reproducibility.

## What This Project Demonstrates

- **Core decoding algorithms:** greedy decoding, temperature sampling, top-k sampling, and nucleus sampling.
- **Constrained generation:** trie-based allowed-vocabulary decoding that masks invalid next tokens during generation.
- **Fast decoding concepts:** Medusa-style multi-head decoding experiments for reducing autoregressive latency.
- **Evaluation mindset:** BLEU, ROUGE, and real-time-factor style comparisons on Hindi-to-English translation.
- **Testing and structure:** reproducible commands, a small local demo, and unit tests for the core decoding components.

## Why It Matters

Modern LLM behavior is not only determined by model weights. Decoding controls how a model explores the next-token distribution. Small changes to temperature, top-k, top-p, constraints, or speculative decoding can substantially change:

- factuality and repetition,
- diversity and creativity,
- translation quality,
- latency,
- agent action selection,
- structured-output reliability.

That makes decoding useful for studying inference-time behavior in language models.

## Repository Map

| Path | Purpose |
| --- | --- |
| `src/llm_decoding_lab/` | Reusable, dependency-free decoding primitives used by the local demo and tests. |
| `tests/` | Unit tests for sampling, nucleus/top-k filtering, trie constraints, and toy decoding. |
| `docs/results.md` | Clean summary of reported BLEU, ROUGE, and RTF results. |
| `docs/technical_overview.md` | Detailed explanation of the decoding methods, architecture, and limitations. |
| `docs/reproducibility.md` | Lightweight and full-benchmark reproducibility instructions. |
| `experiments/` | PyTorch reference implementations and benchmark runners for the translation experiments. |
| `examples/toy_decoder_demo.py` | Tiny dependency-free demo for comparing decoding strategies locally. |
| `word_lists.sample.txt` | Tiny example showing the tab-separated constrained-decoding word-list format. |

## Quick Local Demo

The fastest way to inspect the decoding ideas is the toy demo. It uses a hand-written toy language model, so it does not need PyTorch, Transformers, Hugging Face tokens, Llama weights, or the IN22 dataset.

```bash
python examples/toy_decoder_demo.py --strategy all --steps 12 --temperature 0.8 --top-k 4 --top-p 0.85
```

Example output shape:

```text
greedy  : the agent learns a policy with rewards <eos>
random  : the model samples diverse actions ...
topk    : the agent optimizes rewards ...
nucleus : the policy explores ...
```

The benchmark experiments use Llama-style causal language models and the IN22-Gen translation dataset.

## Local Verification

Run the unit tests:

```bash
$env:PYTHONPATH="src"
python -m unittest discover -s tests
```

On Linux/macOS:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## Full Experiment Setup

The original benchmark used gated model and dataset access:

- Model: `meta-llama/Llama-2-7b-hf` or `meta-llama/Llama-2-7b-chat-hf`
- Dataset: `ai4bharat/IN22-Gen`
- Optional Medusa model: `FasterDecoding/medusa-v1.0-vicuna-7b-v1.5`

Install the core packages:

```bash
pip install -r requirements.txt
```

You will need a Hugging Face token with access to the model and dataset.

## Running the Main Experiments

Greedy decoding:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/task0.py --hf-token "<your_hf_token>" --decoding-strategy greedy
```

Temperature sampling:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/task0.py --hf-token "<your_hf_token>" --decoding-strategy random --tau 0.9
```

Top-k sampling:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/task0.py --hf-token "<your_hf_token>" --decoding-strategy topk --k 10
```

Nucleus sampling:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/task0.py --hf-token "<your_hf_token>" --decoding-strategy nucleus --p 0.9
```

Allowed-vocabulary constrained decoding:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/task1.py --hf-token "<your_hf_token>" --word-list path/to/word_lists.txt
```

`word_lists.sample.txt` shows the tab-separated format used by the benchmark. The full benchmark expects one line per evaluation example.

Medusa-style decoding:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/task2.py --hf-token "<your_hf_token>" --decoding-strategy single-head
CUDA_VISIBLE_DEVICES=0 python experiments/task2.py --hf-token "<your_hf_token>" --decoding-strategy multi-head --beam-width 5 --use-no-medusa-heads 2
```

## Reported Results

The original benchmark evaluated 50 Hindi-to-English translation examples using BLEU and ROUGE. A fuller interpretation is available in `docs/results.md`. The table below summarizes the reported benchmark results.

| Method | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 0.3097 | 0.3538 | 0.1297 | 0.2704 |
| Temperature, tau=0.5 | 0.2856 | 0.2929 | 0.1113 | 0.2387 |
| Temperature, tau=0.9 | 0.1996 | 0.1791 | 0.0549 | 0.1477 |
| Top-k, k=5 | 0.2366 | 0.2267 | 0.0607 | 0.1738 |
| Top-k, k=10 | 0.2200 | 0.2204 | 0.0538 | 0.1683 |
| Nucleus, p=0.5 | 0.2825 | 0.3075 | 0.0999 | 0.2478 |
| Nucleus, p=0.9 | 0.1918 | 0.1908 | 0.0493 | 0.1540 |
| Allowed-vocabulary constrained | 0.3578 | 0.4369 | 0.2507 | 0.3770 |

Medusa-style experiments tracked quality and real-time factor:

| Method | BLEU | ROUGE-1 | ROUGE-2 | RTF |
| --- | ---: | ---: | ---: | ---: |
| Single-head Medusa | 0.2921 | 0.3963 | 0.1483 | 0.1070 |
| Multi-head, W=2, S=2 | 0.2842 | 0.3820 | 0.1385 | 0.0730 |
| Multi-head, W=5, S=2 | 0.2980 | 0.4025 | 0.1522 | 0.0910 |
| Multi-head, W=10, S=2 | 0.3120 | 0.4201 | 0.1605 | 0.1250 |

The main trend is that tighter decoding and explicit constraints improved lexical overlap metrics, while higher-diversity settings reduced BLEU and ROUGE on this translation benchmark. Medusa-style decoding explored the quality-latency tradeoff.
