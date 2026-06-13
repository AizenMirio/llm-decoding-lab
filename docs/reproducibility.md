# Reproducibility Notes

This repo has two execution paths:

1. A lightweight local path that runs without model downloads.
2. The original full experiment path that requires gated model and dataset access.

## Lightweight Path

Run the toy decoding demo:

```bash
python examples/toy_decoder_demo.py --strategy all --steps 12 --temperature 0.8 --top-k 4 --top-p 0.85
```

Run the stdlib test suite:

```bash
$env:PYTHONPATH="src"
python -m unittest discover -s tests
```

On Linux/macOS:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## Full Experiment Path

The original experiments use:

- `meta-llama/Llama-2-7b-hf`
- `meta-llama/Llama-2-7b-chat-hf`
- `ai4bharat/IN22-Gen`
- Medusa model: `FasterDecoding/medusa-v1.0-vicuna-7b-v1.5`

Those resources may require a Hugging Face account, model access approval, and a GPU.

Install the required packages:

```bash
pip install -r requirements.txt
```

## Commands

Standard decoding:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/task0.py --hf-token "<your_hf_token>" --decoding-strategy greedy
CUDA_VISIBLE_DEVICES=0 python experiments/task0.py --hf-token "<your_hf_token>" --decoding-strategy random --tau 0.9
CUDA_VISIBLE_DEVICES=0 python experiments/task0.py --hf-token "<your_hf_token>" --decoding-strategy topk --k 10
CUDA_VISIBLE_DEVICES=0 python experiments/task0.py --hf-token "<your_hf_token>" --decoding-strategy nucleus --p 0.9
```

Allowed-vocabulary constrained decoding:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/task1.py --hf-token "<your_hf_token>" --word-list path/to/word_lists.txt
```

`word_lists.sample.txt` shows the tab-separated format used by the benchmark. The full benchmark expects one line of allowed words per evaluation example.

Medusa-style decoding:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/task2.py --hf-token "<your_hf_token>" --decoding-strategy single-head
CUDA_VISIBLE_DEVICES=0 python experiments/task2.py --hf-token "<your_hf_token>" --decoding-strategy multi-head --beam-width 5 --use-no-medusa-heads 2
```

## Benchmark Notes

- The full benchmark requires gated model and dataset access.
- Only a small sample constraint file is included; the full benchmark-side lists are not committed.
- The Llama/IN22 experiment runners assume batch size 1 in practice.
- Benchmark results are summarized in `docs/results.md`.
- Automatic metrics such as BLEU and ROUGE are useful but incomplete measures of generation quality.
