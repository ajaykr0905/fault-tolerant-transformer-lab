# Paired public-byte tuning

`fttl-compare-public-tuning` runs full-parameter and LoRA training from identical, randomly
initialized tiny Transformer weights. It consumes a verified prepared dataset, trains only on
the train split, and evaluates an ordered validation prefix before and after each arm. No test
split is used for selection. This is not pretrained-model fine-tuning.

## Network-free working prototype

Run these from the repository root on Linux or macOS, with fresh output paths. Environment setup
may download uncached dependencies; dataset preparation and both experiment commands are
network-free after setup:

```bash
uv sync --frozen --extra test
uv run --frozen fttl-demo --output artifacts/tuning-demo-001
uv run --frozen fttl-compare-public-tuning \
  --config artifacts/tuning-demo-001/config.json \
  --dataset-manifest artifacts/tuning-demo-001/dataset/manifest.json \
  --rank 2 --max-eval-tokens 128 \
  --output artifacts/public-tuning-demo-001/result.json
```

The demo uses six independently written CC0 documents, not downloaded PEP text. For the pinned
public corpus, follow the README's preparation command and substitute `configs/peps-cpu.json` and
`.cache/fttl/peps-v1/manifest.json`. Comparison never downloads data.

## Read the result

The JSON printed to stdout is the published report. It includes:

- `matched_initial_predictions`, `paired_batches`, `paired_cpu_rng_sequence`: enforced paired-arm
  controls, including dropout replay at every training step.
- `comparison_contract` and fingerprints: config, verified source/license, tokenizer, adapter
  rank/alpha/targets, random base, evaluation budget and implementation identity.
- `full` and `lora`: trainable parameters, step/token/sample traces, before/after validation NLL
  and state digests. LoRA must keep its base frozen and bitwise unchanged.

Validation includes end-of-document targets and resets context at document-local windows. A
bounded prefix is not a random sample. Overlapping or repeated training windows count compute
tokens, not unique data. Elapsed time includes final validation and is a local observation, not a
benchmark. A single seed and tiny run cannot establish statistical significance, model quality,
comparative memory use, GPU performance or production readiness.

The comparison restores the caller's Python, NumPy and Torch CPU RNGs and deterministic settings
on success or failure. It makes no CUDA RNG preservation claim. Output must be fresh; a competing
writer cannot be overwritten. JSON is flushed before exclusive local hard-link publication;
this is not a power-loss durability proof.

## Portable adapters and standalone inference

For a separately trained `TinyTransformer` with `inject_lora` already applied:

```python
from pathlib import Path
from fttl.adapter import save_adapter, load_adapter_file
from fttl.adapter_merge import merge_lora_for_inference

save_adapter(trained_lora_model, Path("artifacts/my-adapter-001.pt"))
# matching_lora_model must have the exact frozen base, config, rank and scale.
load_adapter_file(matching_lora_model, Path("artifacts/my-adapter-001.pt"))
inference_model = merge_lora_for_inference(matching_lora_model)
```

These APIs do not export the public comparison's internal models or restore optimizer state.
Adapter files contain only isolated CPU adapter tensors and exact base-binding metadata, not base
weights. Invalid loads are validated before any adapter copy; model mode and trainability stay
unchanged. Only load trusted local files: `weights_only=True` and digests are not authentication.

Inference conversion returns an independent, frozen, eval-mode plain Transformer. It folds
`scale * (B @ A)` into each adapted linear weight and leaves the source untouched. Logits match
within floating-point tolerance, not necessarily bitwise. Conversion rejects adapted tied/shared
base parameters, including an adapted tied language-model head, because folding them would
change another use of the same weight. No latency improvement is claimed without a benchmark.
