# Evaluate a CPU checkpoint

After preparing the pinned corpus and training with `configs/peps-cpu.json`, run:

```bash
uv sync --frozen --extra test
uv run --frozen fttl-evaluate-checkpoint \
  --config configs/peps-cpu.json \
  --checkpoint artifacts/local-peps/checkpoints \
  --dataset-manifest .cache/fttl/peps-v1/manifest.json \
  --split validation \
  --max-tokens 4096 \
  --output artifacts/local-peps-validation.json
```

The output must be a new file. Evaluation validates checkpoint integrity and binds
the model to the expected configuration, dataset and tokenizer. It does not alter
the checkpoint, restore training RNG state or update weights. Older code revisions
can be evaluated when their model state and input contracts remain compatible;
the report records their training contract separately from the evaluation code.

Use validation for model selection. Reserve test for the final comparison after
choosing settings. Both splits must come from the same verified dataset manifest.
The training split is rejected by the evaluator.

Each document contributes UTF-8 byte targets followed by EOD. The first byte is
context, so each document contributes exactly its UTF-8 byte length in targets.
Windows never cross document boundaries or wrap the stream. A short final window
is included, and context resets between windows. Total NLL is divided by the total
target count, including partial windows. Perplexity is `exp(mean NLL)`; an overflow
is reported as `null` while preserving finite NLL.

`--max-tokens` selects a deterministic prefix in document-ID order. Consult
`complete_split`, available/evaluated document counts and target counts before
interpreting a result. A small prefix is a smoke check and can be unrepresentative
of the corpus. This byte-level metric cannot be compared directly with another
model's word- or subword-token perplexity. CPU smoke results do not establish GPU
performance or pretrained-model quality.
