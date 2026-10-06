# Generate from the recovered prototype

After running `fttl-demo --output artifacts/my-demo-001`, generate from its actual
recovered checkpoint. No download, API key or GPU is needed after installation.

```bash
uv run --frozen fttl-generate \
  --config artifacts/my-demo-001/config.json \
  --checkpoint artifacts/my-demo-001/run/recovered/checkpoints \
  --dataset-manifest artifacts/my-demo-001/dataset/manifest.json \
  --prompt 'Hello' --max-new-tokens 8 \
  --output artifacts/my-generation-001.json
```

The checkpoint loader verifies dataset, tokenizer and configuration identity,
integrity, FP32 state and generation before constructing a frozen CPU model.
Training RNG and optimizer are not installed in the inference caller. Use only
trusted local checkpoint stores: integrity hashes do not authenticate a source.

Greedy decoding breaks ties by lower token ID. For seeded sampling, add
`--method sample --temperature 0.8 --top-k 8 --seed 23`. Optional
`--stop-token-id 256` stops after emitting EOD. Prompts are UTF-8 bytes; length caps
are 4096 prompt bytes and 256 new tokens. Context is cropped to `block_size` with
position IDs reset, not cached. Byte IDs remain authoritative: arbitrary output
may not decode as UTF-8, and EOD is not a byte. Reports never overwrite earlier evidence.

This demonstrates checkpoint-to-inference continuity. The six-step model is not
a useful language assistant; this command does not establish language quality,
GPU performance, concurrent serving safety or production readiness.

## Does recovery preserve generated tokens?

Compare the actual demo control and recovered checkpoints using both greedy and
private-seeded top-k sampling:

```bash
uv run --frozen fttl-verify-inference-parity \
  --config artifacts/my-demo-001/config.json \
  --control-checkpoint artifacts/my-demo-001/run/control/checkpoints \
  --recovered-checkpoint artifacts/my-demo-001/run/recovered/checkpoints \
  --dataset-manifest artifacts/my-demo-001/dataset/manifest.json \
  --prompt Hello --prompt Recovery --max-new-tokens 8 \
  --output artifacts/my-inference-parity-001.json
```

Exit status is zero only when model state, training step, training contract and
all measured token sequences match. A valid measured mismatch is preserved in
the report and exits with status one. This is bounded prompt coverage, not a
claim that every input was tested.

## Measure this machine, not an SLA

```bash
uv run --frozen fttl-benchmark-inference \
  --config artifacts/my-demo-001/config.json \
  --checkpoint artifacts/my-demo-001/run/recovered/checkpoints \
  --dataset-manifest artifacts/my-demo-001/dataset/manifest.json \
  --iterations 5 --warmups 1 --max-new-tokens 8 \
  --output artifacts/my-inference-benchmark-001.json
```

The bounded CPU report retains every elapsed sample and actual emitted token
count. It measures the complete generation call, including integrity/RNG overhead,
but excludes checkpoint loading and warmups. Median and nearest-rank p95 are
descriptive statistics for these few samples, not tail-latency guarantees.
