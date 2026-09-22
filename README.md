# Fault-Tolerant Transformer Training and Serving Lab

An evidence-first AI infrastructure project that begins with deterministic CPU training and atomic checkpoint recovery before making any GPU, distributed-training, or serving claims.

## Verified in the first release

- Small decoder-only PyTorch transformer with causal attention.
- Stable synthetic smoke dataset and step-keyed batches.
- Autograd gradient check for the causal attention implementation.
- Atomic checkpoints containing the model, optimizer, configuration fingerprint, step, token count, loss history, and RNG state.
- Exact equality between uninterrupted training and checkpoint-restarted training in the CPU test configuration.
- Machine-readable experiment manifest and result artifact.

## Explicitly not claimed yet

- Multi-GPU or FSDP execution.
- vLLM or SGLang serving performance.
- LoRA versus full-fine-tuning results.
- OpenTelemetry, Prometheus, and Grafana measurements.
- Production-scale throughput, latency, memory, cost, or recovery time.

Those capabilities remain gated until real, reproducible evidence is checked in.

## Run the CPU verification path

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
pytest
fttl-train-smoke --config configs/smoke.json --output artifacts/local-smoke
```

The smoke run uses a deterministic synthetic token stream. It is an engineering verification workload, not a language-quality benchmark.

The public `artifacts/cpu-smoke/` directory records the verified manifest and result. The binary checkpoint remains excluded from source control.

## Repository map

- `src/fttl/model.py` — causal attention and the small transformer.
- `src/fttl/checkpoint.py` — atomic, configuration-bound checkpoints.
- `src/fttl/train.py` — deterministic training and restart path.
- `configs/` — versioned experiment manifests.
- `tests/` — gradient, causal, and recovery verification.
- `docs/architecture.md` — system and evidence boundaries.

## Next evidence gates

1. Public-dataset baseline and controlled ablations.
2. LoRA versus full-fine-tuning comparison.
3. Real NVIDIA GPU and multi-GPU execution with hardware metadata.
4. vLLM or SGLang serving with failure injection.
5. OpenTelemetry, Prometheus, and Grafana instrumentation.
6. Throughput, utilization, memory, p50/p95/p99 latency, cost, and recovery report.

## Clean-room statement

All code, data, identifiers, configurations, and results in this repository are independently produced for the public lab. No employer or customer material is used.

## License

MIT
