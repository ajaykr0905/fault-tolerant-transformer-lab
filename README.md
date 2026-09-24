# Fault-Tolerant Transformer Lab

A CPU-verifiable training-recovery system for one concrete question:

> After a crash, incomplete checkpoint, or changed dataset, can training resume from the last
> committed state without skipping data, duplicating data, accepting corruption, or silently
> producing a different result?

This repository treats the model, optimizer, random-number generators, dataset cursor, and input
fingerprints as one recovery contract. It trains a small decoder-only PyTorch transformer, injects
controlled failures, resumes from durable state, and compares the recovered run with an
uninterrupted control.

## Verified now

- A real corpus path using 656 pinned Python Enhancement Proposal documents.
- Deterministic UTF-8 byte tokenization, stable document IDs, and disjoint hash-based splits.
- Transactional training cursor and exact batch/window replay after an uncommitted step.
- Versioned checkpoint generations with atomic publication, SHA-256 integrity manifests, and
  fallback from a corrupt newest generation.
- Exact CPU equality after failures before forward, after backward, after optimizer update, and
  during checkpoint publication.
- Original run, resume, save, and second-resume regression coverage.
- Dataset, tokenizer, configuration, code, run-contract, and final-state fingerprints.
- A bounded USGS earthquake-feed capture ledger that seals immutable snapshots for the same
  offline training/recovery pipeline.

The checked-in [recovery matrix](artifacts/peps-recovery-v0.2/matrix.json) records four failure
scenarios with two consecutive restarts each. All scenarios reached the same final state digest and
passed every declared equality check. See [the evidence ledger](docs/evidence.md) for commands,
environment, provenance, and limitations.

## Five-minute local proof (network-free)

Prerequisites: Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/ajaykr0905/fault-tolerant-transformer-lab.git
cd fault-tolerant-transformer-lab
uv sync --frozen --extra test
uv run pytest
uv run fttl-train-smoke \
  --config configs/smoke.json \
  --output artifacts/local-smoke
```

The tests use an independently written, public-domain fixture and do not download the PEP corpus.
Every output command requires a fresh directory so an earlier run cannot be silently overwritten.

## Reproduce the pinned PEP recovery proof

The preparation step is the only command that needs network access. It downloads one revision-pinned
gzip file into `.cache/`, verifies its compressed SHA-256 before decompression, and writes a
canonical `DatasetManifestV1`. The corpus itself and binary checkpoints stay out of Git.

```bash
uv run fttl-prepare-peps \
  --cache-dir .cache/fttl \
  --output .cache/fttl/peps-v1

uv run fttl-train-smoke \
  --config configs/peps-cpu.json \
  --dataset-manifest .cache/fttl/peps-v1/manifest.json \
  --output artifacts/local-peps

uv run fttl-verify-recovery \
  --config configs/peps-cpu.json \
  --dataset-manifest .cache/fttl/peps-v1/manifest.json \
  --failure-point after-optimizer \
  --restarts 2 \
  --output artifacts/local-peps-recovery
```

Run the complete four-scenario matrix with:

```bash
uv run fttl-recovery-matrix \
  --config configs/peps-cpu.json \
  --dataset-manifest .cache/fttl/peps-v1/manifest.json \
  --restarts 2 \
  --output artifacts/local-peps-matrix
```

## Live-feed capture, deterministic training

Milestone 2 captures versioned USGS event records into a SQLite WAL ledger, deduplicates by
`(source, event_id, updated_at_ms)`, and seals a content-addressed JSONL snapshot. Training consumes
the sealed snapshot, not a mutable network response.

```bash
uv run fttl-capture-usgs \
  --feed all-day \
  --database .cache/fttl/usgs.sqlite3 \
  --output .cache/fttl/usgs-v1 \
  --polls 1
```

The command prints the resulting training-manifest path. This is **real-time feed ingestion with
deterministic snapshot training and recovery**. It is not online learning and it does not predict
earthquakes. CI exercises the same path with a synthetic USGS-compatible fixture and never makes a
network request.

## Recovery transaction

```text
prepare batch -> forward/backward -> optimizer update -> commit step + cursor
      ^                                                        |
      |                    crash before commit                 |
      +--------------------------------------------------------+
```

An uncommitted attempt must reuse the same batch after restart. A checkpoint becomes eligible for
recovery only after its state file and integrity manifest are durable and its generation is
atomically published. The loader validates size, digest, schema, expected keys, model configuration,
dataset identity, tokenizer identity, and run contract before deserialization.

Read [the architecture](docs/architecture.md) and [checkpoint security boundary](docs/checkpoint-security.md)
for the exact state machine and threat model.

## Repository map

- `src/fttl/dataset.py` — pinned PEP preparation, manifest validation, and byte tokenizer.
- `src/fttl/data.py` — synthetic and prepared-data batch sources plus `TrainingCursorV1`.
- `src/fttl/train.py` — transaction boundary, controlled interruption, and run fingerprints.
- `src/fttl/checkpoint.py` — generation-based durable publication and integrity validation.
- `src/fttl/recovery.py` — repeated-restart proof and `RecoveryReportV1`.
- `src/fttl/matrix.py` — four failure scenarios plus incompatibility/corruption checks.
- `src/fttl/usgs.py` — bounded capture, SQLite ledger, sealed snapshot, and dataset adapter.
- `artifacts/peps-recovery-v0.2/` — public-safe manifests and reports; no binary checkpoints.
- `tests/` — unit, integration, recovery, corruption, provenance, and security-boundary tests.

## Evidence boundary

This release proves deterministic recovery for a small, pinned CPU run. It does **not** prove model
quality, pretrained-model fine-tuning, GPU behavior, multi-process recovery, distributed scale,
production readiness, or a service-level recovery objective. Failure injection uses deterministic
Python exceptions; OS-kill and power-loss testing are later gates.

The existing LoRA experiment compares adapter mechanics under a controlled synthetic workload. It
is not evidence that a pretrained model improved.

Full resume checkpoints are trusted local artifacts. `torch.load(..., weights_only=True)` and
SHA-256 validation reduce risk, but a digest is not authentication if an attacker can replace both
the payload and manifest.

## Next evidence gates

1. OS-process kill and filesystem fault injection.
2. A pinned pretrained model with a real LoRA evaluation and retention checks.
3. Actual GPU evidence with hardware and memory measurements.
4. Multi-process PyTorch Distributed Checkpoint or TorchFT experiments.
5. Authenticated checkpoint manifests and remote object-store publication.

## Clean-room and data statement

All implementation code and fixtures are independently written for this public lab. No employer,
customer, private, personal, or social-media data is used. External datasets are revision-pinned,
attributed, validated, and accompanied by their stated limitations.

## License

Code is MIT licensed. Dataset licenses and attribution remain governed by their respective sources.
