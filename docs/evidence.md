# v0.2 evidence ledger

This page separates checked evidence from planned work. The JSON files are the canonical records;
this page is a human-readable index.

## Pinned input

- Dataset: [Common Pile — Python Enhancement Proposals](https://huggingface.co/datasets/common-pile/python_enhancement_proposals)
- Revision: `f932757e3eba16475c893e1418918c77f14a790d`
- Artifact: `raw/documents/00000_peps.jsonl.gz`
- Compressed SHA-256: `659be543c5e79ba776467fb694fbc9efaa350ed022b2e26139b966bae1735b0c`
- Documents: 656 (536 train, 61 validation, 59 test)
- Canonical UTF-8 bytes: 12,057,110
- Tokens including EOD: 12,057,766
- Dataset fingerprint: `8c0076aec77c84fb8f59b5c5772136131805f7837aae6d4d2af928bfc0be6329`
- Tokenizer fingerprint: `fbe3aa2431f27491ba3bf813c9b9054782af25af4e318c671a631d3166ed80a3`

The source marks included documents as public domain and excludes five PEPs with different
licensing. Its dataset card also warns that automated licensing metadata can contain errors. Those
limitations are preserved in the checked-in [dataset manifest](../artifacts/peps-recovery-v0.2/dataset-manifest.json).

## Command executed

```bash
fttl-recovery-matrix \
  --config configs/peps-cpu.json \
  --dataset-manifest .cache/fttl/peps-v1/manifest.json \
  --restarts 2 \
  --output artifacts/peps-recovery-v0.2
```

The downloader first verified the pinned gzip SHA-256. The repository contains the resulting public
manifest and reports, not the downloaded corpus or binary checkpoints.

## Result

The checked matrix is intentionally a six-step recovery smoke test: 12 sample windows from 12
unique documents in the 536-document training split, for 384 input tokens. The manifest covers 656
documents, but this artifact does not claim that every document was consumed or that the model was
trained to useful quality.

| Failure point | Restarts | Exact equality | Durable committed steps/tokens lost | Replayed tokens |
| --- | ---: | --- | --- | ---: |
| Before forward | 2 | Yes | `0 / 0` | 128 |
| After backward | 2 | Yes | `0 / 0` | 128 |
| After optimizer update | 2 | Yes | `0 / 0` | 128 |
| During checkpoint write | 2 | Yes | `0 / 0` | 128 |

Every scenario produced final state digest:

```text
62c3ec4ed4687a3b82ad093d3044a0a9a49dc620843571f4dc65070625b66cfc
```

The exact checks cover batch/window sequence, loss sequence, model tensors, optimizer tensors, RNG
state, cursor, completed steps, token count, final logits, returned model state, final state digest,
and run fingerprint. The matrix also records successful rejection/fallback checks for:

- one changed prepared-dataset byte;
- changed tokenizer identity;
- changed model configuration;
- truncated newest state followed by exact fallback to the previous committed generation;
- corrupt newest manifest followed by exact fallback, resume, and a new durable commit.

Canonical files:

- [Matrix summary](../artifacts/peps-recovery-v0.2/matrix.json)
- [Before-forward report](../artifacts/peps-recovery-v0.2/scenarios/before-forward/recovery-report.json)
- [After-backward report](../artifacts/peps-recovery-v0.2/scenarios/after-backward/recovery-report.json)
- [After-optimizer report](../artifacts/peps-recovery-v0.2/scenarios/after-optimizer/recovery-report.json)
- [Checkpoint-write report](../artifacts/peps-recovery-v0.2/scenarios/during-checkpoint-write/recovery-report.json)

## Recorded environment

- Device: CPU
- Architecture: arm64
- Python: 3.12.14
- PyTorch: 2.13.0
- PyTorch CPU threads: 5

Local wall-clock recovery values are recorded for reproducibility but are not a benchmark or service
level objective. Timing varies with hardware and background load.

## Automated verification

The network-free test suite covers manifest validation, tokenizer round trips and drift, split
isolation, tail/EOD windows, cursor contracts, every training failure boundary, consecutive resumes,
checkpoint crash stages, retention and corrupt fallback, public-safe paths, USGS ledger invariants,
USGS snapshot validation, and sealed-snapshot recovery.

CI does not download external data. It uses an independently written CC0 PEP fixture and a synthetic
USGS-compatible fixture. The real corpus is acquired only through the explicit preparation command.

## Limitations

- Failures are controlled Python exceptions, not OS kills or physical power loss.
- The run is CPU-only and single-process.
- The custom transformer is intentionally small; no model-quality conclusion is supported.
- This is not pretrained-model fine-tuning.
- Recovery timing is local evidence, not an RTO guarantee.
- SHA-256 detects corruption but is not authentication.
- USGS live network capture is implemented, but the checked evidence uses the offline fixture; no
  continuous live-feed availability claim is made.
