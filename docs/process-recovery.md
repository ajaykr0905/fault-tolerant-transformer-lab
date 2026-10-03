# Operate the spawned-worker SIGKILL verifier

Use this verifier to check whether a training worker killed at a known uncommitted
boundary restores the last durable checkpoint and finishes exactly like an independent
control. The network-free tests use the independently written CC0 PEP fixture:

```bash
uv sync --frozen --extra test
uv run --frozen pytest tests/test_process_recovery.py
```

The operating system must support POSIX `SIGKILL`. Use `checkpoint_every=1` and at least
three training steps. `configs/peps-cpu.json` satisfies this contract and enables dropout
so the RNG comparison is observable. To run on the pinned public corpus, prepare it using
the explicit [PEP preparation command](../README.md#reproduce-the-pinned-pep-recovery-proof),
then choose a fresh output directory:

```bash
uv run --frozen fttl-verify-process-recovery \
  --config configs/peps-cpu.json \
  --dataset-manifest .cache/fttl/peps-v1/manifest.json \
  --failure-point after-optimizer \
  --timeout-seconds 60 \
  --output artifacts/local-process-after-optimizer
```

Other accepted boundaries are `before-forward`, `after-backward`, and
`during-checkpoint-write`. Use a separate output directory for each invocation. Omitting
`--dataset-manifest` runs a clearly labeled synthetic smoke workload, not public-corpus
evidence. The verifier never downloads data. Existing non-empty output directories are
rejected rather than overwritten; failed runs remain available for inspection.

## Four independent processes

All workers use Python's `spawn` start method. Only primitive configuration and paths
cross the process boundary; model, optimizer, and RNG state do not inherit from the
parent or another worker.

1. The control trains uninterrupted through the configured number of steps.
2. The interrupted worker commits step 1, then acknowledges the selected step-2 boundary
   over a pipe and blocks there. The parent sends `SIGKILL` and requires exit code `-9`.
3. The replay worker loads the durable generation, consumes exactly the failed batch and
   sample IDs, commits step 2, and exits normally.
4. The completion worker loads that new checkpoint and finishes training.

The checkpoint-write boundary pauses after state serialization but before flush/fsync or
publication. The report requires an abandoned staging state file at that boundary,
requires selection of durable step 1, and verifies cleanup on the next successful save.
It does not treat the abandoned file as a committed generation.

`--timeout-seconds` must be finite and greater than zero. It bounds each worker from the
parent's start call through startup, IPC, and exit. On timeout or verification error, the
parent kills any remaining child and gives cleanup a separate bounded five-second join.
No worker is intentionally left running. A dataset/configuration/tokenizer/run-contract
mismatch during resume fails closed; it is not repaired or treated as corrupt fallback.

## Read the result

Successful commands print one JSON report and write the same `process-recovery-report.json`
under the output directory. `ProcessRecoveryReportV1` records:

- signal exit code and the four worker PIDs;
- attempted and durable steps, selected generation, and tokens recovered;
- failed/replayed batch and sample IDs, replayed tokens, and discarded compute;
- abandoned staging file paths relative to the output root and cleanup status;
- model and optimizer tensor equality, Python/NumPy/PyTorch RNG equality, cursor, histories,
  tokens, logits, checkpoint state, final digest, and run fingerprint equality;
- code/data/tokenizer/configuration fingerprints, environment, and claim limitations.

Any failed equality is recorded with `exact_equality=false`, and the command exits with an
error. Inspect the preserved control/recovered results and report before choosing another
fresh directory. Do not overwrite the evidence or weaken the comparison to obtain a pass.

Timing definitions are included alongside the measured values. The replay-to-commit value
starts at the parent's process start call and ends at receipt of the durable step-2 commit
message, so it includes imports, startup, state restoration, replay, save, and IPC delivery.
It is not a pure checkpoint-load or model-throughput measurement.

## Evidence limits

This checks a real OS-process kill at a deterministic paused boundary, not arbitrary
asynchronous kill timing. Workers run sequentially on CPU; this is not distributed or
simultaneous multi-worker training. The filesystem and OS remain alive, so the result does
not prove power-loss durability. Local timings are not benchmarks or recovery SLOs.
The small model does not establish model quality, GPU scale, or production readiness.
Checkpoint digests detect corruption but do not authenticate attacker-controlled state.

Binary checkpoints and prepared corpora remain local. Publish only compact public-safe
manifests/reports whose code revision and exact reproduction commands are recorded, as in
[the evidence ledger](evidence.md).
