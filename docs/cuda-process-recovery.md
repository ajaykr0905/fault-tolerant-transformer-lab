# Verify a killed CUDA training worker

This experiment extends the single-device lab with one process-death boundary:
after the optimizer updates step 2, before that step publishes a checkpoint.
It uses actual POSIX `SIGKILL`, not a caught exception or reconstruction inside
the same training process. The corrected October 4 CLI report verifies this one
boundary on a T4, and all 70 tests in the four-target GPU regression gate passed
without skips. The October 3 T4 report separately checks only same-process
reconstruction.

## Measured run and verification status

The [corrected raw Colab export](../artifacts/colab-cuda-process-2026-10-04/cuda-process-recovery-report-65d2808.json)
was downloaded after the notebook's report assertions passed. Its SHA-256 is:
`f3cd1f8cddbc4fb26e0446d1d704a02bc3b758e7e00fb03a361d8aa890e96ff3`.
Its timestamp is October 4, 2026, 04:58:08 IST
(`2026-10-03T23:28:08.040536+00:00`). It used public PEP data and implementation
revision `65d280888e0f34aa9dc7107f8197dcef7f48fa34` on a Tesla T4, PyTorch
`2.13.0+cu130`, CUDA build 13.0, eager FP32 and strict deterministic controls.

Four distinct spawned worker PIDs, SIGKILL exit `-9`, durable/selected step 1,
step-2 batch/sample replay, all 22 exact-equality checks, and six completed steps
(384 tokens) are recorded. Sixty-four compute tokens were discarded; no durable
committed steps or tokens were lost. Replay spawn to committed-step receipt was
6.187 seconds, including imports, CUDA initialization, restore, save and IPC;
it is an observation, not a performance benchmark.

At this pinned revision, a clean Colab bootstrap completed on the observed T4.
The following GPU regression targets ran in fresh subprocesses and passed with
no skips:

- `tests/test_cuda_process_recovery.py`: 29 passed.
- `tests/test_cuda_runtime.py`: 18 passed.
- `tests/test_cuda_recovery.py`: 22 passed.
- `tests/test_checkpoint_integrity.py::test_real_cuda_rng_checkpoint_is_weights_only_safe_and_cpu_loadable`: 1 passed.

The regression cell completed in 56 seconds. The public-data CLI experiment
completed in 34 seconds with all report assertions passing, followed by the
report download. These are observed notebook-cell durations, not benchmarks.

The [original raw report](../artifacts/colab-cuda-process-2026-10-04/cuda-process-recovery-report.json)
remains unchanged: revision `8dd38161ea2563eb3fb82ee960c039141b00cae2`,
October 4, 03:51:44 IST, SHA-256
`ee77c790103d13d145067570ab77d48c03a2428a048e9564c12552013b4ef8cd`.
That CLI experiment passed, but its original process test module had 28 passes
and one failure because a CPU AdamW observer test initialized the pytest parent's
CUDA context. The original hardware integration also passed alone in a fresh
pytest process. The corrected gate above includes the test-isolation fix,
preserves the verifier's CPU-only-parent guard and exercises the public CLI in
a separate process. The earlier report is not relabeled as a corrected run.

The CPU artifact test protects both exact downloads and historical report
contracts; it does not authenticate hardware execution. No tensor/checkpoint
binaries were exported, so readers cannot independently rehash the reported final
tensor digest from this JSON. No new results screenshot or executed public Drive
copy is published. Do not treat the unexecuted operating notebook as GPU proof.

## Run

Open the [pinned process-recovery notebook in Colab](https://colab.research.google.com/github/ajaykr0905/fault-tolerant-transformer-lab/blob/3afb0df065d97eaad49c0972972fe44fcbb53ea5/notebooks/colab_cuda_process_recovery.ipynb),
select a free T4 GPU when available, and run its cells in order. It
checks the actual hardware, refuses skipped GPU tests, and exports the report.
The notebook pins implementation revision
`65d280888e0f34aa9dc7107f8197dcef7f48fa34`; it does not train whatever happens to
be on a moving branch. The public notebook is unexecuted and is not GPU evidence.
This revision includes the test-isolation fix and matches the corrected measured
run. Both historical reports retain their original revisions and timestamps.

Use a free interactive Colab GPU or an existing Linux NVIDIA environment with
the committed dependency lock. Prepare the public PEP dataset using the existing
data command. In a fresh process, run:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m fttl.cuda_process_recovery \
  --config configs/peps-cpu.json \
  --dataset-manifest .cache/fttl/peps-v1/manifest.json \
  --output artifacts/local-cuda-process-001
```

Use a new output path. The configuration must enable dropout, save every step,
and contain at least three steps. CUDA must be available to the workers. There
is no CPU fallback. Keep checkpoints and worker snapshots trusted-local and
outside Git. Export the compact JSON report before a disposable runtime expires.

## What the experiment checks

The parent starts four independent workers with Python's `spawn` method. Only
workers initialize CUDA. An uninterrupted control completes the workload. A
second worker pauses at the declared step-2 boundary after synchronizing CUDA;
the parent kills it and checks its signal exit status. A fresh worker restores
the last durable step-1 checkpoint, replays step 2, and publishes it. Another
fresh worker resumes that checkpoint and finishes training.

The verifier checks the failed and replayed batch/sample identities, full loss
history, model and optimizer tensors, Python/NumPy/CPU Torch/selected CUDA RNG,
cursor, token count, and final logits computed on the GPU. CPU copies are used
only to compare and serialize the captured states. The workers must agree on
the code, configuration, dataset, tokenizer, and observed GPU execution contract.
An error, timeout, missing checkpoint, or mismatch must not publish a success
report. The parent uses bounded deadlines and reaps workers on every exit path.

The report defines its timing boundaries. Spawn-to-replayed-commit time includes
Python import and CUDA initialization, checkpoint selection and restoration,
one replayed step, durable save, and IPC delivery. It is not just a kernel time.

## Limits

This checks one paused transaction boundary on one observed GPU/runtime using
eager FP32 and a small transformer. It does not demonstrate abrupt Colab runtime
deletion, host or physical-power failure, arbitrary asynchronous failure points,
remote durable storage, multi-GPU recovery, serving, or production reliability.
Exact equality is required within the recorded execution contract, not across
different GPUs, drivers, PyTorch releases, or CPU/GPU implementations.

CPU tests validate input contracts and process supervision. Skipped CUDA tests
on a Mac are expected and must not be reported as successful GPU validation.
For real hardware, run `tests/test_cuda_process_recovery.py` in a fresh subprocess
with `CUBLAS_WORKSPACE_CONFIG=:4096:8` and require no hardware-dependent skips.
