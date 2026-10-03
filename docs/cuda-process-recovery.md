# Verify a killed CUDA training worker

This experiment extends the single-device lab with one process-death boundary:
after the optimizer updates step 2, before that step publishes a checkpoint.
It uses actual POSIX `SIGKILL`, not a caught exception or reconstruction inside
the same training process. GPU validation is pending until a measured report is
published. The October 3 T4 report proves only same-process reconstruction.

## Run

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
