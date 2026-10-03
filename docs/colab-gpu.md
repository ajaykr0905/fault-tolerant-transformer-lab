# Run the first single-GPU checkpoint experiment

Use the Colab notebook in `notebooks/colab_cuda_recovery.ipynb`. Select **Runtime > Change runtime
type > T4 GPU** when a free GPU is available, then run its cells in order. Do not buy compute,
enable a paid fallback, use multiple accounts to evade quotas, or expose an SSH/public service.
The notebook uses interactive compute, not an always-on deployment.

The notebook checks out an immutable implementation revision and creates an isolated environment
with the committed lockfile. Its first hardware probe records the actual GPU, driver and memory.
The locked Linux PyTorch build uses CUDA 13 dependencies. If the allocated runtime's GPU/driver
cannot run it, stop: changing dependencies requires a separate reviewed lockfile, not disabling
the PyTorch security floor or silently accepting CPU execution.

## What is being implemented

The existing CPU training/recovery commands and published CPU evidence stay unchanged. A separate,
small `fttl.cuda_recovery` experiment uses the same transformer, prepared public PEP dataset,
training cursor, checkpoint store and numerical guards on one NVIDIA device:

1. Require CUDA; configure deterministic eager FP32 before initialization. No AMP, compilation or
   fused optimizer is used in this first check.
2. Train an uninterrupted control with nonzero dropout.
3. Train a second run to a declared step, publish a checkpoint containing selected-device CUDA
   RNG plus Python, NumPy and CPU Torch RNG, then discard its model and optimizer objects.
4. Construct fresh CUDA objects, verify checkpoint/data/config/execution contracts, restore RNG and
   the cursor, and finish training.
5. Compare all declared state and history checks exactly on the same observed CUDA environment.
6. Evaluate a CPU copy of the GPU-trained weights on bounded held-out validation bytes. The report
   explicitly labels this CPU evaluation; it does not claim GPU inference evidence.

The default PEP configuration performs six steps and processes 384 training input tokens. It is a
correctness smoke workload, not training over the entire 656-document corpus or a meaningful model
quality experiment. Byte-level NLL/perplexity is not comparable with a word/subword benchmark.

## Run without a notebook

In a fresh Linux process with the locked CUDA-capable environment and a prepared public dataset:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m fttl.cuda_recovery \
  --config configs/peps-cpu.json \
  --dataset-manifest .cache/fttl/peps-v1/manifest.json \
  --interruption-step 2 \
  --max-eval-tokens 4096 \
  --output artifacts/local-cuda-001
```

Use a fresh output directory. CUDA unavailability, incompatible execution metadata, invalid state,
non-finite results or an equality mismatch must fail without a success report. Keep binary
checkpoints trusted-local and outside Git. Download the compact JSON report before the disposable
Colab runtime disappears. Persisting data across runtime deletion needs a separately verified
storage/export workflow; the notebook does not mount or grant access to all of Google Drive.

## Evidence boundary

The [2026-10-03 measured report](../artifacts/colab-cuda-2026-10-03/cuda-recovery-report.json)
was exported from a free Colab Tesla T4 allocation with driver 580.82.07 and locked
PyTorch 2.13.0+cu130. Implementation revision `4ffaf1da88f9092078d61ed3c801597eefcd0809`
completed all 41 focused tests without skips, including nine hardware-dependent cases.
The six-step public PEP experiment passed all 11 equality checks. Experiment-wide peak
allocated memory was 68,169,728 bytes; peak reserved memory was 69,206,016 bytes.
The export SHA-256 is `13dca14329edf830df3da6cafdeb69c0834ec7cbc71f1d4daa2daf3330324ffe`.
These timings and memory measurements are one small correctness run, not performance claims.

A run only becomes GPU evidence after the real hardware tests and experiment complete, with actual
hardware metadata, a clean pinned revision and all equality checks recorded. CPU-side mocks and
skipped CUDA tests are not that evidence.

This first experiment reconstructs training objects in the same running process. It is **not** a
GPU SIGKILL, runtime-disconnection, physical-power-loss, multi-GPU or production-serving proof.
The existing process-kill verifier remains CPU-only. Exact equality is required only within the
observed GPU/runtime policy, not across CPU/GPU, releases, different cards or machines. Memory and
synchronized timings include checks and checkpoint operations; they are not kernel benchmarks.

## Free compute limits

[Colab's official FAQ](https://research.google.com/colaboratory/faq.html) describes dynamic quotas,
non-guaranteed GPU allocation, idle termination and an at-most-12-hour free notebook lifetime.
That is not a promise of twelve GPU hours. Free Colab is appropriate for an interactive, bounded
experiment, not an unattended service or distributed worker deployment.

[Kaggle](https://www.kaggle.com/docs/efficient-gpu-usage) also documents free GPU quotas, generally
30 weekly hours or a demand-dependent higher allocation. Check the account's current quota and
available accelerator rather than relying on a fixed card or entitlement.

Contributing directly to CUDA is not a requirement. PyTorch already dispatches the model's tensor
operations to CUDA. Upstream PyTorch/Triton/kernel work should follow a reproduced correctness bug
or measured bottleneck after this baseline, not precede it for portfolio activity.
