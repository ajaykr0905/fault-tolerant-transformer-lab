# Colab GPU run: notebook and evidence

[Open the executed notebook](https://colab.research.google.com/drive/1MaDetRnr9XVXDmpL_vwMRrctlIhYaVaa).
Its sharing permissions are unverified; this link may require the owner's access.
For a public, reproducible copy, [open the repository notebook in Colab](https://colab.research.google.com/github/ajaykr0905/fault-tolerant-transformer-lab/blob/main/notebooks/colab_cuda_recovery.ipynb)
and follow the [GPU runbook](colab-gpu.md).

The [original measured JSON report](../artifacts/colab-cuda-2026-10-03/cuda-recovery-report.json)
is the machine-readable evidence. The experiment used implementation revision
`4ffaf1da88f9092078d61ed3c801597eefcd0809` on 3 October 2026.

## Recorded run

The free Colab allocation used a Tesla T4, driver 580.82.07 and locked PyTorch 2.13.0+cu130 with
deterministic eager FP32. All 41 focused tests passed without skips, including nine
hardware-dependent cases. This is evidence for the observed environment, not equality across
different GPUs or releases.

The six-step run processed 384 training input tokens. It saved at step two, reconstructed fresh model
and optimizer objects, restored the cursor and CPU/CUDA RNG, and passed all 11 equality checks.
The bounded held-out evaluation used a CPU copy of the GPU-trained weights and 4,096 target tokens.

The original report's SHA-256 is
`13dca14329edf830df3da6cafdeb69c0834ec7cbc71f1d4daa2daf3330324ffe`.

This proves fresh-object checkpoint reconstruction within the same running process. It does not
prove GPU SIGKILL recovery, runtime-disconnection recovery, distributed training, model quality or
production readiness. The [runbook](colab-gpu.md#evidence-boundary) records the full claim boundary.

## Screenshot capture status

Screenshot capture was interrupted and has not been completed. No screenshot files are published here.
Capture and privacy review must finish before adding the hardware, environment, test-result,
recovery-equality and export-checksum images.
