# Research basis and learning path

This lab is independently implemented. The sources below motivated invariants and test cases; no
code was copied, no upstream benchmark was reproduced, and none of their scale results are claimed
for this repository.

## Model and adapter references

- [Attention Is All You Need — Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) introduces
  the attention-based Transformer architecture. This lab uses those architectural concepts in an
  independently written decoder-only `TinyTransformer`, not the paper's complete encoder–decoder
  model or translation experiment. The PEP configuration starts from random weights and has
  34,720 parameters; it does not load a pretrained foundation model.
- [LoRA: Low-Rank Adaptation of Large Language Models — Hu et al., 2021](https://arxiv.org/abs/2106.09685)
  motivates frozen base weights with trainable low-rank updates. The current implementation adds
  adapters to attention projections and checks their mechanics on a synthetic CPU workload.
  No pretrained model is currently fine-tuned here, and the comparison is not evidence of
  downstream quality improvement. A pinned pretrained model and held-out evaluation remain a
  separate, unfinished evidence gate.

## Practical fault-tolerance reference

[PyTorch's fault-tolerant Llama experiment — Rice and Huang, 2025](https://pytorch.org/blog/fault-tolerant-llama-training-with-2000-synthetic-failures-every-15-seconds-and-no-checkpoints-on-crusoe-l40s/)
is an upstream systems experiment, not a paper whose results this lab reproduces. It demonstrates
TorchFT/TorchTitan training under injected failures using distributed replica groups and live peer
recovery. This lab instead verifies checkpoint/restart contracts on a small controlled workload.

Its own [CPU process-kill proof](process-recovery.md) and [single-T4 reconstruction report](../artifacts/colab-cuda-2026-10-03/cuda-recovery-report.json)
are separate evidence. The GPU run reconstructs fresh training objects in the same process; it is
not GPU process-death recovery. No TorchFT integration or multi-worker execution is claimed.

## Engineering signals

- [PyTorch on TorchFT and TorchTitan](https://x.com/PyTorch/status/1936131972285247504) shows why
  repeated synthetic failure testing is an active ML-infrastructure concern. It is an upstream
  experiment report, not evidence for this lab.
- [Stas Bekman on per-step fault tolerance](https://x.com/StasBekman/status/1875242682689450060)
  describes its infrastructure value while explicitly calling TorchFT a prototype.
- [Tristan Rice on failed-step batch reuse](https://x.com/rice_fry/status/1876412208940560688)
  motivates the invariant that an uncommitted step cannot consume its batch.
- [Tristan Rice on rebalancing](https://x.com/rice_fry/status/1876412619558641740) separates
  training-state recovery from the future problem of distributed shard reassignment.

X posts are treated as design signals and opinions. Executable behavior in this repository is
supported only by its own tests and versioned artifacts.

## Repositories and incidents studied

- [meta-pytorch/torchft](https://github.com/meta-pytorch/torchft) — per-step distributed recovery
  concepts. This lab does not claim TorchFT execution.
- [pytorch/torchtitan](https://github.com/pytorch/torchtitan) — full training-system architecture,
  reproducibility, and checkpointing patterns.
- [TorchTitan issue #3907](https://github.com/pytorch/torchtitan/issues/3907) — a second-resume bug
  where the first restore succeeded but a later checkpoint held stale dataloader state. It directly
  motivates the original → resume → save → second-resume regression in this lab.
- [meta-pytorch/data](https://github.com/meta-pytorch/data) — stateful dataloader interfaces and
  their distributed-rank limitations.
- [safetensors/safetensors](https://github.com/huggingface/safetensors) — tensor-only public weight
  formats and the distinction between weight export and a full optimizer/RNG resume checkpoint.
- [trailofbits/fickling](https://github.com/trailofbits/fickling) — why pickle-based ML artifacts
  must be treated as executable input and why full resume state remains trusted-local here.
- [PyTorch advisory GHSA-63cw-57p8-fm3p](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p)
  — the reason this project requires the patched PyTorch 2.10+ line even with
  `weights_only=True`.

These repositories are study references only. Stars were not applied automatically; that remains a
user-controlled GitHub action.

## Parallel upstream contributions

Status checked on 3 October 2026. These are contributions to separate repositories, not features
installed in this lab or evidence that an upstream maintainer endorsed its results.

- [TorchTitan PR #4912 — Type decoder layer configurations](https://github.com/pytorch/torchtitan/pull/4912):
  authored by `ajaykr0905`; open and unmerged at the status check.
- [TorchTitan PR #4864 — Collect CPU-safe RL tests](https://github.com/pytorch/torchtitan/pull/4864):
  authored by `ajaykr0905`; closed **without merging** on 2 October 2026. The maintainer superseded
  it with [PR #4889](https://github.com/pytorch/torchtitan/pull/4889); that replacement is not
  Ajay's merged contribution.
- [Higgsfield skills PR #12 — Make frontmatter portable to Codex](https://github.com/higgsfield-ai/skills/pull/12):
  authored by `ajaykr0905`; open and unmerged at the status check. This is skill/tooling
  portability work, separate from the lab's training-recovery implementation.

The studied [second-resume issue #3907](https://github.com/pytorch/torchtitan/issues/3907) was
reported by another contributor. Its connection to this lab is the regression scenario, not
authorship of the upstream report or fix. Live PR pages remain authoritative as statuses change.

## Learning progression

1. Prove single-process CPU transaction and replay invariants on immutable real data.
2. Capture a changing public feed, then train only on a sealed deterministic snapshot.
3. Add a pinned pretrained model and real LoRA evaluation without weakening recovery contracts.
4. Record actual GPU evidence, including hardware, memory, and determinism limits.
5. Test multi-process checkpointing and peer recovery separately from data-shard rebalancing.

Each gate must publish its own configuration, environment, raw result, reproduction command, and
limitations before the corresponding claim appears in public documentation.
