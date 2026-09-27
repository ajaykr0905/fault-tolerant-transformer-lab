# Research basis and learning path

This lab is independently implemented. The sources below motivated invariants and test cases; no
code was copied, no upstream benchmark was reproduced, and none of their scale results are claimed
for this repository.

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
  concepts. This CPU lab does not claim TorchFT execution.
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

## Learning progression

1. Prove single-process CPU transaction and replay invariants on immutable real data.
2. Capture a changing public feed, then train only on a sealed deterministic snapshot.
3. Add a pinned pretrained model and real LoRA evaluation without weakening recovery contracts.
4. Record actual GPU evidence, including hardware, memory, and determinism limits.
5. Test multi-process checkpointing and peer recovery separately from data-shard rebalancing.

Each gate must publish its own configuration, environment, raw result, reproduction command, and
limitations before the corresponding claim appears in public documentation.
