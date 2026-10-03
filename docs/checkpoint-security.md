# Checkpoint integrity and security boundary

## Threat model

This release protects against accidental truncation, partial publication, bit corruption, stale or
incompatible local state, and malformed metadata. It does not claim to protect checkpoint integrity
from an attacker who can replace both a checkpoint and its manifest.

Full training resume requires optimizer and RNG state in addition to model weights. Those files are
therefore treated as **trusted local artifacts** created by this process, stored outside Git, and
loaded only after their surrounding contract is validated.

## Controls implemented

- PyTorch is constrained to `>=2.10,<3`.
- Every load uses `torch.load(..., weights_only=True)`.
- For v2 generation stores, state byte length and SHA-256 are checked before deserialization.
- The manifest schema, expected keys, model configuration, dataset, tokenizer, and run contract are
  validated before state is accepted.
- Rejected model, optimizer, or RNG state restores the caller's original training objects and random
  generators. Loading takes in-memory copies for rollback, so allow space for another model and
  optimizer state. This guard is intended for this lab's small CPU models, not a GPU-scale claim.
- Generation publication uses unique temporary paths, file and directory `fsync`, atomic renames,
  and a durable, atomically replaced `LATEST` commit record.
- Only committed generations are fallback candidates.
- A damaged commit pointer fails closed; a save does not erase generations it cannot classify.
- The two newest valid generations are retained so a newest generation with unreadable bytes,
  invalid integrity metadata, or an undecodable payload can fall back safely. After selection,
  model shape, optimizer restoration, or RNG semantic failures reject the load and restore the
  caller; they do not retry an older generation.
- Fresh training cannot overwrite a non-empty run directory.

The explicit legacy-v1 migration does not have a sidecar length or digest. It accepts only the
synthetic-data compatibility contract, still uses `weights_only=True`, and remains trusted-local.

## Why PyTorch 2.10 is the floor

PyTorch published [GHSA-63cw-57p8-fm3p](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p),
an arbitrary-code-execution advisory affecting `torch.load(..., weights_only=True)` through 2.9.1.
The package floor is therefore a security requirement, not a formatting or convenience choice.

This does not make untrusted checkpoint loading safe by itself. The project still refuses to present
downloaded or third-party resume checkpoints as supported input.

## What SHA-256 does and does not prove

A content digest detects accidental change when the expected digest is trustworthy. If an attacker
can replace both `state.pt` and `manifest.json`, the attacker can calculate a matching new digest.
SHA-256 in this design is an integrity check, not origin authentication.

A later milestone may sign canonical manifests and verify them against an out-of-band public key.
Until that exists, documentation and reports must not call checkpoints signed, authenticated, or
tamper-proof.

## Operational guidance

- Keep `.cache/`, `artifacts/**/checkpoints/`, and all `*.pt` files out of Git.
- Use a new output directory for each fresh run.
- Resume only from a checkpoint store created locally by the same pinned code/data contract.
- Treat a compatibility mismatch as a new experiment, not a reason to bypass validation.
- Publish JSON reports and manifests; do not publish optimizer-bearing binary state.
