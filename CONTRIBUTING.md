# Contributing

This lab values small, reproducible changes that make training recovery easier to
inspect. A contribution should state the failure or correctness property it tests,
provide deterministic evidence, and describe what the result does **not** prove.

## Set up the locked environment

Use Python 3.12 and [`uv`](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync --frozen --extra test
```

The committed `uv.lock` is the dependency contract. If a dependency must change,
edit `pyproject.toml`, run `uv lock`, inspect the resolved versions, and commit both
files together.

## Run the same checks as CI

```bash
uv run --frozen ruff check .
uv run --frozen python -m compileall -q src tests
uv run --frozen pytest
uv run --frozen pytest tests/test_real_data_recovery.py::test_complete_matrix_is_public_safe_and_passes_all_scenarios
uv run --frozen pytest tests/test_process_recovery.py
git diff --check
```

The named recovery-matrix test prepares the small, independently written CC0
fixture in `tests/fixtures/`. It must remain network-free; CI never downloads the
Common Pile corpus.

The process-recovery tests use POSIX `SIGKILL` against spawned workers at all four
transaction boundaries. They verify exact independent-process state equality,
bounded startup/completion deadlines, child cleanup, dataset mismatch rejection,
and the installed CLI. See [the operating instructions](docs/process-recovery.md).

Exercise the installed command-line interfaces with fresh output paths:

```bash
uv run --frozen fttl-train-smoke \
  --config configs/smoke.json \
  --output artifacts/local/smoke-001

uv run --frozen fttl-compare-tuning \
  --config configs/smoke.json \
  --output artifacts/local/tuning-001/result.json
```

Checkpoint directories are stateful evidence stores. Do not reuse an output path
for an unrelated or restarted experiment; choose a new run identifier instead.
Local checkpoints, prepared corpora, and caches are intentionally ignored by Git.

## Data and public-safety rules

- Use only independently written code and data whose public use and redistribution
  terms are documented.
- Never commit credentials, private data, employer code, internal identifiers, or
  downloaded corpora.
- Keep network access behind an explicit preparation or ingestion command. Tests
  use committed, minimal fixtures and must run without the network.
- Reject checksum, schema, tokenizer, configuration, and split mismatches instead
  of silently repairing them.
- Commit only compact manifests and machine-readable reports needed to reproduce a
  claim; do not commit binary checkpoints or generated bulk data.

## Claim boundary

Current recovery evidence demonstrates deterministic CPU behavior for this lab:
model, optimizer, RNG, cursor, samples, losses, token counts, logits, and final
state can be compared across controlled failures. It is not evidence of model
quality, GPU or multi-worker scale, production readiness, online learning, or an
authenticated checkpoint supply chain. New claims require matching tests and a
versioned artifact before they appear in documentation.

## Changes and commits

1. Keep each commit coherent: one contract, implementation, test, or evidence
   update with a descriptive message.
2. Add regression tests for correctness changes and failure-boundary tests for
   recovery changes.
3. Preserve existing public interfaces unless the change documents a migration.
4. In the pull request, list the commands run, their results, the generated
   evidence files, and the remaining limitations.
5. Confirm `git status`, `git diff --check`, and the secret-sensitive data rules
   above before requesting review.
