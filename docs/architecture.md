# Architecture and recovery contract

## Goal

The lab verifies that a failed training attempt cannot advance durable progress unless the model,
optimizer, RNG state, sample history, and data cursor advance together. Recovery is correct only if
the resumed run is exactly equivalent to an uninterrupted control under the checked CPU contract.

## System boundary

```text
Pinned source artifact                     Versioned live feed
        |                                         |
        v                                         v
hash-before-decompress                    bounded fetch + retry
        |                                         |
        v                                         v
canonical documents.jsonl                 SQLite WAL capture ledger
        |                                         |
        +-------------------+---------------------+
                            v
                 immutable DatasetManifestV1
                            |
                            v
                deterministic window schedule
                            |
                            v
                      TrainingCursorV1
                            |
                            v
 batch -> forward/backward -> optimizer -> cursor/checkpoint commit
                            |
                            v
                  CheckpointManifestV2
                            |
                            v
                    RecoveryReportV1
```

Network acquisition and training are deliberately separated. The training path accepts only a
validated immutable manifest and canonical local documents, so a changing upstream source cannot
silently alter a resumed run.

## Versioned contracts

### `DatasetManifestV1`

It binds:

- source repository/feed, immutable revision or snapshot identity, and artifact hash;
- canonical document file size and SHA-256;
- document count, byte count, and token count;
- stable document IDs and mutually exclusive train/validation/test assignments;
- tokenizer and preprocessing fingerprints;
- license description and known source limitations.

The PEP adapter uses `utf8-byte-v1`: byte IDs `0..255`, end-of-document ID `256`, vocabulary size
`257`. A final anchored window ensures the tail and EOD target of every eligible document remain
reachable instead of being silently discarded.

### `TrainingCursorV1`

The cursor contains the epoch, document ID, token-window offset, batch ID, next sample IDs, and
absolute batch index. Sample identities are derived from stable document IDs and offsets. Batch
identity also includes its occurrence index, which prevents repeated windows from being mistaken
for the same committed step. Windows are deterministically interleaved across sorted document IDs
before later windows from the same document, so a short recovery smoke run crosses document
boundaries without allowing any individual window to span two documents.

### `CheckpointManifestV2`

Each checkpoint manifest binds:

- generation and completed step;
- serialized state byte length and SHA-256;
- model configuration, dataset, tokenizer, and run-contract fingerprints;
- cursor, tokens seen, loss history, and stable batch/sample histories;
- expected state keys and schema version.

### `RecoveryReportV1`

The report records the failure point, attempted and committed steps, selected generations, replayed
sample IDs, discarded compute, measured durable loss, checkpoint-selection time, replay-to-commit
time, normal post-recovery completion time, equality verdicts, environment, and limitations.

## Step transaction

```text
1. Read the durable cursor and prepare batch N.
2. Run forward and backward.
3. Apply optimizer update.
4. Advance cursor, histories, step, and token counter in memory.
5. Publish one checkpoint generation containing the complete new state.
6. Only then treat step N as committed.
```

A failure at steps 1–5 leaves the prior generation as the recovery point. Even when the optimizer
already changed in the failed process, the restarted process reloads the previous committed
optimizer and RNG state and reuses batch N.

## Durable checkpoint publication

```text
serialize state to unique temporary file
  -> flush + fsync file
  -> compute state size and SHA-256
  -> write + fsync CheckpointManifestV2
  -> atomic rename temporary generation
  -> fsync checkpoint directory
  -> atomically publish durable LATEST pointer
  -> fsync checkpoint directory
  -> retain newest two committed generations
```

Temporary or unreferenced generations are not committed. A malformed `LATEST` pointer fails closed
instead of deleting valid history. A new save cannot append a different data/config/tokenizer/run
contract into an existing store. Fresh training also refuses to overwrite a non-empty output path.

## Recovery selection

The loader starts from the committed `LATEST` generation and may fall back to the preceding
committed generation only when the newest candidate fails integrity validation. Configuration or
dataset/tokenizer/run-contract mismatches are compatibility errors, not corruption, and therefore
fail immediately rather than selecting unrelated state.

For a `CheckpointManifestV2` generation, the loader validates the pointer, schema, byte length,
state digest, expected state keys, and all compatibility fingerprints before `torch.load`.
Deserialization uses `weights_only=True` and the project requires PyTorch `>=2.10` because of the
security boundary described in [checkpoint-security.md](checkpoint-security.md). The explicit
legacy-v1 migration has no sidecar length or digest, accepts only the synthetic-data contract, and
must remain a trusted-local compatibility path.

## Equality proof

For every failure scenario, the verifier compares recovered and uninterrupted runs for:

- batch ID and sample/window ID sequences;
- loss sequence;
- model and optimizer tensor trees;
- Python, NumPy, and PyTorch CPU RNG state;
- final cursor, completed steps, and token count;
- verification logits;
- final state digest and run fingerprint.

Dropout is enabled in the real-data configuration so RNG restoration is observable. The matrix also
rejects a changed dataset byte, tokenizer identity, and model configuration, and verifies exact
fallback from a truncated newest generation.

## USGS ingestion adapter

The capture path is a bounded single-writer design:

```text
allowlisted USGS feed key
  -> bounded HTTPS response + retry policy
  -> strict finite-number/timestamp/schema validation
  -> SQLite WAL transaction
  -> UNIQUE(source, event_id, updated_at_ms)
  -> canonical content-addressed JSONL snapshot
  -> DatasetManifestV1 adapter
  -> unchanged training/recovery pipeline
```

An identical event version is idempotent. The same identity with different content is rejected and
the poll transaction rolls back, because silently accepting divergent payloads would destroy lineage.
Snapshot manifests use safe relative paths and bind source/feed metadata as well as content.

This design proves deterministic replay of captured versions. It does not implement online learning,
earthquake prediction, Kafka, object storage, PostgreSQL, or multiple capture workers.

## Deliberate non-goals for v0.2

- No GPU, NCCL, FSDP, or multi-host claim.
- No OS-kill, power-loss, or remote object-store durability claim.
- No model-quality or pretrained-model claim.
- No authenticated/signature-backed checkpoint claim.
- No production RPO/RTO or throughput claim.
