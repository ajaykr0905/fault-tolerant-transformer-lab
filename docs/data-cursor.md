# Source-bound batch cursors

`TrainingCursorV1` keeps its existing fields and sample IDs. Batch IDs now use
`source-bound-batch-v2`: SHA-256 of canonical JSON containing `contract`,
`batch_index` and the ordered `sample_ids` list. JSON uses UTF-8, sorted keys,
compact separators, `ensure_ascii=False` and `allow_nan=False`.

The contract binds source kind, dataset/tokenizer fingerprints, actual content,
block size and batch size. Prepared sources capture each document ID/text once;
all captured document bytes, including unsampled documents, affect the content
fingerprint. Synthetic batches hash and emit the same captured token tensor.
`batch_identity_contract` returns an isolated metadata dictionary.

Old batch IDs intentionally differ. Existing v1-shaped cursors with the old IDs
are rejected; do not rewrite old reports or checkpoints. The changed code
fingerprint changes the training run contract, so old verified runs cannot be
resumed as if they used this algorithm. Start a fresh run with this version.

Hashes establish reproducible content identity, not authentication or proof
that caller-provided licensing/source claims are true.
