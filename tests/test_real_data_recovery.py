import gzip
import hashlib
from pathlib import Path

import pytest

from fttl.config import ExperimentConfig, ModelConfig
from fttl.dataset import DatasetSource, DatasetValidationError, prepare_peps
from fttl.recovery import verify_recovery
from fttl.train import run_training


FIXTURE = Path(__file__).parent / "fixtures" / "public_domain_peps_fixture.jsonl"


def real_data_config() -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=257,
            block_size=8,
            d_model=8,
            n_heads=2,
            n_layers=1,
            dropout=0.25,
        ),
        seed=23,
        steps=4,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=1,
    )


def prepared_fixture(tmp_path: Path) -> Path:
    archive = gzip.compress(FIXTURE.read_bytes(), mtime=0)
    source = DatasetSource(
        repository="fttl://tests/public-domain-pep-fixture",
        revision="fixture-v1",
        path="public_domain_peps_fixture.jsonl.gz",
        compressed_sha256=hashlib.sha256(archive).hexdigest(),
        expected_document_count=6,
        license="CC0-1.0",
        license_limitations=("Independently written CI fixture.",),
    )

    def downloader(_url: str, destination: Path) -> None:
        destination.write_bytes(archive)

    output = tmp_path / "prepared"
    prepare_peps(tmp_path / "cache", output, downloader=downloader, source=source)
    return output / "manifest.json"


def test_two_consecutive_restarts_match_uninterrupted_real_data_run(tmp_path: Path):
    report = verify_recovery(
        real_data_config(),
        prepared_fixture(tmp_path / "dataset"),
        tmp_path / "evidence",
        failure_point="after-optimizer",
        restarts=2,
    )

    assert report.exact_equality
    assert report.completed_restarts == 2
    assert report.rpo_lost_steps == 2
    assert report.rpo_lost_tokens == 32
    assert all(attempt.sample_ids == attempt.replayed_sample_ids for attempt in report.attempts)
    assert all(report.equality.values())
    assert (tmp_path / "evidence" / "recovery-report.json").is_file()


@pytest.mark.parametrize(
    "failure_point",
    ["before-forward", "after-backward", "during-checkpoint-write"],
)
def test_each_transaction_boundary_replays_the_same_batch(
    tmp_path: Path, failure_point: str
):
    report = verify_recovery(
        real_data_config(),
        prepared_fixture(tmp_path / "dataset"),
        tmp_path / "evidence",
        failure_point=failure_point,
        restarts=1,
    )

    assert report.exact_equality
    assert report.attempts[0].sample_ids == report.attempts[0].replayed_sample_ids


def test_real_dataset_manifest_and_cursor_are_bound_to_the_checkpoint(tmp_path: Path):
    config = real_data_config()
    manifest = prepared_fixture(tmp_path / "dataset")
    _, partial = run_training(
        config,
        tmp_path / "run",
        dataset_manifest=manifest,
        stop_after_step=2,
    )
    _, resumed = run_training(
        config,
        tmp_path / "run",
        dataset_manifest=manifest,
        resume_from=Path(partial.checkpoint),
    )

    assert resumed.steps == 4
    assert resumed.final_cursor["batch_index"] == 4
    assert len(resumed.sample_ids) == len(resumed.batch_ids) == 4
    assert resumed.data_fingerprint == partial.data_fingerprint


def test_one_changed_dataset_byte_is_rejected_before_resume(tmp_path: Path):
    config = real_data_config()
    manifest = prepared_fixture(tmp_path / "dataset")
    _, partial = run_training(
        config,
        tmp_path / "run",
        dataset_manifest=manifest,
        stop_after_step=2,
    )
    documents = manifest.parent / "documents.jsonl"
    content = documents.read_bytes()
    documents.write_bytes(content[:-2] + b"x\n")

    with pytest.raises(DatasetValidationError, match="SHA-256|byte length"):
        run_training(
            config,
            tmp_path / "run",
            dataset_manifest=manifest,
            resume_from=Path(partial.checkpoint),
        )
