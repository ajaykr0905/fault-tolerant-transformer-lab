from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from fttl.checkpoint import CheckpointMismatchError
from fttl.config import ExperimentConfig
from fttl.dataset import DatasetValidationError, load_dataset_manifest
from fttl.recovery import RecoveryReportV1, verify_recovery
from fttl.train import FAILURE_POINTS, run_training


@dataclass(frozen=True)
class MatrixScenarioV1:
    failure_point: str
    exact_equality: bool
    completed_restarts: int
    replayed_steps: int
    replayed_tokens: int
    discarded_compute_tokens: int
    checkpoint_selection_load_seconds: float
    recovery_duration_seconds: float
    ordinary_completion_seconds: float
    final_state_digest: str
    report_path: str


@dataclass(frozen=True)
class RecoveryMatrixV1:
    schema: str
    config_fingerprint: str
    dataset_fingerprint: str
    tokenizer_fingerprint: str
    run_contract_fingerprint: str
    code_fingerprint: str
    code_revision: str
    completed_steps: int
    tokens_seen: int
    sample_windows: int
    unique_documents_sampled: int
    scenarios: tuple[MatrixScenarioV1, ...]
    integrity_checks: dict[str, bool]
    all_passed: bool
    limitations: tuple[str, ...]


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _scenario(report: RecoveryReportV1, report_path: str) -> MatrixScenarioV1:
    return MatrixScenarioV1(
        failure_point=report.failure_point,
        exact_equality=report.exact_equality,
        completed_restarts=report.completed_restarts,
        replayed_steps=report.replayed_steps,
        replayed_tokens=report.replayed_tokens,
        discarded_compute_tokens=report.discarded_compute_tokens,
        checkpoint_selection_load_seconds=report.checkpoint_selection_load_seconds,
        recovery_duration_seconds=report.recovery_duration_seconds,
        ordinary_completion_seconds=report.ordinary_completion_seconds,
        final_state_digest=report.final_state_digest,
        report_path=report_path,
    )


def _changed_dataset_is_rejected(
    config: ExperimentConfig, dataset_manifest: Path, scratch: Path
) -> bool:
    copied = scratch / "changed-dataset"
    shutil.copytree(dataset_manifest.parent, copied)
    manifest = copied / dataset_manifest.name
    parsed = load_dataset_manifest(manifest)
    documents = copied / str(parsed.documents["path"])
    content = documents.read_bytes()
    documents.write_bytes(content[:-2] + b"x\n")
    try:
        run_training(config, scratch / "changed-data-run", dataset_manifest=manifest)
    except DatasetValidationError:
        return True
    return False


def _changed_tokenizer_is_rejected(
    config: ExperimentConfig, dataset_manifest: Path, scratch: Path
) -> bool:
    copied = scratch / "changed-tokenizer"
    shutil.copytree(dataset_manifest.parent, copied)
    manifest = copied / dataset_manifest.name
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["tokenizer"]["fingerprint"] = "0" * 64
    identity = dict(payload)
    identity.pop("dataset_fingerprint")
    payload["dataset_fingerprint"] = hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    try:
        run_training(config, scratch / "changed-tokenizer-run", dataset_manifest=manifest)
    except DatasetValidationError:
        return True
    return False


def _changed_model_config_is_rejected(
    config: ExperimentConfig, dataset_manifest: Path, scratch: Path
) -> bool:
    run_dir = scratch / "changed-config-run"
    _, partial = run_training(
        config,
        run_dir,
        dataset_manifest=dataset_manifest,
        stop_after_step=1,
    )
    changed_dropout = 0.1 if config.model.dropout == 0 else config.model.dropout / 2
    changed = replace(config, model=replace(config.model, dropout=changed_dropout))
    try:
        run_training(
            changed,
            run_dir,
            dataset_manifest=dataset_manifest,
            resume_from=run_dir / partial.checkpoint,
        )
    except CheckpointMismatchError:
        return True
    return False


def _truncated_newest_state_falls_back_exactly(
    config: ExperimentConfig, dataset_manifest: Path, scratch: Path
) -> bool:
    control_dir = scratch / "state-fallback-control"
    recovered_dir = scratch / "state-fallback-recovered"
    _, control = run_training(config, control_dir, dataset_manifest=dataset_manifest)
    _, partial = run_training(
        config,
        recovered_dir,
        dataset_manifest=dataset_manifest,
        stop_after_step=2,
    )
    store = recovered_dir / partial.checkpoint
    latest = (store / "LATEST").read_text(encoding="utf-8").strip()
    newest_state = store / latest / "state.pt"
    newest_state.write_bytes(newest_state.read_bytes()[:64])
    _, recovered = run_training(
        config,
        recovered_dir,
        dataset_manifest=dataset_manifest,
        resume_from=store,
    )
    return recovered.recovered_from_generation == 1 and (
        recovered.final_state_digest == control.final_state_digest
    )


def _corrupt_newest_manifest_falls_back_exactly(
    config: ExperimentConfig, dataset_manifest: Path, scratch: Path
) -> bool:
    control_dir = scratch / "manifest-fallback-control"
    recovered_dir = scratch / "manifest-fallback-recovered"
    _, control = run_training(config, control_dir, dataset_manifest=dataset_manifest)
    _, partial = run_training(
        config,
        recovered_dir,
        dataset_manifest=dataset_manifest,
        stop_after_step=2,
    )
    store = recovered_dir / partial.checkpoint
    latest = (store / "LATEST").read_text(encoding="utf-8").strip()
    newest_manifest = store / latest / "manifest.json"
    newest_manifest.write_text("{\"truncated\":", encoding="utf-8")
    _, recovered = run_training(
        config,
        recovered_dir,
        dataset_manifest=dataset_manifest,
        resume_from=store,
    )
    return recovered.recovered_from_generation == 1 and (
        recovered.final_state_digest == control.final_state_digest
    )


def verify_recovery_matrix(
    config: ExperimentConfig,
    dataset_manifest: Path,
    output_dir: Path,
    *,
    restarts: int = 2,
) -> RecoveryMatrixV1:
    if output_dir.exists():
        raise ValueError("matrix output already exists; choose a fresh directory")
    output_dir.mkdir(parents=True)
    scenarios: list[MatrixScenarioV1] = []
    first_report: RecoveryReportV1 | None = None
    for failure_point in FAILURE_POINTS:
        relative = Path("scenarios") / failure_point
        report = verify_recovery(
            config,
            dataset_manifest,
            output_dir / relative,
            failure_point=failure_point,
            restarts=restarts,
        )
        first_report = first_report or report
        scenarios.append(_scenario(report, str(relative / "recovery-report.json")))
    assert first_report is not None

    with tempfile.TemporaryDirectory(prefix="fttl-matrix-") as temporary:
        scratch = Path(temporary)
        integrity_checks = {
            "changed_dataset_byte_rejected": _changed_dataset_is_rejected(
                config, dataset_manifest, scratch
            ),
            "changed_tokenizer_rejected": _changed_tokenizer_is_rejected(
                config, dataset_manifest, scratch
            ),
            "changed_model_config_rejected": _changed_model_config_is_rejected(
                config, dataset_manifest, scratch
            ),
            "truncated_newest_state_falls_back_exactly": (
                _truncated_newest_state_falls_back_exactly(
                    config, dataset_manifest, scratch
                )
            ),
            "corrupt_newest_manifest_falls_back_exactly": (
                _corrupt_newest_manifest_falls_back_exactly(
                config, dataset_manifest, scratch
                )
            ),
        }

    matrix = RecoveryMatrixV1(
        schema="RecoveryMatrixV1",
        config_fingerprint=first_report.config_fingerprint,
        dataset_fingerprint=first_report.dataset_fingerprint,
        tokenizer_fingerprint=first_report.tokenizer_fingerprint,
        run_contract_fingerprint=first_report.run_contract_fingerprint,
        code_fingerprint=first_report.code_fingerprint,
        code_revision=first_report.code_revision,
        completed_steps=first_report.completed_steps,
        tokens_seen=first_report.tokens_seen,
        sample_windows=first_report.sample_windows,
        unique_documents_sampled=first_report.unique_documents_sampled,
        scenarios=tuple(scenarios),
        integrity_checks=integrity_checks,
        all_passed=all(scenario.exact_equality for scenario in scenarios)
        and all(integrity_checks.values()),
        limitations=(
            "Failures are deterministic Python crash surrogates, not OS-kill or power-loss tests.",
            "Results prove this pinned CPU run, not model quality, GPU scale, or production readiness.",
            "SHA-256 detects accidental corruption but does not authenticate attacker-controlled artifacts.",
        ),
    )
    _write_json(output_dir / "matrix.json", asdict(matrix))
    _write_json(
        output_dir / "dataset-manifest.json",
        json.loads(dataset_manifest.read_text(encoding="utf-8")),
    )
    if not matrix.all_passed:
        raise AssertionError("one or more recovery-matrix checks failed")
    return matrix
