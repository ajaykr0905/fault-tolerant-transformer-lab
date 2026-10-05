"""Durable, integrity-checked checkpoints for trusted local training runs.

The SHA-256 manifest in this module detects accidental truncation or corruption;
it is not an authenticity mechanism. A party that can replace both a state file
and its manifest can still substitute a checkpoint. Callers must therefore only
resume checkpoints from a trusted local artifact store. State files are always
opened with ``torch.load(..., weights_only=True)``.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import re
import shutil
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from fttl.config import ExperimentConfig
from fttl.numerical import require_finite_state
from fttl.state import capture_rng_state, restore_rng_state

try:
    import fcntl
except ImportError:
    fcntl = None

CHECKPOINT_SCHEMA_VERSION = 2
DEFAULT_DATA_FINGERPRINT = "synthetic-token-stream-v1"
DEFAULT_TOKENIZER_FINGERPRINT = "synthetic-tokenizer-v1"
DEFAULT_RUN_CONTRACT_FINGERPRINT = "training-contract-v2"
STATE_FILENAME = "state.pt"
MANIFEST_FILENAME = "manifest.json"
LATEST_FILENAME = "LATEST"
WRITER_LOCK_FILENAME = ".writer.lock"
_GENERATION_PATTERN = re.compile(r"^generation-(\d{8})$")
_REQUIRED_STATE_KEYS = frozenset(
    {
        "schema_version",
        "config",
        "config_fingerprint",
        "data_fingerprint",
        "tokenizer_fingerprint",
        "run_contract_fingerprint",
        "model",
        "optimizer",
        "step",
        "tokens_seen",
        "losses",
        "cursor",
        "batch_ids",
        "sample_ids",
        "rng_state",
    }
)


class CheckpointMismatchError(ValueError):
    """Raised when a valid checkpoint belongs to a different training contract."""


class CheckpointIntegrityError(OSError):
    """Raised when no committed generation passes structural integrity checks."""


FailureInjector = Callable[[str], None]


def _validate_schema(value: Any, expected: int, label: str) -> None:
    if type(value) is not int:
        raise CheckpointIntegrityError(f"checkpoint {label} schema must be an integer")
    if value != expected:
        raise CheckpointMismatchError(f"unsupported checkpoint {label} schema")


def _validate_counter(value: Any, label: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise CheckpointIntegrityError(
            f"checkpoint {label} must be an integer greater than or equal to {minimum}"
        )


def _validate_cursor(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise CheckpointIntegrityError("checkpoint cursor must be an object or null")
    # Generic history cursors remain supported; versioned training cursors use
    # the same strict constructor as their data-source consumers.
    if "schema_version" in value:
        from fttl.data import TrainingCursorV1

        try:
            TrainingCursorV1.from_dict(value)
        except (TypeError, ValueError) as error:
            raise CheckpointIntegrityError(
                f"checkpoint training cursor is invalid: {error}"
            ) from error


@dataclass(frozen=True)
class CheckpointManifestV2:
    """Public metadata binding one checkpoint generation to its training inputs."""

    schema_version: int
    generation: int
    state_file: str
    state_sha256: str
    state_bytes: int
    completed_step: int
    tokens_seen: int
    config_fingerprint: str
    data_fingerprint: str
    tokenizer_fingerprint: str
    run_contract_fingerprint: str
    state_keys: tuple[str, ...]
    cursor: dict[str, Any] | None

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, CHECKPOINT_SCHEMA_VERSION, "manifest")
        _validate_counter(self.generation, "generation", 1)
        _validate_counter(self.state_bytes, "byte length", 1)
        _validate_counter(self.completed_step, "completed step")
        _validate_counter(self.tokens_seen, "token count")
        _validate_cursor(self.cursor)

    @property
    def dataset_fingerprint(self) -> str:
        """Compatibility alias for the more concise data fingerprint name."""

        return self.data_fingerprint

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["state_keys"] = list(self.state_keys)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CheckpointManifestV2":
        required = {
            "schema_version",
            "generation",
            "state_file",
            "state_sha256",
            "state_bytes",
            "completed_step",
            "tokens_seen",
            "config_fingerprint",
            "data_fingerprint",
            "tokenizer_fingerprint",
            "run_contract_fingerprint",
            "state_keys",
            "cursor",
        }
        missing = sorted(required.difference(value))
        if missing:
            raise CheckpointIntegrityError(
                f"checkpoint manifest is missing keys: {', '.join(missing)}"
            )
        _validate_schema(value["schema_version"], CHECKPOINT_SCHEMA_VERSION, "manifest")
        if not isinstance(value["state_keys"], list) or not all(
            isinstance(item, str) for item in value["state_keys"]
        ):
            raise CheckpointIntegrityError("checkpoint state key list is malformed")
        if value["cursor"] is not None and not isinstance(value["cursor"], dict):
            raise CheckpointIntegrityError("checkpoint cursor must be an object or null")
        string_fields = (
            "state_file",
            "state_sha256",
            "config_fingerprint",
            "data_fingerprint",
            "tokenizer_fingerprint",
            "run_contract_fingerprint",
        )
        if any(not isinstance(value[field], str) for field in string_fields):
            raise CheckpointIntegrityError("checkpoint manifest strings are malformed")
        return cls(
            schema_version=value["schema_version"],
            generation=value["generation"],
            state_file=value["state_file"],
            state_sha256=value["state_sha256"],
            state_bytes=value["state_bytes"],
            completed_step=value["completed_step"],
            tokens_seen=value["tokens_seen"],
            config_fingerprint=value["config_fingerprint"],
            data_fingerprint=value["data_fingerprint"],
            tokenizer_fingerprint=value["tokenizer_fingerprint"],
            run_contract_fingerprint=value["run_contract_fingerprint"],
            state_keys=tuple(value["state_keys"]),
            cursor=value["cursor"],
        )


def _inject(failure_injector: FailureInjector | None, stage: str) -> None:
    if failure_injector is not None:
        failure_injector(stage)


def _json_compatible(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_compatible(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"checkpoint metadata contains unsupported value {type(value)!r}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_durable_text(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generation_number(path: Path) -> int | None:
    match = _GENERATION_PATTERN.fullmatch(path.name)
    return int(match.group(1)) if match else None


def _generation_directories(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        (
            child
            for child in root.iterdir()
            if child.is_dir() and _generation_number(child) is not None
        ),
        key=lambda child: _generation_number(child) or -1,
        reverse=True,
    )


def _cleanup_stale_temporaries(root: Path) -> None:
    for child in root.glob(".generation-*.tmp-*"):
        if child.is_dir():
            shutil.rmtree(child)
    for child in root.glob(f".{LATEST_FILENAME}.tmp-*"):
        if child.is_file():
            child.unlink()


def _cleanup_uncommitted_generations(root: Path) -> None:
    """Discard generations published before a crash that never reached LATEST.

    LATEST is the commit record. Without this cleanup, an orphaned generation
    could become a future fallback even though it was never committed.
    """

    generations = _generation_directories(root)
    latest_path = root / LATEST_FILENAME
    if not latest_path.is_file():
        if generations:
            raise CheckpointIntegrityError(
                "checkpoint generations exist without a valid LATEST pointer; manual repair required"
            )
        return
    try:
        latest_name = latest_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as error:
        raise CheckpointIntegrityError("checkpoint LATEST pointer is unreadable") from error
    match = _GENERATION_PATTERN.fullmatch(latest_name)
    if match is None:
        raise CheckpointIntegrityError("checkpoint LATEST pointer is malformed")
    latest_number = int(match.group(1))
    if not any(path.name == latest_name for path in generations):
        raise CheckpointIntegrityError("checkpoint LATEST generation is missing")
    for generation_dir in generations:
        generation = _generation_number(generation_dir)
        if generation is not None and generation > latest_number:
            shutil.rmtree(generation_dir)
    _fsync_directory(root)


def _assert_store_contract(
    root: Path,
    *,
    config_fingerprint: str,
    data_fingerprint: str,
    tokenizer_fingerprint: str,
    run_contract_fingerprint: str,
) -> None:
    if not (root / LATEST_FILENAME).is_file():
        return
    integrity_errors: list[CheckpointIntegrityError] = []
    for generation_dir in _candidate_generation_directories(root):
        try:
            manifest = _validate_generation_files(generation_dir)
        except CheckpointIntegrityError as error:
            integrity_errors.append(error)
            continue
        _assert_fingerprints(
            manifest,
            expected_config_fingerprint=config_fingerprint,
            expected_data_fingerprint=data_fingerprint,
            expected_tokenizer_fingerprint=tokenizer_fingerprint,
            expected_run_contract_fingerprint=run_contract_fingerprint,
        )
        return
    detail = str(integrity_errors[0]) if integrity_errors else "unknown error"
    raise CheckpointIntegrityError(
        f"checkpoint store has no valid generation contract: {detail}"
    )


def _next_generation(root: Path) -> int:
    numbers = [
        number
        for child in root.iterdir()
        if (number := _generation_number(child)) is not None
    ]
    return max(numbers, default=0) + 1


def _read_manifest(path: Path) -> CheckpointManifestV2:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointIntegrityError("checkpoint manifest is unreadable") from error
    if not isinstance(raw, dict):
        raise CheckpointIntegrityError("checkpoint manifest must be a JSON object")
    return CheckpointManifestV2.from_dict(raw)


def _require_dense_checkpoint_state(value: Any, label: str) -> None:
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.device.type == "meta":
            raise ValueError(f"{label} must be a materialized dense tensor")
    elif isinstance(value, Mapping):
        for name, item in value.items():
            _require_dense_checkpoint_state(item, f"{label}.{name}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _require_dense_checkpoint_state(item, f"{label}[{index}]")


def _validate_checkpoint_numerics(value: Any) -> None:
    _require_dense_checkpoint_state(value, "checkpoint state")
    require_finite_state(value, "checkpoint state")


def _validate_stored_numerics(payload: dict[str, Any]) -> None:
    try:
        _validate_checkpoint_numerics(payload)
    except (FloatingPointError, ValueError, TypeError, RuntimeError, OverflowError) as error:
        raise CheckpointIntegrityError(f"checkpoint numerical state is invalid: {error}") from error


def _validate_generation_files(generation_dir: Path) -> CheckpointManifestV2:
    manifest = _read_manifest(generation_dir / MANIFEST_FILENAME)
    expected_name = f"generation-{manifest.generation:08d}"
    if generation_dir.name != expected_name:
        raise CheckpointIntegrityError("checkpoint generation name does not match manifest")
    if manifest.state_file != STATE_FILENAME:
        raise CheckpointIntegrityError("checkpoint state path is not the expected local file")
    if frozenset(manifest.state_keys) != _REQUIRED_STATE_KEYS:
        raise CheckpointIntegrityError("checkpoint manifest state keys are incomplete")
    state_path = generation_dir / manifest.state_file
    try:
        actual_length = state_path.stat().st_size
    except OSError as error:
        raise CheckpointIntegrityError("checkpoint state file is missing") from error
    if actual_length != manifest.state_bytes:
        raise CheckpointIntegrityError("checkpoint state byte length does not match manifest")
    try:
        actual_sha256 = _sha256(state_path)
    except OSError as error:
        raise CheckpointIntegrityError("checkpoint state file is unreadable") from error
    if actual_sha256 != manifest.state_sha256:
        raise CheckpointIntegrityError("checkpoint state SHA-256 does not match manifest")
    return manifest


def _candidate_generation_directories(root: Path) -> list[Path]:
    generations = _generation_directories(root)
    latest_path = root / LATEST_FILENAME
    if not latest_path.is_file():
        return []
    try:
        latest_name = latest_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return []
    match = _GENERATION_PATTERN.fullmatch(latest_name)
    if match is None:
        return []
    latest_number = int(match.group(1))
    by_name = {path.name: path for path in generations}
    candidates: list[Path] = []
    if latest_name in by_name:
        candidates.append(by_name[latest_name])
    candidates.extend(
        path
        for path in generations
        if (_generation_number(path) or 0) < latest_number
    )
    return candidates


def _assert_fingerprints(
    manifest: CheckpointManifestV2,
    *,
    expected_config_fingerprint: str,
    expected_data_fingerprint: str | None,
    expected_tokenizer_fingerprint: str | None,
    expected_run_contract_fingerprint: str | None,
) -> None:
    if manifest.config_fingerprint != expected_config_fingerprint:
        raise CheckpointMismatchError(
            "checkpoint configuration does not match experiment"
        )
    if (
        expected_data_fingerprint is not None
        and manifest.data_fingerprint != expected_data_fingerprint
    ):
        raise CheckpointMismatchError("checkpoint data fingerprint does not match")
    if (
        expected_tokenizer_fingerprint is not None
        and manifest.tokenizer_fingerprint != expected_tokenizer_fingerprint
    ):
        raise CheckpointMismatchError("checkpoint tokenizer fingerprint does not match")
    if (
        expected_run_contract_fingerprint is not None
        and manifest.run_contract_fingerprint != expected_run_contract_fingerprint
    ):
        raise CheckpointMismatchError(
            "checkpoint run contract fingerprint does not match"
        )


def _load_generation_payload(
    generation_dir: Path,
    *,
    model: torch.nn.Module,
    expected_config_fingerprint: str,
    expected_data_fingerprint: str | None,
    expected_tokenizer_fingerprint: str | None,
    expected_run_contract_fingerprint: str | None,
) -> tuple[dict[str, Any], CheckpointManifestV2]:
    manifest = _validate_generation_files(generation_dir)
    _assert_fingerprints(
        manifest,
        expected_config_fingerprint=expected_config_fingerprint,
        expected_data_fingerprint=expected_data_fingerprint,
        expected_tokenizer_fingerprint=expected_tokenizer_fingerprint,
        expected_run_contract_fingerprint=expected_run_contract_fingerprint,
    )
    try:
        state_snapshot = (generation_dir / manifest.state_file).read_bytes()
    except OSError as error:
        raise CheckpointIntegrityError("checkpoint state snapshot is unreadable") from error
    if len(state_snapshot) != manifest.state_bytes:
        raise CheckpointIntegrityError("checkpoint state snapshot byte length does not match manifest")
    if hashlib.sha256(state_snapshot).hexdigest() != manifest.state_sha256:
        raise CheckpointIntegrityError("checkpoint state snapshot SHA-256 does not match manifest")
    try:
        payload = torch.load(
            io.BytesIO(state_snapshot),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise CheckpointIntegrityError(
            "checkpoint state could not be decoded safely"
        ) from error
    if not isinstance(payload, dict):
        raise CheckpointIntegrityError("checkpoint state must be a mapping")
    missing = sorted(_REQUIRED_STATE_KEYS.difference(payload))
    if missing:
        raise CheckpointIntegrityError(
            f"checkpoint state is missing keys: {', '.join(missing)}"
        )
    _validate_schema(payload["schema_version"], CHECKPOINT_SCHEMA_VERSION, "state")
    _validate_counter(payload["step"], "completed step")
    _validate_counter(payload["tokens_seen"], "token count")
    _validate_cursor(payload["cursor"])
    if payload.get("config_fingerprint") != manifest.config_fingerprint:
        raise CheckpointIntegrityError("checkpoint state and manifest configuration differ")
    if payload.get("data_fingerprint") != manifest.data_fingerprint:
        raise CheckpointIntegrityError("checkpoint state and manifest data differ")
    if payload.get("tokenizer_fingerprint") != manifest.tokenizer_fingerprint:
        raise CheckpointIntegrityError("checkpoint state and manifest tokenizer differ")
    if payload.get("run_contract_fingerprint") != manifest.run_contract_fingerprint:
        raise CheckpointIntegrityError("checkpoint state and manifest run contract differ")
    if payload.get("step") != manifest.completed_step:
        raise CheckpointIntegrityError("checkpoint state and manifest steps differ")
    if payload.get("tokens_seen") != manifest.tokens_seen:
        raise CheckpointIntegrityError("checkpoint state and manifest token counts differ")
    if payload.get("cursor") != manifest.cursor:
        raise CheckpointIntegrityError("checkpoint state and manifest cursors differ")
    _validate_stored_numerics(payload)
    _validate_model_aliases(model, payload["model"])
    return payload, manifest


def _validate_model_aliases(model: torch.nn.Module, state: Mapping[str, Any]) -> None:
    """Reject contradictory values for declared shared parameter/buffer objects."""
    aliases: dict[int, list[str]] = {}
    for name, tensor in (
        *model.named_parameters(remove_duplicate=False),
        *model.named_buffers(remove_duplicate=False),
    ):
        if name in state:
            aliases.setdefault(id(tensor), []).append(name)
    for names in aliases.values():
        if len(names) < 2:
            continue
        first = state[names[0]]
        for name in names[1:]:
            other = state[name]
            if (
                not isinstance(first, torch.Tensor)
                or not isinstance(other, torch.Tensor)
                or first.shape != other.shape
                or first.dtype != other.dtype
                or not torch.equal(first, other)
            ):
                raise CheckpointIntegrityError(
                    f"checkpoint model aliases contain conflicting values: {', '.join(names)}"
                )


def _prune_generations(root: Path, retention: int) -> None:
    valid: list[Path] = []
    for generation_dir in _generation_directories(root):
        try:
            _validate_generation_files(generation_dir)
        except (CheckpointIntegrityError, CheckpointMismatchError):
            continue
        valid.append(generation_dir)
    keep = set(valid[:retention])
    for generation_dir in _generation_directories(root):
        if generation_dir not in keep:
            shutil.rmtree(generation_dir)
    _fsync_directory(root)


@contextmanager
def _checkpoint_writer_lock(root: Path, timeout: float) -> Iterator[None]:
    if fcntl is None:
        raise RuntimeError("checkpoint writers require POSIX advisory flock support")
    descriptor = os.open(root / WRITER_LOCK_FILENAME, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    acquired = False
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("checkpoint writer lock timed out") from None
                time.sleep(min(0.01, remaining))
        yield
    finally:
        try:
            if acquired:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    step: int,
    tokens_seen: int,
    losses: list[float],
    cursor: Mapping[str, Any] | Any | None = None,
    batch_ids: list[str] | tuple[str, ...] | None = None,
    sample_ids: list[list[str]] | tuple[tuple[str, ...], ...] | None = None,
    data_fingerprint: str = DEFAULT_DATA_FINGERPRINT,
    tokenizer_fingerprint: str = DEFAULT_TOKENIZER_FINGERPRINT,
    run_contract_fingerprint: str = DEFAULT_RUN_CONTRACT_FINGERPRINT,
    failure_injector: FailureInjector | None = None,
    retention: int = 2,
    rng_state: Mapping[str, Any] | None = None,
    writer_lock_timeout: float = 30.0,
) -> CheckpointManifestV2:
    """Atomically publish a trusted-local checkpoint generation.

    ``path`` is a checkpoint-store directory. For compatibility, a path ending
    in ``.pt`` is also treated as a directory; callers can pass the same value to
    :func:`load_checkpoint`. ``failure_injector`` receives named durability
    stages and exists solely for deterministic crash testing.
    POSIX writers serialize through a persistent per-store advisory lock;
    ``writer_lock_timeout`` bounds acquisition, not checkpoint serialization.
    """

    for label, value in (("step", step), ("tokens_seen", tokens_seen)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"checkpoint {label} must be a non-negative integer")
    if isinstance(retention, bool) or not isinstance(retention, int) or retention < 2:
        raise ValueError("checkpoint retention must keep at least two generations")
    try:
        valid_timeout = (
            isinstance(writer_lock_timeout, (int, float))
            and not isinstance(writer_lock_timeout, bool)
            and math.isfinite(writer_lock_timeout)
            and writer_lock_timeout > 0
        )
    except OverflowError:
        valid_timeout = False
    if not valid_timeout:
        raise ValueError("checkpoint writer lock timeout must be finite and positive")
    if fcntl is None:
        raise RuntimeError("checkpoint writers require POSIX advisory flock support")
    if any(
        not isinstance(value, str) or not value
        for value in (data_fingerprint, tokenizer_fingerprint, run_contract_fingerprint)
    ):
        raise ValueError("checkpoint input fingerprints must be non-empty strings")
    try:
        cursor_value = None if cursor is None else _json_compatible(cursor)
        if cursor_value is not None and not isinstance(cursor_value, dict):
            raise ValueError("checkpoint cursor must be an object or null")
        try:
            _validate_cursor(cursor_value)
        except CheckpointIntegrityError as error:
            raise ValueError(str(error)) from error
        json.dumps(cursor_value, allow_nan=False)
        batch_id_values = [str(value) for value in (batch_ids or [])]
        sample_id_values = [[str(sample_id) for sample_id in batch] for batch in (sample_ids or [])]
        loss_values = [float(loss) for loss in losses]
        json.dumps(loss_values, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"invalid checkpoint metadata: {error}") from error

    if rng_state is None:
        rng_state_value = None
    else:
        from fttl.cuda_runtime import validate_cpu_rng_state, validate_cuda_rng_state

        if not isinstance(rng_state, Mapping):
            raise ValueError("explicit checkpoint RNG state must be a mapping")
        if set(rng_state).difference({"python", "numpy", "torch_cpu", "torch_cuda"}):
            raise ValueError("explicit checkpoint RNG state contains unsupported fields")
        rng_state_value = copy.deepcopy(dict(rng_state))
        validate_cpu_rng_state(rng_state_value)
        if "torch_cuda" in rng_state_value:
            validate_cuda_rng_state(rng_state_value, torch.device("cuda:0"))

    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint(),
        "data_fingerprint": data_fingerprint,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "run_contract_fingerprint": run_contract_fingerprint,
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "step": step,
        "tokens_seen": tokens_seen,
        "losses": loss_values,
        "cursor": cursor_value,
        "batch_ids": batch_id_values,
        "sample_ids": sample_id_values,
        "rng_state": capture_rng_state() if rng_state_value is None else rng_state_value,
    }
    _validate_checkpoint_numerics(
        {"payload": payload, "runtime_buffers": dict(model.named_buffers(remove_duplicate=False))}
    )
    try:
        _validate_model_aliases(model, payload["model"])
    except CheckpointIntegrityError as error:
        raise ValueError(str(error)) from error

    root = Path(path)
    if root.exists() and not root.is_dir():
        raise CheckpointMismatchError(
            "checkpoint v2 requires a directory; move the legacy checkpoint first"
        )
    root.mkdir(parents=True, exist_ok=True)
    with _checkpoint_writer_lock(root, writer_lock_timeout):
        return _publish_checkpoint(
            root, payload, model=model, failure_injector=failure_injector, retention=retention
        )


def _assert_store_progress(root: Path, payload: dict[str, Any], model: torch.nn.Module) -> None:
    for generation_dir in _candidate_generation_directories(root):
        try:
            previous, _ = _load_generation_payload(
                generation_dir,
                model=model,
                expected_config_fingerprint=payload["config_fingerprint"],
                expected_data_fingerprint=payload["data_fingerprint"],
                expected_tokenizer_fingerprint=payload["tokenizer_fingerprint"],
                expected_run_contract_fingerprint=payload["run_contract_fingerprint"],
            )
        except CheckpointIntegrityError:
            continue
        if payload["step"] < previous["step"] or payload["tokens_seen"] < previous["tokens_seen"]:
            raise CheckpointMismatchError(
                "checkpoint progress must not decrease completed step or tokens seen"
            )
        return


def _publish_checkpoint(
    root: Path,
    payload: dict[str, Any],
    *,
    model: torch.nn.Module,
    failure_injector: FailureInjector | None,
    retention: int,
) -> CheckpointManifestV2:
    """Publish and prune while the caller holds this store's writer lock."""
    _cleanup_stale_temporaries(root)
    _cleanup_uncommitted_generations(root)
    _assert_store_contract(
        root,
        config_fingerprint=payload["config_fingerprint"],
        data_fingerprint=payload["data_fingerprint"],
        tokenizer_fingerprint=payload["tokenizer_fingerprint"],
        run_contract_fingerprint=payload["run_contract_fingerprint"],
    )
    _assert_store_progress(root, payload, model)
    generation = _next_generation(root)
    generation_name = f"generation-{generation:08d}"
    temporary_dir = root / f".{generation_name}.tmp-{uuid.uuid4().hex}"
    published_dir = root / generation_name
    temporary_dir.mkdir()

    state_path = temporary_dir / STATE_FILENAME
    with state_path.open("wb") as handle:
        torch.save(payload, handle)
        _inject(failure_injector, "after-state-serialize")
        handle.flush()
        _inject(failure_injector, "before-state-fsync")
        os.fsync(handle.fileno())
    _inject(failure_injector, "after-state-fsync")

    manifest = CheckpointManifestV2(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        generation=generation,
        state_file=STATE_FILENAME,
        state_sha256=_sha256(state_path),
        state_bytes=state_path.stat().st_size,
        completed_step=payload["step"],
        tokens_seen=payload["tokens_seen"],
        config_fingerprint=payload["config_fingerprint"],
        data_fingerprint=payload["data_fingerprint"],
        tokenizer_fingerprint=payload["tokenizer_fingerprint"],
        run_contract_fingerprint=payload["run_contract_fingerprint"],
        state_keys=tuple(sorted(_REQUIRED_STATE_KEYS)),
        cursor=payload["cursor"],
    )
    _write_durable_text(
        temporary_dir / MANIFEST_FILENAME,
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
    )
    _fsync_directory(temporary_dir)
    _inject(failure_injector, "after-manifest-fsync")
    _inject(failure_injector, "before-generation-publish")
    os.replace(temporary_dir, published_dir)
    _fsync_directory(root)
    _inject(failure_injector, "after-generation-publish")

    latest_temporary = root / f".{LATEST_FILENAME}.tmp-{uuid.uuid4().hex}"
    _write_durable_text(latest_temporary, generation_name + "\n")
    _inject(failure_injector, "before-latest-publish")
    os.replace(latest_temporary, root / LATEST_FILENAME)
    _fsync_directory(root)
    _inject(failure_injector, "after-latest-publish")
    _prune_generations(root, retention)
    return manifest


def _load_legacy_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    expected_config: ExperimentConfig,
    expected_data_fingerprint: str | None,
    expected_tokenizer_fingerprint: str | None,
    expected_run_contract_fingerprint: str | None,
) -> dict[str, Any]:
    if expected_data_fingerprint not in (None, DEFAULT_DATA_FINGERPRINT):
        raise CheckpointMismatchError("legacy checkpoint has no data fingerprint")
    if expected_tokenizer_fingerprint not in (None, DEFAULT_TOKENIZER_FINGERPRINT):
        raise CheckpointMismatchError("legacy checkpoint has no tokenizer fingerprint")
    if expected_run_contract_fingerprint is not None:
        raise CheckpointMismatchError("legacy checkpoint has no run contract fingerprint")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise CheckpointIntegrityError(
            "legacy checkpoint could not be decoded safely"
        ) from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise CheckpointMismatchError("unsupported checkpoint schema")
    if payload.get("config_fingerprint") != expected_config.fingerprint():
        raise CheckpointMismatchError(
            "checkpoint configuration does not match experiment"
        )
    _validate_stored_numerics(payload)
    if "model" in payload:
        _validate_model_aliases(model, payload["model"])
    return payload


def load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_config: ExperimentConfig,
    expected_data_fingerprint: str | None = None,
    expected_tokenizer_fingerprint: str | None = None,
    expected_run_contract_fingerprint: str | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load a verified checkpoint from a trusted local artifact store.

    V2 integrity, schema, size, digest, and fingerprint checks happen before
    deserialization. Corrupt newest generations fall back to the preceding
    valid committed generation. Contract mismatches are rejected rather than
    silently falling back. Numerically invalid payloads also fall back before
    receiver mutation, even when RNG restoration is disabled.
    The trusted-local v1 migration has no sidecar
    length or digest and is restricted to the synthetic compatibility path.
    """

    source = Path(path)
    if source.is_file():
        payload = _load_legacy_checkpoint(
            source,
            model=model,
            expected_config=expected_config,
            expected_data_fingerprint=expected_data_fingerprint,
            expected_tokenizer_fingerprint=expected_tokenizer_fingerprint,
            expected_run_contract_fingerprint=expected_run_contract_fingerprint,
        )
        payload["selected_generation"] = 0
    else:
        if (source / MANIFEST_FILENAME).is_file():
            candidates = [
                candidate
                for candidate in _candidate_generation_directories(source.parent)
                if candidate == source
            ]
        else:
            candidates = _candidate_generation_directories(source)
        if not candidates:
            raise CheckpointIntegrityError("checkpoint store has no committed generations")

        integrity_errors: list[CheckpointIntegrityError] = []
        payload = None
        for generation_dir in candidates:
            try:
                payload, selected_manifest = _load_generation_payload(
                    generation_dir,
                    model=model,
                    expected_config_fingerprint=expected_config.fingerprint(),
                    expected_data_fingerprint=expected_data_fingerprint,
                    expected_tokenizer_fingerprint=expected_tokenizer_fingerprint,
                    expected_run_contract_fingerprint=expected_run_contract_fingerprint,
                )
                payload["selected_generation"] = selected_manifest.generation
                break
            except CheckpointIntegrityError as error:
                integrity_errors.append(error)
        if payload is None:
            detail = str(integrity_errors[0]) if integrity_errors else "unknown error"
            raise CheckpointIntegrityError(
                f"no checkpoint generation passed integrity validation: {detail}"
            )

    if payload.get("config") != expected_config.to_dict():
        raise CheckpointMismatchError(
            "checkpoint serialized configuration does not match experiment"
        )
    required = {"model", "optimizer", "step", "tokens_seen", "losses"}
    missing = sorted(required.difference(payload))
    if missing:
        raise CheckpointIntegrityError(
            f"checkpoint state is missing keys: {', '.join(missing)}"
        )
    original_model = copy.deepcopy(model.state_dict())
    original_optimizer = copy.deepcopy(optimizer.state_dict())
    original_rng = capture_rng_state()

    def rollback() -> None:
        model.load_state_dict(original_model)
        optimizer.load_state_dict(original_optimizer)
        restore_rng_state(original_rng)

    try:
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
    except Exception as error:
        rollback()
        raise CheckpointMismatchError(
            "checkpoint model or optimizer state does not match experiment"
        ) from error
    if restore_rng:
        try:
            if "rng_state" in payload:
                restore_rng_state(payload["rng_state"])
            elif "torch_rng_state" in payload:
                torch.set_rng_state(payload["torch_rng_state"])
        except (KeyError, TypeError, RuntimeError, ValueError, OverflowError) as error:
            rollback()
            raise CheckpointIntegrityError("checkpoint RNG state is invalid") from error
    return payload
