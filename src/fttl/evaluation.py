"""Bounded, document-local next-byte evaluation for the CPU transformer lab."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from fttl.dataset import PreparedDatasetSnapshot, load_dataset_snapshot
from fttl.model import TinyTransformer
from fttl.state import capture_rng_state, restore_rng_state, state_digest


def _complete_model_state_digest(model: TinyTransformer) -> str:
    state = {}
    for kind, values in (
        ("parameter", model.named_parameters(remove_duplicate=False)),
        ("buffer", model.named_buffers(remove_duplicate=False)),
    ):
        for name, tensor in values:
            label = f"evaluation model state.{kind}:{name}"
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.layout != torch.strided
                or tensor.device.type != "cpu"
                or tensor.is_quantized
            ):
                raise ValueError(f"{label} must be dense materialized CPU state")
            if kind == "parameter" and not tensor.is_floating_point():
                raise ValueError(f"{label} must have a floating-point dtype")
            try:
                finite = torch.isfinite(tensor).all().item()
            except RuntimeError as error:
                raise ValueError(f"{label} has an unsupported dtype") from error
            if not finite:
                raise ValueError(f"{label} contains non-finite values")
            state[f"{kind}:{name}"] = tensor
    return state_digest(state)


@dataclass(frozen=True)
class EvaluationResultV1:
    schema_version: int
    split: str
    data_fingerprint: str
    tokenizer_fingerprint: str
    model_state_digest: str
    available_documents: int
    evaluated_documents: int
    available_target_tokens: int
    evaluated_target_tokens: int
    windows: int
    complete_split: bool
    total_negative_log_likelihood: float
    mean_negative_log_likelihood: float
    perplexity: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_held_out(
    model: TinyTransformer,
    manifest_path: Path | PreparedDatasetSnapshot,
    *,
    split: str = "validation",
    max_tokens: int = 4096,
) -> EvaluationResultV1:
    """Evaluate each target once, including EOD, without wrapping or crossing documents.

    A partial final window is included. Context resets at each window, and the first
    byte of each document is context rather than a target. The token budget selects
    a deterministic prefix of document-ID order; it is not a random corpus sample.
    Complete model-state mutation is rejected, not rolled back. Arbitrary custom
    hook weight/gradient edits remain the caller's responsibility.
    """
    if split not in {"validation", "test"}:
        raise ValueError("held-out evaluation requires validation or test split")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    if model.config.vocab_size != 257:
        raise ValueError("byte evaluation requires model vocab_size=257")
    complete_before = _complete_model_state_digest(model)
    model_digest_before = state_digest(model.state_dict())
    modes = [(module, module.training) for module in model.modules()]
    flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    caller_rng = capture_rng_state()
    try:
        with torch.device("cpu"):
            snapshot = (
                manifest_path
                if isinstance(manifest_path, PreparedDatasetSnapshot)
                else load_dataset_snapshot(manifest_path)
            )
            manifest = snapshot.manifest
            documents = snapshot.documents_for(split)
            available_tokens = sum(len(document.text.encode("utf-8")) for document in documents)
            if not available_tokens:
                raise ValueError("selected held-out split has no target tokens")
            losses = []
            token_count = window_count = document_count = 0
            model.eval()
            with torch.inference_mode():
                for document in sorted(documents, key=lambda item: item.document_id):
                    tokens = tuple(document.text.encode("utf-8")) + (256,)
                    used_document = False
                    for start in range(0, len(tokens) - 1, model.config.block_size):
                        width = min(
                            model.config.block_size,
                            len(tokens) - 1 - start,
                            max_tokens - token_count,
                        )
                        if width == 0:
                            break
                        window = torch.tensor(
                            [tokens[start : start + width + 1]], dtype=torch.long, device="cpu"
                        )
                        logits, _ = model(window[:, :-1])
                        loss = F.cross_entropy(
                            logits.reshape(-1, 257), window[:, 1:].reshape(-1), reduction="sum"
                        )
                        value = float(loss)
                        if not math.isfinite(value):
                            raise ValueError("held-out evaluation produced non-finite loss")
                        losses.append(value)
                        token_count += width
                        window_count += 1
                        used_document = True
                    document_count += int(used_document)
                    if token_count == max_tokens:
                        break
            if _complete_model_state_digest(model) != complete_before:
                raise ValueError("evaluation model state changed during evaluation")
            total_nll = math.fsum(losses)
            mean_nll = total_nll / token_count
            try:
                perplexity = math.exp(mean_nll)
            except OverflowError:
                perplexity = None
            return EvaluationResultV1(
                schema_version=1,
                split=split,
                data_fingerprint=manifest.fingerprint(),
                tokenizer_fingerprint=manifest.tokenizer_fingerprint,
                model_state_digest=model_digest_before,
                available_documents=len(documents),
                evaluated_documents=document_count,
                available_target_tokens=available_tokens,
                evaluated_target_tokens=token_count,
                windows=window_count,
                complete_split=token_count == available_tokens,
                total_negative_log_likelihood=total_nll,
                mean_negative_log_likelihood=mean_nll,
                perplexity=perplexity,
            )
    finally:
        for module, training in modes:
            module.training = training
        for parameter, requires_grad in flags:
            parameter.requires_grad = requires_grad
        with torch.device("cpu"):
            restore_rng_state(caller_rng)
