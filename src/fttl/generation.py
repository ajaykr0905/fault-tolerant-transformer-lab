"""Bounded token-level CPU inference; generated integers are not language-quality evidence."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import torch

from fttl.model import TinyTransformer
from fttl.numerical import require_finite_state
from fttl.state import capture_rng_state, restore_rng_state, state_digest

MAX_PROMPT_TOKENS = 4096
MAX_NEW_TOKENS = 256


@dataclass(frozen=True)
class GenerationResultV1:
    schema_version: int
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    generated_count: int
    stopping_reason: str
    max_new_tokens: int
    method: str
    temperature: float
    top_k: int | None
    seed: int
    stop_token_id: int | None
    model_state_digest: str
    model_config_fingerprint: str
    context_policy: str
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _integer(value: object, name: str, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")


def generate_tokens(
    model: TinyTransformer,
    prompt: Sequence[int] | bytes,
    *,
    max_new_tokens: int = 32,
    method: str = "greedy",
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int = 0,
    stop_token_id: int | None = None,
) -> GenerationResultV1:
    """Generate integer IDs with bounded context and private seeded CPU sampling.

    Greedy selection ignores temperature, top_k and seed, but validates them.
    Equal logits prefer the lower token ID, including a top-k cutoff tie. A stop
    token in the prompt does not stop generation; a newly emitted stop is included.
    Cropping resets learned position IDs to zero on every uncached forward pass.
    """
    if not isinstance(model, TinyTransformer):
        raise ValueError("generation requires a TinyTransformer")
    vocabulary = model.config.vocab_size
    _integer(max_new_tokens, "max_new_tokens", 0, MAX_NEW_TOKENS)
    _integer(seed, "seed", 0, 2**63 - 1)
    if not isinstance(method, str) or method not in {"greedy", "sample"}:
        raise ValueError("method must be greedy or sample")
    try:
        valid_temperature = (
            not isinstance(temperature, bool)
            and isinstance(temperature, (int, float))
            and math.isfinite(temperature)
            and temperature > 0
        )
    except OverflowError:
        valid_temperature = False
    if not valid_temperature:
        raise ValueError("temperature must be a finite positive number")
    if top_k is not None:
        _integer(top_k, "top_k", 1, vocabulary)
    if stop_token_id is not None:
        _integer(stop_token_id, "stop_token_id", 0, vocabulary - 1)
    if isinstance(prompt, str) or not isinstance(prompt, Sequence):
        raise ValueError("prompt must be a nonempty integer sequence or bytes")
    if not 1 <= len(prompt) <= MAX_PROMPT_TOKENS:
        raise ValueError(f"prompt must contain between 1 and {MAX_PROMPT_TOKENS} token IDs")
    prompt_ids = tuple(prompt)
    for token_id in prompt_ids:
        _integer(token_id, "prompt token", 0, vocabulary - 1)
    if any(value.device.type != "cpu" for value in (*model.parameters(), *model.buffers())):
        raise ValueError("generation supports CPU models only")
    # Non-persistent buffers are omitted from state_dict but still participate in
    # the runtime model contract and must not hide non-finite source values.
    source_state = {
        "parameters": dict(model.named_parameters()),
        "buffers": dict(model.named_buffers()),
    }
    if any(
        value.layout != torch.strided or value.is_nested
        for group in source_state.values()
        for value in group.values()
    ):
        raise ValueError(
            "generation source parameters and buffers must be materialized dense tensors"
        )
    require_finite_state(source_state, "model state")
    runtime_identity = state_digest(source_state)
    identity = state_digest(model.state_dict())
    configuration = json.dumps(asdict(model.config), sort_keys=True, separators=(",", ":"))
    modes = [(module, module.training) for module in model.modules()]
    caller_rng = capture_rng_state()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    all_ids = list(prompt_ids)
    generated: list[int] = []
    reason = "max_new_tokens"
    try:
        model.eval()
        with torch.inference_mode(), torch.device("cpu"):
            for _ in range(max_new_tokens):
                context = all_ids[-model.config.block_size :]
                tokens = torch.tensor([context], dtype=torch.long, device="cpu")
                logits, _ = model(tokens)
                if (
                    not isinstance(logits, torch.Tensor)
                    or logits.device.type != "cpu"
                    or logits.layout != torch.strided
                    or logits.is_nested
                    or not logits.is_floating_point()
                    or tuple(logits.shape) != (1, len(context), vocabulary)
                ):
                    raise ValueError(
                        "generation logits violate the CPU [1, context, vocabulary] contract"
                    )
                require_finite_state(logits, "generation logits")
                last = logits[0, -1]
                if method == "greedy":
                    selected = int(last.argmax())
                else:
                    # Stable descending order keeps the original token-ID tie order.
                    indices = torch.argsort(last, descending=True, stable=True)
                    if top_k is not None:
                        indices = indices[:top_k]
                    scores = last[indices].to(dtype=torch.float64)
                    scores = (scores - scores.max()) / temperature
                    probabilities = scores.softmax(dim=0)
                    require_finite_state(probabilities, "sampling probabilities")
                    index = torch.multinomial(probabilities, 1, generator=generator)
                    selected = int(indices[index].item())
                generated.append(selected)
                all_ids.append(selected)
                if selected == stop_token_id:
                    reason = "stop_token"
                    break
        current_source_state = {
            "parameters": dict(model.named_parameters()),
            "buffers": dict(model.named_buffers()),
        }
        if (
            state_digest(model.state_dict()) != identity
            or state_digest(current_source_state) != runtime_identity
        ):
            raise ValueError("model state changed during generation")
        return GenerationResultV1(
            schema_version=1,
            prompt_token_ids=prompt_ids,
            generated_token_ids=tuple(generated),
            token_ids=tuple(all_ids),
            generated_count=len(generated),
            stopping_reason=reason,
            max_new_tokens=max_new_tokens,
            method=method,
            temperature=float(temperature),
            top_k=top_k,
            seed=seed,
            stop_token_id=stop_token_id,
            model_state_digest=identity,
            model_config_fingerprint=hashlib.sha256(configuration.encode()).hexdigest(),
            context_policy="crop-to-block-size-reset-position-ids-no-cache-v1",
            limitations=(
                "Integer tokens, including random bytes, are not evidence of meaningful language.",
                "Sampling replay is scoped to this model state and CPU software environment.",
                "Small floating-point differences can change tokens; no cross-platform tolerance is guaranteed.",
                "Length caps bound requests, not model parameter size or peak memory.",
                "No KV cache, throughput benchmark, GPU, serving, or production claim is made.",
            ),
        )
    finally:
        for module, training in modes:
            module.training = training
        restore_rng_state(caller_rng)
