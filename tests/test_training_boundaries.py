import pytest

from fttl.config import ExperimentConfig, ModelConfig
from fttl.state import capture_rng_state, state_digest
from fttl.train import run_training


def config():
    return ExperimentConfig(
        model=ModelConfig(vocab_size=16, block_size=4, d_model=8, n_heads=2, n_layers=1),
        steps=3,
        batch_size=2,
        checkpoint_every=1,
    )


@pytest.mark.parametrize("field", ["stop_after_step", "failure_step"])
@pytest.mark.parametrize("value", [True, False, 1.0, 1.5, float("nan"), float("inf"), "1", 0, -1])
def test_invalid_step_controls_fail_before_rng_or_output_changes(field, value, tmp_path):
    output = tmp_path / "run"
    before = state_digest(capture_rng_state())
    kwargs = {field: value}
    if field == "failure_step":
        kwargs["failure_point"] = "before-forward"
    with pytest.raises(ValueError, match=field):
        run_training(config(), output, **kwargs)
    assert not output.exists()
    assert state_digest(capture_rng_state()) == before


def test_invalid_failure_range_does_not_create_output(tmp_path):
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="failure_step"):
        run_training(config(), output, failure_point="before-forward", failure_step=4)
    assert not output.exists()


def test_integer_stop_boundary_still_produces_a_resumable_checkpoint(tmp_path):
    _, first = run_training(config(), tmp_path / "first", stop_after_step=1)
    _, resumed = run_training(
        config(), tmp_path / "resumed", resume_from=tmp_path / "first/checkpoints"
    )
    _, control = run_training(config(), tmp_path / "control")
    assert first.steps == 1
    assert resumed.steps == control.steps == 3
    assert resumed.final_state_digest == control.final_state_digest
