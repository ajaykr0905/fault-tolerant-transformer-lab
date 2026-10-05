from test_dataset_snapshot import alternate_fixture, snapshot_config, switch_directory
from test_real_data_recovery import prepared_fixture

from fttl import public_tuning
from fttl.dataset import PreparedDatasetSnapshot


def test_public_comparison_uses_one_snapshot_for_both_arms_and_every_evaluation(
    tmp_path, monkeypatch
):
    first = prepared_fixture(tmp_path / "first")
    second = alternate_fixture(tmp_path / "second")
    config = snapshot_config()
    expected = public_tuning.compare_public_tuning(
        config, first, tmp_path / "expected.json", rank=2, max_eval_tokens=13
    )
    live = tmp_path / "live"
    live.symlink_to(first.parent, target_is_directory=True)
    original_load = public_tuning.load_dataset_snapshot
    original_evaluate = public_tuning.evaluate_held_out
    captured = []
    evaluated = []

    def switched_after_capture(path):
        snapshot = original_load(path)
        captured.append(snapshot)
        switch_directory(live, second.parent)
        return snapshot

    def check_evaluation(model, snapshot, **kwargs):
        assert isinstance(snapshot, PreparedDatasetSnapshot)
        evaluated.append(snapshot)
        return original_evaluate(model, snapshot, **kwargs)

    monkeypatch.setattr(public_tuning, "load_dataset_snapshot", switched_after_capture)
    monkeypatch.setattr(public_tuning, "evaluate_held_out", check_evaluation)
    actual = public_tuning.compare_public_tuning(
        config, live / "manifest.json", tmp_path / "actual.json", rank=2, max_eval_tokens=13
    )
    assert len(captured) == 1
    assert len(evaluated) == 4
    assert all(snapshot is captured[0] for snapshot in evaluated)
    assert actual.data_fingerprint == expected.data_fingerprint
    assert actual.comparison_fingerprint == expected.comparison_fingerprint
    for mode in ("full", "lora"):
        observed = getattr(actual, mode)
        reference = getattr(expected, mode)
        assert observed.trace == reference.trace
        assert observed.validation_before == reference.validation_before
        assert observed.validation_after == reference.validation_after
