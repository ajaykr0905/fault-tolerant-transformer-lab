"""Run the real process-kill verifier and export a small, offline evidence viewer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from html import escape
from pathlib import Path

from fttl.config import ExperimentConfig, ModelConfig
from fttl.dataset import DatasetSource, prepare_document_records
from fttl.process_recovery import verify_process_recovery
from fttl.train import FAILURE_POINTS

FIXTURE = Path("tests/fixtures/public_domain_peps_fixture.jsonl")
FIXTURE_SHA256 = "da4e4fd7ac86f2129864570c0f288378c2a5063ec3079ae07ff6b29c918cae67"


def _validate_preview(report: dict, control: dict, recovered: dict) -> None:
    """Check V1 claim consistency, not authenticity of attacker-controlled evidence."""
    required_checks = {
        "batch_id_sequence",
        "sample_id_sequence",
        "loss_sequence",
        "model_tensors",
        "optimizer_tensors",
        "rng_state",
        "cursor",
        "completed_steps",
        "token_count",
        "final_logits_tensors",
        "final_logits_digest",
        "final_checkpoint_state",
        "final_state_digest",
        "run_fingerprint",
        "run_contract",
        "staging_cleanup",
    }
    try:
        pids = report["worker_pids"]
        counts = (
            "completed_steps",
            "attempted_step",
            "selected_step",
            "durable_step_before_kill",
            "tokens_seen",
            "durable_committed_steps_lost",
            "durable_committed_tokens_lost",
        )
        if (
            report["schema"] != "ProcessRecoveryReportV1"
            or report["kill_signal"] != "SIGKILL"
            or report["interrupted_exitcode"] != -9
            or report["process_start_method"] != "spawn"
            or report["failure_point"] not in FAILURE_POINTS
            or set(pids) != {"control", "interrupted", "replay", "completion"}
            or any(type(pid) is not int or pid < 1 for pid in pids.values())
            or len(set(pids.values())) != 4
            or set(report["equality"]) != required_checks
            or report["staging_cleanup_complete"] is not True
            or any(type(report[name]) is not int for name in counts)
            or not 1
            <= report["selected_step"]
            < report["attempted_step"]
            <= report["completed_steps"]
            or report["selected_step"] != report["durable_step_before_kill"]
            or report["attempted_step"] != report["selected_step"] + 1
            or report["durable_committed_steps_lost"] != 0
            or report["durable_committed_tokens_lost"] != 0
            or any(
                type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0
                for seconds in report["timings_seconds"].values()
            )
        ):
            raise ValueError("refusing to display a diverged run as verified")
        for field in (
            "losses",
            "batch_ids",
            "sample_ids",
            "final_cursor",
            "model_digest",
            "optimizer_digest",
            "rng_digest",
            "final_logits_digest",
            "final_state_digest",
            "config_fingerprint",
            "data_fingerprint",
            "tokenizer_fingerprint",
            "run_contract_fingerprint",
            "run_fingerprint",
            "code_fingerprint",
            "steps",
            "tokens_seen",
        ):
            if control[field] != recovered[field]:
                raise ValueError("refusing to display a diverged run as verified")
        for trace in (control, recovered):
            if (
                type(trace["steps"]) is not int
                or type(trace["tokens_seen"]) is not int
                or trace["steps"] != report["completed_steps"]
                or trace["tokens_seen"] != report["tokens_seen"]
                or len(trace["batch_ids"]) != trace["steps"]
                or len(trace["sample_ids"]) != trace["steps"]
                or report["failed_batch_id"] != report["replayed_batch_id"]
                or report["failed_batch_id"] != trace["batch_ids"][report["attempted_step"] - 1]
                or report["failed_sample_ids"] != report["replayed_sample_ids"]
                or report["failed_sample_ids"] != trace["sample_ids"][report["attempted_step"] - 1]
            ):
                raise ValueError("refusing to display a diverged run as verified")
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError) as error:
        raise ValueError("refusing to display a diverged run as verified") from error


def export_preview(run: Path, destination: Path) -> None:
    """Render measured results, never run a simulated crash or relabel a failed proof."""
    report = json.loads((run / "process-recovery-report.json").read_text())
    control = json.loads((run / "control/result.json").read_text())
    recovered = json.loads((run / "recovered/result.json").read_text())
    _validate_preview(report, control, recovered)
    losses = control["losses"]
    if (
        report["exact_equality"] is not True
        or not report["equality"]
        or any(value is not True for value in report["equality"].values())
        or control["losses"] != recovered["losses"]
        or control["final_state_digest"] != recovered["final_state_digest"]
        or report["final_state_digest"] != recovered["final_state_digest"]
        or len(losses) != report["completed_steps"]
        or len(losses) < 2
        or any(type(value) not in (int, float) or not math.isfinite(value) for value in losses)
    ):
        raise ValueError("refusing to display a diverged run as verified")
    destination.mkdir(parents=True, exist_ok=False)
    for relative in (
        "process-recovery-report.json",
        "control/result.json",
        "recovered/result.json",
    ):
        target = destination / relative
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(run / relative, target)

    checks = len(report["equality"])
    replay_seconds = report["timings_seconds"]["resume_spawn_to_replayed_commit_receipt"]
    stages = (
        ("01 TRAIN", f"Durable step {report['durable_step_before_kill']}", "#60a5fa"),
        ("02 KILL", f"SIGKILL at step {report['attempted_step']}", "#fb7185"),
        ("03 RESTORE", f"Replay step {report['selected_step'] + 1}", "#fbbf24"),
        ("04 VERIFY", f"{checks}/{checks} checks pass", "#34d399"),
    )
    cards = ""
    for index, (title, detail, color) in enumerate(stages):
        x = 40 + index * 268
        cards += (
            f'<rect x="{x}" y="160" width="248" height="100" rx="12" fill="#142239"/>'
            f'<text x="{x + 18}" y="195" fill="{color}" font-size="16">{title}</text>'
            f'<text x="{x + 18}" y="231" fill="#e2e8f0" font-size="18">{detail}</text>'
        )
    low, high = min(losses), max(losses)
    span = max(high - low, 0.001)
    points = " ".join(
        f"{70 + index * 970 / max(len(losses) - 1, 1):.2f},{425 - (loss - low) / span * 90:.2f}"
        for index, loss in enumerate(losses)
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 670"
role="img" aria-labelledby="title desc">
<title id="title">Train. Kill. Recover. Verify.</title>
<desc id="desc">Measured CPU recovery: {checks} equality checks pass after a real SIGKILL.
Both loss sequences overlap exactly. This is a local reliability proof, not model quality.</desc>
<rect width="1120" height="670" rx="20" fill="#0b1220"/>
<g font-family="system-ui, sans-serif">
<text x="40" y="45" fill="#34d399" font-size="14">FAULT-TOLERANT TRANSFORMER LAB / WORKING PROTOTYPE</text>
<text x="40" y="99" fill="#f8fafc" font-size="38" font-weight="700">A crash should not change the result.</text>
<text x="40" y="132" fill="#94a3b8" font-size="17">Real worker death. Durable checkpoint. Exact replay.</text>
{cards}
<text x="40" y="302" fill="#e2e8f0" font-size="18">Per-step cross-entropy loss / zoomed axis</text>
<text x="8" y="340" fill="#94a3b8" font-size="12">{high:.3f}</text>
<text x="8" y="429" fill="#94a3b8" font-size="12">{low:.3f}</text>
<line x1="70" y1="440" x2="1040" y2="440" stroke="#334155"/>
<polyline points="{points}" fill="none" stroke="#60a5fa" stroke-width="6"/>
<polyline points="{points}" fill="none" stroke="#34d399" stroke-width="3" stroke-dasharray="8 6"/>
<text x="70" y="470" fill="#94a3b8" font-size="14">Step 1</text>
<text x="890" y="470" fill="#94a3b8" font-size="14">Step {report["completed_steps"]} / exact overlap</text>
<text x="40" y="517" fill="#60a5fa" font-size="15">Blue: uninterrupted control</text>
<text x="370" y="517" fill="#34d399" font-size="15">Green dashed: recovered run</text>
<text x="40" y="561" fill="#f8fafc" font-size="23">{report["durable_committed_steps_lost"]} committed steps lost</text>
<text x="370" y="561" fill="#f8fafc" font-size="23">{replay_seconds:.3f}s spawn-to-replay commit</text>
<text x="40" y="600" fill="#94a3b8" font-size="14">CPU / independent CC0 fixture / one paused worker killed during a declared boundary</text>
<text x="40" y="626" fill="#94a3b8" font-size="14">Timing includes startup, restore, replay, save and IPC. Not a production recovery SLA.</text>
</g></svg>'''
    (destination / "preview.svg").write_text(svg, encoding="utf-8")
    check_rows = "".join(
        f"<li><strong>PASS</strong> {escape(name.replace('_', ' '))}</li>"
        for name in sorted(report["equality"])
    )
    limitations = "".join(f"<li>{escape(item)}</li>" for item in report["limitations"])
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fault-Tolerant Transformer Lab — measured demo</title>
<style>
body{{margin:0;background:#0b1220;color:#e2e8f0;font:16px/1.6 system-ui,sans-serif}}
main{{max-width:1120px;margin:30px auto;padding:0 20px}}img{{width:100%;height:auto}}
a{{color:#93c5fd}}strong{{color:#6ee7b7}}code{{overflow-wrap:anywhere}}
section{{padding:20px;border:1px solid #334155;border-radius:12px;margin:20px 0}}
li{{margin:8px 0}}summary{{cursor:pointer}}nav{{display:flex;gap:24px;flex-wrap:wrap}}
</style><main><img src="preview.svg" alt="Train, kill, restore and verify: measured recovery proof">
<section><h1>A real run, not a simulation</h1>
<p>This page displays a completed experiment. It does not start training in your browser.
A parent sent SIGKILL to one independent CPU worker at
<b>{escape(report["failure_point"])}</b>. Fresh processes restored the checkpoint,
replayed the interrupted batch and matched an independent control.</p>
<p>Verified: {escape(report["verified_at_utc"])} · exit code:
{report["interrupted_exitcode"]} · {report["completed_steps"]} completed steps ·
{report["tokens_seen"]} committed tokens.</p>
<nav><a href="process-recovery-report.json">Raw recovery report</a>
<a href="control/result.json">Control trace</a>
<a href="recovered/result.json">Recovered trace</a></nav></section>
<section><details open><summary>Inspect all {checks} equality checks</summary>
<ul>{check_rows}</ul></details>
<p>Final state digest: <code>{escape(report["final_state_digest"])}</code></p>
<p>Code revision: <code>{escape(report["code_revision"])}</code></p></section>
<section><h2>Reproduce from the repository root</h2>
<pre>uv sync --frozen --extra test
uv run --frozen fttl-demo --output artifacts/my-demo-001</pre>
<p>Open <code>artifacts/my-demo-001/preview/index.html</code>.
Use a new output directory for each run. Linux/macOS, Python 3.12; CPU only.
The six-document CC0 fixture is independently written, not actual PEP text.</p>
<h2>What this does not prove</h2><ul>{limitations}</ul></section></main></html>"""
    (destination / "index.html").write_text(page, encoding="utf-8")


def run_demo(
    output: Path,
    *,
    fixture: Path = FIXTURE,
    failure_point: str = "during-checkpoint-write",
    preview_dir: Path | None = None,
) -> Path:
    """Run a bounded offline proof using the repository's six-document CC0 fixture."""
    if os.name != "posix":
        raise ValueError("the demo requires Linux/macOS with POSIX SIGKILL")
    if failure_point not in FAILURE_POINTS:
        raise ValueError("unknown failure point")
    if output.exists() or (preview_dir is not None and preview_dir.exists()):
        raise ValueError(
            "choose fresh output and preview directories; existing evidence is preserved"
        )
    if preview_dir is not None and (
        preview_dir.resolve().is_relative_to(output.resolve())
        or output.resolve().is_relative_to(preview_dir.resolve())
    ):
        raise ValueError("custom preview and output directories must not overlap")
    source_bytes = fixture.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != FIXTURE_SHA256:
        raise ValueError("fixture SHA-256 mismatch; use the pinned independent CC0 fixture")
    records = [json.loads(line) for line in source_bytes.decode("utf-8").splitlines() if line]
    source = DatasetSource(
        repository="fttl://tests/public-domain-pep-fixture",
        revision="fixture-v1",
        path=fixture.name,
        compressed_sha256=hashlib.sha256(source_bytes).hexdigest(),
        expected_document_count=6,
        license="CC0-1.0",
        license_limitations=("Independently written demo fixture, not actual PEP documents.",),
        artifact_kind="jsonl",
    )
    prepare_document_records(records, output / "dataset", source_artifact=fixture, source=source)
    config = ExperimentConfig(
        model=ModelConfig(
            vocab_size=257, block_size=8, d_model=8, n_heads=2, n_layers=1, dropout=0.25
        ),
        seed=23,
        steps=6,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=1,
    )
    (output / "config.json").write_text(config.canonical_json() + "\n", encoding="utf-8")
    print("Training a control, killing a worker, restoring and comparing...", flush=True)
    report = verify_process_recovery(
        config,
        output / "run",
        failure_point=failure_point,
        dataset_manifest=output / "dataset/manifest.json",
        timeout_seconds=60,
    )
    destination = preview_dir if preview_dir is not None else output / "preview"
    export_preview(output / "run", destination)
    shutil.copyfile(output / "config.json", destination / "config.json")
    shutil.copyfile(output / "dataset/manifest.json", destination / "dataset-manifest.json")
    print(f"PASS: {len(report.equality)}/{len(report.equality)} exact equality checks")
    print(
        f"Real {report.kill_signal}, exit {report.interrupted_exitcode}; "
        f"restored step {report.selected_step}, replayed step 2, finished {report.completed_steps}"
    )
    print(f"Open {destination / 'index.html'}")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Fresh local evidence directory")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=FIXTURE,
        help="Independent CC0 fixture; run from the repository root",
    )
    parser.add_argument(
        "--failure-point", choices=FAILURE_POINTS, default="during-checkpoint-write"
    )
    parser.add_argument(
        "--preview-dir", type=Path, help="Optional fresh directory for a public preview"
    )
    args = parser.parse_args(argv)
    try:
        run_demo(
            args.output,
            fixture=args.fixture,
            failure_point=args.failure_point,
            preview_dir=args.preview_dir,
        )
    except (ValueError, OSError) as error:
        parser.exit(1, f"Demo failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
