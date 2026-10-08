from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from levy_experiments.errors import ArtifactError
from levy_experiments.pilot_gate import verify_pilot_gate

ROOT = Path(__file__).resolve().parents[2]


def _pilot_identity(experiment: int):
    path = ROOT / "configs" / "pilot" / f"exp{experiment}.toml"
    if experiment == 1:
        from experiments.exp1.config import (
            enumerate_tasks,
            load_config,
            pilot_compatibility_payload,
        )
        from experiments.exp1.storage import compute_code_hash
    elif experiment == 2:
        from experiments.exp2.config import (
            enumerate_tasks,
            load_config,
            pilot_compatibility_payload,
        )
        from experiments.exp2.storage import compute_code_hash
    else:
        from levy_experiments.config import (
            enumerate_tasks,
            load_config,
            pilot_compatibility_payload,
        )
        from levy_experiments.storage import compute_code_hash
    config = load_config(path, precision_override="float64")
    return (
        config,
        compute_code_hash(),
        tuple(enumerate_tasks(config)),
        pilot_compatibility_payload(config),
    )


def _write_hashed(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _write_pilot_artifacts(
    root: Path,
    experiment: int,
    *,
    overrides: dict[str, object] | None = None,
) -> Path:
    config, code_hash, tasks, compatibility = _pilot_identity(experiment)
    payload: dict[str, object] = {
        "config_hash": config.resolved_hash,
        "code_hash": code_hash,
        "task_count": len(tasks),
        "publication_gate_pass": True,
        "publication_gate_failures": [],
    }
    payload.update({} if overrides is None else overrides)
    run_dir = root / f"exp{experiment}-{config.resolved_hash[:12]}"
    diagnostic = run_dir / "aggregates" / "diagnostics.json"
    diagnostic_bytes = (json.dumps(payload, sort_keys=True) + "\n").encode()
    diagnostic_hash = _write_hashed(diagnostic, diagnostic_bytes)

    raw: list[tuple[object, str, str]] = []
    for task in tasks:
        directory = run_dir / "raw" / task.task_id
        arrays_hash = _write_hashed(
            directory / "arrays.npz", f"arrays:{task.task_id}".encode()
        )
        metadata_hash = _write_hashed(
            directory / "metadata.json", f"metadata:{task.task_id}".encode()
        )
        raw.append((task, arrays_hash, metadata_hash))

    common = {
        "status": "complete",
        "config_hash": config.resolved_hash,
        "code_hash": code_hash,
        "task_count": len(tasks),
    }
    if experiment == 1:
        manifest = common | {
            "tier": "pilot",
            "task_npz_hashes": {
                f"raw/{task.task_id}/arrays.npz": arrays_hash
                for task, arrays_hash, _ in raw
            },
            "task_metadata_hashes": {
                f"raw/{task.task_id}/metadata.json": metadata_hash
                for task, _, metadata_hash in raw
            },
            "files": {"aggregates/diagnostics.json": diagnostic_hash},
        }
    elif experiment == 2:
        manifest = {key: value for key, value in common.items() if key != "task_count"} | {
            "publication_gate_pass": payload["publication_gate_pass"],
            "publication_gate_failures": payload["publication_gate_failures"],
            "source": {
                "task_count": len(tasks),
                "tasks": [
                    {
                        "task_id": task.task_id,
                        "arrays_path": str(
                            (run_dir / "raw" / task.task_id / "arrays.npz").resolve()
                        ),
                        "arrays_sha256": arrays_hash,
                        "metadata_path": str(
                            (run_dir / "raw" / task.task_id / "metadata.json").resolve()
                        ),
                        "metadata_sha256": metadata_hash,
                    }
                    for task, arrays_hash, metadata_hash in raw
                ],
            },
            "artifacts": {
                "diagnostics.json": {
                    "path": str(diagnostic.resolve()),
                    "sha256": diagnostic_hash,
                    "size_bytes": diagnostic.stat().st_size,
                }
            },
        }
    else:
        from levy_experiments.config import pilot_compatibility_hash

        aggregate_hashes = {"diagnostics.json": diagnostic_hash}
        for name in (
            "metrics_per_seed.csv",
            "metrics_summary.csv",
            "refinement_per_seed.csv",
        ):
            aggregate_hashes[name] = _write_hashed(
                run_dir / "aggregates" / name, f"{name}\n".encode()
            )
        manifest = common | {
            "pilot_compatibility_payload": compatibility,
            "pilot_compatibility_hash": pilot_compatibility_hash(config),
            "publication_gate_pass": payload["publication_gate_pass"],
            "publication_gate_failures": payload["publication_gate_failures"],
            "aggregate_artifacts": aggregate_hashes,
            "task_artifacts": [
                {
                    "task_id": task.task_id,
                    "arrays_npz_sha256": arrays_hash,
                    "metadata_json_sha256": metadata_hash,
                }
                for task, arrays_hash, metadata_hash in raw
            ],
        }
    (run_dir / "aggregates" / "aggregate_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return run_dir


def test_exp2_nested_manifest_task_count_is_checked(tmp_path: Path) -> None:
    run_dir = _write_pilot_artifacts(tmp_path, 2)
    manifest_path = run_dir / "aggregates" / "aggregate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"]["task_count"] = -1
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ArtifactError, match="manifest identity mismatch"):
        verify_pilot_gate(
            experiment=2,
            final_config=ROOT / "configs" / "final" / "exp2.toml",
            output_root=tmp_path,
            precision="float64",
        )


@pytest.mark.parametrize("experiment", (1, 2, 3))
def test_matching_passed_pilot_authorizes_final(experiment: int, tmp_path: Path) -> None:
    pilot_root = tmp_path / "pilots"
    expected_run = _write_pilot_artifacts(pilot_root, experiment)
    receipt = verify_pilot_gate(
        experiment=experiment,
        final_config=ROOT / "configs" / "final" / f"exp{experiment}.toml",
        output_root=tmp_path / "finals",
        pilot_output_root=pilot_root,
        precision="float64",
    )
    assert receipt.run_dir == expected_run.resolve()
    assert receipt.task_count == len(_pilot_identity(experiment)[2])


def test_missing_pilot_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="requires a completed matching pilot"):
        verify_pilot_gate(
            experiment=1,
            final_config=ROOT / "configs" / "final" / "exp1.toml",
            output_root=tmp_path,
            precision="float64",
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"config_hash": "wrong"}, "configuration hash mismatch"),
        ({"code_hash": "wrong"}, "code hash mismatch"),
        ({"task_count": -1}, "task-count mismatch"),
        (
            {
                "publication_gate_pass": False,
                "publication_gate_failures": ["scientific gate failed"],
            },
            "scientific gate failed",
        ),
    ),
)
def test_incompatible_pilot_is_rejected(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    _write_pilot_artifacts(tmp_path, 2, overrides=overrides)
    with pytest.raises(ArtifactError, match=message):
        verify_pilot_gate(
            experiment=2,
            final_config=ROOT / "configs" / "final" / "exp2.toml",
            output_root=tmp_path,
            precision="float64",
        )


@pytest.mark.parametrize("experiment", (1, 2, 3))
def test_manifest_or_artifact_corruption_is_rejected(
    experiment: int, tmp_path: Path
) -> None:
    run_dir = _write_pilot_artifacts(tmp_path, experiment)
    (run_dir / "aggregates" / "diagnostics.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        verify_pilot_gate(
            experiment=experiment,
            final_config=ROOT / "configs" / "final" / f"exp{experiment}.toml",
            output_root=tmp_path,
            precision="float64",
        )


def test_missing_manifest_is_rejected(tmp_path: Path) -> None:
    run_dir = _write_pilot_artifacts(tmp_path, 3)
    (run_dir / "aggregates" / "aggregate_manifest.json").unlink()
    with pytest.raises(ArtifactError, match="aggregate manifest"):
        verify_pilot_gate(
            experiment=3,
            final_config=ROOT / "configs" / "final" / "exp3.toml",
            output_root=tmp_path,
            precision="float64",
        )


def _direct_cli_cases():
    from experiments.exp1 import aggregate as exp1_aggregate
    from experiments.exp1 import run as exp1_run
    from experiments.exp2 import aggregate as exp2_aggregate
    from experiments.exp2 import run as exp2_run
    from levy_experiments import cli_aggregate_exp3, cli_common, cli_exp3

    return (
        (exp1_run, exp1_run, 1, ["--task-index", "0"]),
        (exp1_aggregate, exp1_aggregate, 1, []),
        (exp2_run, exp2_run, 2, ["--task-index", "0"]),
        (exp2_aggregate, exp2_aggregate, 2, []),
        (cli_exp3, cli_common, 3, ["--task-index", "0"]),
        (cli_aggregate_exp3, cli_common, 3, []),
    )


@pytest.mark.parametrize(
    ("module", "gate_module", "experiment", "extra"), _direct_cli_cases()
)
def test_direct_final_dry_run_does_not_require_pilot(
    module,
    gate_module,
    experiment: int,
    extra: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_gate(**_kwargs):
        raise AssertionError("dry-run must not inspect pilot artifacts")

    monkeypatch.setattr(gate_module, "verify_pilot_gate", forbidden_gate)
    arguments = [
        "--config",
        str(ROOT / "configs" / "final" / f"exp{experiment}.toml"),
        "--device",
        "cpu",
        "--precision",
        "float64",
        "--output-dir",
        str(tmp_path / "finals"),
        "--allow-publication-scale",
        "--dry-run",
        *extra,
    ]
    assert module.main(arguments) == 0
    assert not (tmp_path / "finals").exists()


@pytest.mark.parametrize(
    ("module", "gate_module", "experiment", "extra"), _direct_cli_cases()
)
def test_direct_final_execution_fails_before_writing_without_pilot(
    module,
    gate_module,
    experiment: int,
    extra: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def reject_gate(**kwargs):
        calls.append(kwargs)
        raise ArtifactError("matching pilot deliberately absent")

    monkeypatch.setattr(gate_module, "verify_pilot_gate", reject_gate)
    pilot_root = tmp_path / "separate-pilots"
    arguments = [
        "--config",
        str(ROOT / "configs" / "final" / f"exp{experiment}.toml"),
        "--device",
        "cpu",
        "--precision",
        "float64",
        "--output-dir",
        str(tmp_path / "finals"),
        "--pilot-output-dir",
        str(pilot_root),
        "--allow-publication-scale",
        *extra,
    ]
    with pytest.raises(SystemExit) as raised:
        module.main(arguments)
    assert raised.value.code == 2
    assert len(calls) == 1
    assert calls[0]["pilot_output_root"] == pilot_root
    assert not (tmp_path / "finals").exists()


def test_cpu_wrapper_propagates_distinct_pilot_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import run_cpu_parallel

    commands: list[list[str]] = []
    pilot_root = tmp_path / "separate-pilots"

    def passed_gate(**_kwargs):
        return SimpleNamespace(run_dir=pilot_root / "accepted", code_hash="current")

    def fake_execute(command, **_kwargs):
        commands.append(command)
        return "1" if "--print-task-count" in command else ""

    monkeypatch.setattr(run_cpu_parallel, "verify_pilot_gate", passed_gate)
    monkeypatch.setattr(run_cpu_parallel, "_execute", fake_execute)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_cpu_parallel.py",
            "--experiment",
            "3",
            "--config",
            str(ROOT / "configs" / "final" / "exp3.toml"),
            "--output-dir",
            str(tmp_path / "finals"),
            "--pilot-output-dir",
            str(pilot_root),
            "--jobs",
            "1",
            "--allow-publication-scale",
        ],
    )
    assert run_cpu_parallel.main() == 0
    assert len(commands) == 4
    for command in commands:
        option = command.index("--pilot-output-dir")
        assert Path(command[option + 1]) == pilot_root.resolve()
