from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from experiments.exp1.aggregate import _enforce_final_publication_gate
from experiments.exp1.config import (
    FINAL_RUN_GUARD,
    checkpoint_step_indices,
    enumerate_tasks,
    load_config,
    pilot_compatibility_payload,
    validate_config,
)
from experiments.exp1.storage import prepare_run, run_layout
from levy_experiments.errors import ArtifactError, ConfigurationError

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(("tier", "count"), (("smoke", 22), ("pilot", 60), ("final", 132)))
def test_frozen_task_maps_are_unique_and_have_expected_counts(tier: str, count: int) -> None:
    config = load_config(ROOT / "configs" / tier / "exp1.toml")
    tasks = enumerate_tasks(config)
    assert len(tasks) == count
    assert [task.index for task in tasks] == list(range(count))
    assert len({task.task_id for task in tasks}) == count
    dynamic = [task for task in tasks if task.kind == "dynamic"]
    assert {(task.model, task.nu) for task in dynamic} == {
        ("stable", 1.2),
        ("stable", 1.5),
        ("stable", 3.0),
        ("vp", 1.5),
    }
    for task in dynamic:
        assert task.coupling_steps % task.steps == 0
        assert checkpoint_step_indices(config, task)[0] == 0
        assert checkpoint_step_indices(config, task)[-1] == task.steps


def test_seed_namespaces_are_disjoint_and_override_is_strict() -> None:
    smoke = load_config(ROOT / "configs" / "smoke" / "exp1.toml")
    pilot = load_config(ROOT / "configs" / "pilot" / "exp1.toml")
    final = load_config(ROOT / "configs" / "final" / "exp1.toml")
    assert set(smoke.experiment.primary_seeds).isdisjoint(pilot.experiment.primary_seeds)
    assert set(pilot.experiment.primary_seeds).isdisjoint(final.experiment.primary_seeds)
    assert set(final.experiment.primary_seeds).isdisjoint(final.experiment.refinement_seeds)
    selected = load_config(ROOT / "configs" / "smoke" / "exp1.toml", seed_override=0)
    assert len(enumerate_tasks(selected)) == 11
    with pytest.raises(ConfigurationError, match="outside the frozen"):
        load_config(ROOT / "configs" / "smoke" / "exp1.toml", seed_override=1000)


def test_eta_and_moment_domain_cannot_be_silently_changed() -> None:
    config = load_config(ROOT / "configs" / "smoke" / "exp1.toml")
    with pytest.raises(ConfigurationError, match=r"alpha=1\.5, eta=0\.5"):
        validate_config(replace(config, stable=replace(config.stable, eta=1.0)))
    with pytest.raises(ConfigurationError, match="use W1 only"):
        validate_config(
            replace(config, analysis=replace(config.analysis, wasserstein_orders=(1.5,)))
        )


def test_final_write_guard_precedes_output_creation(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "final" / "exp1.toml")
    output = tmp_path / "outputs"
    with pytest.raises(ConfigurationError, match="allow_final=True"):
        prepare_run(output, config)
    assert not output.exists()
    authorized = load_config(ROOT / "configs" / "final" / "exp1.toml", allow_final=True)
    assert authorized.experiment.final_run_guard == FINAL_RUN_GUARD


def test_final_rendering_is_refused_when_publication_gate_fails(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "final" / "exp1.toml", allow_final=True)
    layout = run_layout(tmp_path, config)
    layout.aggregate_dir.mkdir(parents=True)
    with pytest.raises(ArtifactError, match="rendering refused"):
        _enforce_final_publication_gate(
            config,
            layout,
            {
                "publication_gate_pass": False,
                "publication_gate_failures": ["prespecified failure"],
            },
        )
    assert (layout.aggregate_dir / "publication_gate_failure.json").is_file()
    assert not any(layout.figure_dir.glob("*.pdf"))


def test_pilot_and_final_share_the_same_scientific_compatibility_payload() -> None:
    pilot = load_config(ROOT / "configs" / "pilot" / "exp1.toml")
    final = load_config(ROOT / "configs" / "final" / "exp1.toml", allow_final=True)
    assert pilot_compatibility_payload(pilot) == pilot_compatibility_payload(final)

    larger_table = replace(
        pilot,
        score_table=replace(
            pilot.score_table,
            time_points=2 * pilot.score_table.time_points - 1,
            space_points=2 * pilot.score_table.space_points - 1,
        ),
    )
    assert pilot_compatibility_payload(pilot) == pilot_compatibility_payload(larger_table)
    changed_eta = replace(pilot, stable=replace(pilot.stable, eta=0.6))
    assert pilot_compatibility_payload(pilot) != pilot_compatibility_payload(changed_eta)
