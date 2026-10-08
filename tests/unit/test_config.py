from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from levy_experiments.config import enumerate_tasks, load_config, validate_config
from levy_experiments.errors import ConfigurationError
from levy_experiments.experiment3 import checkpoint_step_indices, output_forward_times

ROOT = Path(__file__).resolve().parents[2]


def test_task_map_is_deterministic_and_complete() -> None:
    config = load_config(ROOT / "configs/smoke/exp3.toml")
    first = enumerate_tasks(config)
    second = enumerate_tasks(config)
    assert first == second
    assert [task.index for task in first] == list(range(len(first)))
    assert len({task.task_id for task in first}) == len(first) == 36
    assert sum(task.method == "exact_reference" for task in first) == 2
    assert sum(task.method == "hybrid_ei_invalid" for task in first) == 2


def test_checkpoint_orientation_and_exact_grid_alignment() -> None:
    config = load_config(ROOT / "configs/pilot/exp3.toml")
    assert output_forward_times(config) == (2.5, 1.5, 0.5, 0.25)
    assert checkpoint_step_indices(config, 1760) == (320, 960, 1600, 1760)
    assert checkpoint_step_indices(config, 3520) == (640, 1920, 3200, 3520)
    assert config.nonstationary.steps == (1760, 3520)


def test_publication_scale_rejects_lower_precision() -> None:
    config = load_config(ROOT / "configs/final/exp3.toml")
    invalid_experiment = dataclasses.replace(config.experiment, precision="mixed")
    with pytest.raises(ConfigurationError, match="publication-scale"):
        validate_config(dataclasses.replace(config, experiment=invalid_experiment))


def test_seed_zero_is_valid_but_zero_steps_are_not() -> None:
    config = load_config(ROOT / "configs/smoke/exp3.toml", seed_override=0)
    assert config.experiment.seeds == (0,)
    invalid_nonstationary = dataclasses.replace(config.nonstationary, steps=(0,))
    with pytest.raises(ConfigurationError, match="strictly positive"):
        validate_config(dataclasses.replace(config, nonstationary=invalid_nonstationary))


def test_prescribed_target_and_joint_probe_cannot_be_silently_mutated() -> None:
    config = load_config(ROOT / "configs/final/exp3.toml")
    wrong_target = dataclasses.replace(config.nonstationary, atoms=(-2.0, 0.0, 2.0))
    with pytest.raises(ConfigurationError, match="prescribed three-atom target"):
        validate_config(dataclasses.replace(config, nonstationary=wrong_target))

    null_probe = dataclasses.replace(config.analysis, joint_probe_u=0.0, joint_probe_v=0.0)
    with pytest.raises(ConfigurationError, match="joint probe"):
        validate_config(dataclasses.replace(config, analysis=null_probe))

    nondiscriminating = dataclasses.replace(
        config.experiment, etas=(0.99, 1.0, 1.01, 1.02)
    )
    with pytest.raises(ConfigurationError, match="insufficient analytic joint-CF separation"):
        validate_config(dataclasses.replace(config, experiment=nondiscriminating))
