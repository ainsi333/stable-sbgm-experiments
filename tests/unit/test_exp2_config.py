from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from experiments.exp2.config import (
    checkpoint_step_indices,
    design_points,
    enumerate_tasks,
    load_config,
    pilot_compatibility_payload,
    validate_config,
    validate_execution,
)
from levy_experiments.errors import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def config_path(tier: str) -> Path:
    return PROJECT_ROOT / "configs" / tier / "exp2.toml"


@pytest.mark.parametrize(
    ("tier", "task_count", "seed_count", "checkpoint_count"),
    (("smoke", 12, 2, 3), ("pilot", 128, 8, 5), ("final", 192, 12, 5)),
)
def test_frozen_tiers_enumerate_unique_tasks(
    tier: str, task_count: int, seed_count: int, checkpoint_count: int
) -> None:
    config = load_config(config_path(tier))
    tasks = enumerate_tasks(config)

    assert len(tasks) == task_count
    assert len(config.experiment.seeds) == seed_count
    assert len(config.design.checkpoint_fractions) == checkpoint_count
    assert len({task.task_id for task in tasks}) == task_count
    assert len({(task.model, task.horizon, task.steps, task.seed) for task in tasks}) == task_count
    assert all(
        task.initialization_arms == ("exact_p_T", "stationary_reference")
        for task in tasks
    )
    assert all(
        task.reference_samples
        == ("exact_p_epsilon_a", "exact_p_epsilon_b", "exact_p_epsilon_c")
        for task in tasks
    )


def test_design_deduplicates_overlap_and_preserves_both_purposes() -> None:
    config = load_config(config_path("smoke"))

    assert design_points(config) == (
        (0.25, 4, ("horizon_sweep",)),
        (2.0, 8, ("horizon_sweep", "refinement")),
        (2.0, 16, ("refinement",)),
    )
    task = enumerate_tasks(config)[0]
    assert checkpoint_step_indices(config, task) == (0, 2, 4)


@pytest.mark.parametrize(
    ("tier", "stable_domain", "vp_domain"),
    (
        ("smoke", (131072, 2048.0, 192.0, 128.0, 192.0), (32768, 256.0, 32.0, 12.0, 16.0)),
        (
            "pilot",
            (262144, 4096.0, 256.0, 192.0, 256.0),
            (131072, 512.0, 64.0, 12.0, 16.0),
        ),
        (
            "final",
            (524288, 8192.0, 384.0, 256.0, 384.0),
            (262144, 1024.0, 96.0, 12.0, 16.0),
        ),
    ),
)
def test_score_table_domains_are_model_specific_and_frozen(
    tier: str,
    stable_domain: tuple[int, float, float, float, float],
    vp_domain: tuple[int, float, float, float, float],
) -> None:
    table = load_config(config_path(tier)).score_table

    assert (
        table.stable_fft_points,
        table.stable_tail_l,
        table.stable_x_max,
        table.stable_blend_start,
        table.stable_blend_end,
    ) == stable_domain
    assert (
        table.vp_fft_points,
        table.vp_tail_l,
        table.vp_x_max,
        table.vp_blend_start,
        table.vp_blend_end,
    ) == vp_domain


@pytest.mark.parametrize("tier", ("smoke", "pilot", "final"))
def test_hires_score_table_is_nested_and_changes_only_resolution(tier: str) -> None:
    config = load_config(config_path(tier))
    main = config.score_table
    hires = config.score_table_hires

    assert hires.time_points == 2 * main.time_points - 1
    assert hires.space_points == 2 * main.space_points - 1
    assert hires.stable_fft_points == 2 * main.stable_fft_points
    assert hires.vp_fft_points == 2 * main.vp_fft_points
    assert dataclasses.replace(
        hires,
        time_points=main.time_points,
        space_points=main.space_points,
        stable_fft_points=main.stable_fft_points,
        vp_fft_points=main.vp_fft_points,
    ) == main


def test_hires_score_table_cannot_reuse_main_resolution() -> None:
    config = load_config(config_path("smoke"))
    invalid = dataclasses.replace(config, score_table_hires=config.score_table)

    with pytest.raises(ConfigurationError, match=r"score_table_hires\.time_points"):
        validate_config(invalid)


def test_pilot_final_compatibility_keeps_science_and_allows_scale_to_change() -> None:
    pilot = load_config(config_path("pilot"))
    final = load_config(config_path("final"))

    assert pilot_compatibility_payload(pilot) == pilot_compatibility_payload(final)

    rescaled = dataclasses.replace(
        pilot,
        experiment=dataclasses.replace(
            pilot.experiment,
            tier="final",
            seeds=tuple(range(300, 320)),
            particles=32768,
            batch_size=8192,
            publication_scale=True,
            final_run_guard="different_resource_guard",
        ),
        design=dataclasses.replace(
            pilot.design,
            horizon_sweep=tuple(
                dataclasses.replace(setting, steps=setting.steps * 2)
                for setting in pilot.design.horizon_sweep
            ),
            refinement_steps=(100, 200, 400),
        ),
        score_table=final.score_table,
        score_table_hires=final.score_table_hires,
    )
    assert pilot_compatibility_payload(rescaled) == pilot_compatibility_payload(pilot)


def test_pilot_compatibility_payload_changes_for_scientific_mutations() -> None:
    config = load_config(config_path("pilot"))
    reference = pilot_compatibility_payload(config)
    mutations = (
        dataclasses.replace(
            config,
            stable=dataclasses.replace(config.stable, eta=0.75),
        ),
        dataclasses.replace(
            config,
            design=dataclasses.replace(
                config.design,
                checkpoint_fractions=(0.0, 0.5, 1.0),
            ),
        ),
        dataclasses.replace(
            config,
            design=dataclasses.replace(
                config.design,
                horizon_sweep=tuple(
                    dataclasses.replace(setting, horizon=setting.horizon + 0.5)
                    for setting in config.design.horizon_sweep
                ),
            ),
        ),
        dataclasses.replace(
            config,
            analysis=dataclasses.replace(
                config.analysis,
                wasserstein_orders=(1.0,),
            ),
        ),
    )

    assert all(pilot_compatibility_payload(value) != reference for value in mutations)


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"stable_fft_points": 1000}, "stable_fft_points"),
        ({"vp_fft_points": 1000}, "vp_fft_points"),
        ({"stable_x_max": 150.0}, "stable tail-blend"),
        ({"vp_x_max": 14.0}, "VP tail-blend"),
        ({"stable_tail_l": 192.0}, "stable_tail_l"),
        ({"vp_tail_l": 32.0}, "vp_tail_l"),
    ),
)
def test_each_model_score_domain_is_validated_independently(
    changes: dict[str, float | int], message: str
) -> None:
    config = load_config(config_path("smoke"))
    invalid = dataclasses.replace(
        config, score_table=dataclasses.replace(config.score_table, **changes)
    )

    with pytest.raises(ConfigurationError, match=message):
        validate_config(invalid)


def test_cli_overrides_are_audited_and_change_the_resolved_hash() -> None:
    base = load_config(config_path("smoke"))
    overridden = load_config(
        config_path("smoke"), precision_override="float32", seed_override=1
    )

    assert overridden.experiment.precision == "float32"
    assert overridden.experiment.seeds == (1,)
    assert overridden.overrides.precision == "float32"
    assert overridden.overrides.seed == 1
    assert len(enumerate_tasks(overridden)) == 6
    assert overridden.resolved_hash != base.resolved_hash

    with pytest.raises(ConfigurationError, match="frozen smoke protocol"):
        load_config(config_path("smoke"), seed_override=99)
    with pytest.raises(ConfigurationError, match="publication-scale float64"):
        load_config(config_path("final"), precision_override="float32")


def test_final_dry_run_loads_but_execution_requires_explicit_authorization() -> None:
    dry_run = load_config(config_path("final"))
    authorized = load_config(config_path("final"), allow_final=True)

    assert dry_run.resolved_hash == authorized.resolved_hash
    with pytest.raises(ConfigurationError, match="explicitly pass allow_final=True"):
        validate_execution(dry_run)
    validate_execution(authorized)


def test_checkpoint_divisibility_is_strict() -> None:
    config = load_config(config_path("smoke"))
    invalid_design = dataclasses.replace(
        config.design, checkpoint_fractions=(0.0, 0.3, 1.0)
    )
    invalid = dataclasses.replace(config, design=invalid_design)

    with pytest.raises(ConfigurationError, match="not a grid point"):
        validate_config(invalid)


def test_refinement_levels_must_nest_in_the_finest_grid() -> None:
    config = load_config(config_path("pilot"))
    invalid = dataclasses.replace(
        config,
        design=dataclasses.replace(config.design, refinement_steps=(156, 312, 625)),
    )

    with pytest.raises(ConfigurationError, match="divide the finest N"):
        validate_config(invalid)


def test_stable_wasserstein_order_must_be_strictly_below_alpha() -> None:
    config = load_config(config_path("smoke"))
    invalid_analysis = dataclasses.replace(
        config.analysis, wasserstein_orders=(1.0, config.stable.alpha)
    )
    invalid = dataclasses.replace(config, analysis=invalid_analysis)

    with pytest.raises(ConfigurationError, match="p < alpha"):
        validate_config(invalid)


def test_unknown_toml_keys_are_rejected(tmp_path: Path) -> None:
    source = config_path("smoke").read_text(encoding="utf-8")
    invalid = tmp_path / "invalid.toml"
    invalid.write_text(source + "\nunknown_root_key = 1\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Unknown key"):
        load_config(invalid)
