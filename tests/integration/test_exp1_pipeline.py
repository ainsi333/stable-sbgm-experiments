from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.exp1.config import enumerate_tasks, load_config
from experiments.exp1.runner import run_task
from experiments.exp1.score_tables import (
    ensure_score_table,
    ensure_sensitivity_score_table,
    table_key,
)
from experiments.exp1.storage import compute_code_hash, load_task_artifact, prepare_run
from levy_experiments.devices import configure_runtime


def test_exp1_cpu_dynamic_and_control_tasks_are_atomic_and_resumable(
    tmp_path: Path,
) -> None:
    config = load_config(Path("configs/smoke/exp1.toml"), seed_override=0)
    tasks = enumerate_tasks(config)
    dynamic = next(
        task
        for task in tasks
        if task.kind == "dynamic"
        and task.model == "stable"
        and task.nu == 1.2
        and task.steps == 4
    )
    control = next(task for task in tasks if task.kind == "control" and task.model == "pareto")
    layout = prepare_run(tmp_path, config)
    table = ensure_score_table(layout.run_dir, config, "stable", 1.2, resume=True)
    sensitivity = ensure_sensitivity_score_table(
        layout.run_dir, config, "stable", 1.2, resume=True
    )
    key = table_key("stable", 1.2)
    tables = {key: table, f"{key}:hires": sensitivity}
    device, runtime = configure_runtime("cpu", config.experiment.precision)

    for task in (dynamic, control):
        result = run_task(
            config,
            task,
            layout,
            device=device,
            runtime=runtime,
            tables=tables,
            resume=False,
        )
        assert result["status"] == "complete"
        resumed = run_task(
            config,
            task,
            layout,
            device=device,
            runtime=runtime,
            tables=tables,
            resume=True,
        )
        assert resumed["status"] == "skipped"
        arrays, metadata = load_task_artifact(
            layout,
            task,
            expected_config_hash=config.resolved_hash,
            expected_code_hash=compute_code_hash(),
        )
        assert arrays["terminal_sample"].shape == (task.particles,)
        assert np.all(np.isfinite(arrays["terminal_sample"]))
        assert metadata["backend"]["platform"] == "cpu"
        assert metadata["throughput_particle_steps_per_second"] > 0.0
        if task.kind == "dynamic":
            assert arrays["control_numerical"].shape == (
                3,
                task.control_particles,
            )
            assert arrays["control_exact_c"].shape == arrays["control_numerical"].shape
            assert arrays["score_hires_terminal"].shape == arrays["terminal_sample"].shape
            expected_work = (2 * task.particles + task.control_particles) * task.steps
            assert metadata["work_particle_steps"] == expected_work
            observed_work = (
                metadata["throughput_particle_steps_per_second"]
                * metadata["execution_seconds_excluding_compilation"]
            )
            assert observed_work == pytest.approx(expected_work)
