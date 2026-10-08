from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.exp2.config import enumerate_tasks, load_config
from experiments.exp2.runner import ensure_score_tables, run_task
from experiments.exp2.storage import (
    compute_code_hash,
    load_task_artifact,
    prepare_run,
)
from levy_experiments.devices import configure_runtime


def test_exp2_cpu_tasks_are_finite_atomic_and_resumable(tmp_path: Path) -> None:
    config = load_config(Path("configs/smoke/exp2.toml"), seed_override=0)
    tasks = enumerate_tasks(config)
    selected = (
        next(task for task in tasks if task.model == "stable"),
        next(task for task in tasks if task.model == "vp"),
    )
    layout = prepare_run(tmp_path, config)
    tables = ensure_score_tables(layout, config, resume=True)
    device, runtime = configure_runtime("cpu", config.experiment.precision)

    for task in selected:
        completed = run_task(
            config,
            task,
            layout,
            device=device,
            runtime=runtime,
            table=tables[task.model],
            hires_table=tables[f"{task.model}_hires"],
            resume=False,
        )
        assert completed["status"] == "complete"
        resumed = run_task(
            config,
            task,
            layout,
            device=device,
            runtime=runtime,
            table=tables[task.model],
            hires_table=tables[f"{task.model}_hires"],
            resume=True,
        )
        assert resumed["status"] == "skipped"
        arrays, metadata = load_task_artifact(
            layout,
            task,
            expected_config_hash=config.resolved_hash,
            expected_code_hash=compute_code_hash(),
        )
        for name in (
            "numerical_exact_init",
            "numerical_reference_init",
            "exact_a",
            "exact_b",
            "exact_c",
        ):
            assert arrays[name].shape == (3, config.experiment.particles)
            assert np.all(np.isfinite(arrays[name]))
        assert arrays["forward_times"][-1] == config.experiment.epsilon
        assert arrays["score_sensitivity_wp"].shape == (3, 2)
        assert arrays["d_controlled"].shape == (3, 2)
        assert arrays["d_controlled"].dtype == np.dtype(bool)
        assert np.all(arrays["score_sensitivity_wp"] >= 0.0)
        assert np.array_equal(arrays["score_sensitivity_wp"][0], np.zeros(2))
        assert not np.array_equal(arrays["exact_a"], arrays["exact_b"])
        assert not np.array_equal(arrays["exact_b"], arrays["exact_c"])
        assert metadata["backend"]["platform"] == "cpu"
        assert metadata["score_table_hash"] != metadata["score_table_hires_hash"]
        assert metadata["score_table_sensitivity_coupling"] == (
            "same_exact_pT_initialization_and_same_primitive_noise"
        )
        assert metadata["throughput_particle_steps_per_second"] > 0.0
