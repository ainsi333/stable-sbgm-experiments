from __future__ import annotations

from pathlib import Path

import numpy as np

import levy_experiments.analysis_exp3 as analysis_exp3
from levy_experiments.config import ExperimentTask, load_config

ROOT = Path(__file__).resolve().parents[2]


def _fixture():
    config = load_config(ROOT / "tests/data/exp3_tiny.toml")
    rng = np.random.default_rng(731)
    times = np.asarray([2.5, 1.5, 0.5, 0.25], dtype=np.float64)
    reference_arrays = {
        "forward_times": times,
        "reference_a": rng.standard_normal((times.size, 128)),
        "reference_b": rng.standard_normal((times.size, 128)),
    }
    arrays = {
        "forward_times": times,
        "samples": rng.standard_normal((times.size, 128)),
    }
    tasks = tuple(
        ExperimentTask(index, "three_atoms", method, 2.75, eta, noise_eta, 11, 7)
        for index, (method, eta, noise_eta) in enumerate(
            (
                ("popov_ei", 0.25, 0.25),
                ("popov_ei", 0.5, 0.5),
                ("popov_ei", 1.0, 1.0),
                ("popov_ei", 2.0, 2.0),
                ("hybrid_ei_invalid", 0.5, 1.0),
            )
        )
    )
    marginal, joint_u, joint_v = analysis_exp3._frequency_grids(config)
    return config, arrays, reference_arrays, tasks, marginal, joint_u, joint_v


def test_reference_cache_preserves_every_metric_exactly() -> None:
    config, arrays, references, tasks, marginal, joint_u, joint_v = _fixture()
    uncached = [
        analysis_exp3._nonstationary_metrics(
            config, task, arrays, references, marginal, joint_u, joint_v
        )
        for task in tasks
    ]
    cache = analysis_exp3._exact_reference_metrics(config, references, joint_u, joint_v)
    cached = [
        analysis_exp3._nonstationary_metrics(
            config, task, arrays, references, marginal, joint_u, joint_v, cache
        )
        for task in tasks
    ]
    assert cached == uncached


def test_hybrid_rows_are_never_marked_as_valid_popov_members() -> None:
    *_, tasks, _, _, _ = _fixture()
    valid_row = analysis_exp3._row(
        tasks[0], section="audit", metric="identity", value=0.0
    )
    hybrid_row = analysis_exp3._row(
        tasks[-1], section="audit", metric="identity", value=0.0
    )

    assert valid_row["method"] == "popov_ei"
    assert valid_row["valid_method"] is True
    assert hybrid_row["method"] == "hybrid_ei_invalid"
    assert hybrid_row["valid_method"] is False


def test_reference_cache_reduces_joint_ecf_evaluations(monkeypatch) -> None:
    config, arrays, references, tasks, marginal, joint_u, joint_v = _fixture()
    original = analysis_exp3.empirical_cf_joint
    call_count = 0

    def counted(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(analysis_exp3, "empirical_cf_joint", counted)
    for task in tasks:
        analysis_exp3._nonstationary_metrics(
            config, task, arrays, references, marginal, joint_u, joint_v
        )
    uncached_calls = call_count

    call_count = 0
    cache = analysis_exp3._exact_reference_metrics(config, references, joint_u, joint_v)
    for task in tasks:
        analysis_exp3._nonstationary_metrics(
            config, task, arrays, references, marginal, joint_u, joint_v, cache
        )
    cached_calls = call_count

    pair_count = references["forward_times"].size - 1
    assert uncached_calls == 3 * pair_count * len(tasks)
    assert cached_calls == pair_count * (1 + 2 * len(tasks))
    assert uncached_calls - cached_calls == pair_count * (len(tasks) - 1)
