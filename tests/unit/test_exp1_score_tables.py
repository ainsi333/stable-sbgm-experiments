from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.exp1.config import load_config
from experiments.exp1.score_tables import (
    build_score_table,
    ensure_score_table,
    interpolate_score_jax,
    interpolate_score_numpy,
    score_table_path,
    sensitivity_config,
)
from levy_experiments.errors import ArtifactError

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def smoke_tables():
    config = load_config(ROOT / "configs" / "smoke" / "exp1.toml")
    return config, {
        ("stable", 1.2): build_score_table(config, "stable", 1.2),
        ("vp", 1.5): build_score_table(config, "vp", 1.5),
    }


def test_tables_are_model_target_specific_and_pass_independent_validation(smoke_tables) -> None:
    _, tables = smoke_tables
    stable = tables[("stable", 1.2)]
    vp = tables[("vp", 1.5)]
    assert stable.table_hash != vp.table_hash
    for table in tables.values():
        validation = table.metadata["validation"]
        assert validation["independent_bulk_max_relative"] <= validation["bulk_rtol"]
        assert validation["embedded_fft2_blend_max_relative"] <= validation["blend_rtol"]
        assert table.scores.dtype == np.float64
        assert np.all(np.isfinite(table.scores))


def test_jax_numpy_parity_oddness_and_no_spatial_clipping(smoke_tables) -> None:
    _, tables = smoke_tables
    for table in tables.values():
        values = np.asarray([-10_000.0, -3.0, 0.0, 3.0, 10_000.0])
        time = 0.83
        expected = np.asarray(interpolate_score_numpy(values, time, table))

        def evaluate(x, current_time=time, current_table=table):
            return interpolate_score_jax(x, current_time, current_table, jnp.float64)

        observed = np.asarray(jax.jit(evaluate)(jnp.asarray(values)))
        assert np.all(np.isfinite(expected))
        assert np.allclose(observed, expected, rtol=2.0e-11, atol=2.0e-11)
        assert np.allclose(expected, -expected[::-1], rtol=2.0e-12, atol=2.0e-12)
        boundary = float(interpolate_score_numpy(table.space_nodes[-1], time, table))
        assert not np.isclose(expected[-1], boundary, rtol=1.0e-4, atol=1.0e-8)
        assert np.isnan(interpolate_score_numpy(1.0, 0.49, table))


def test_content_addressed_cache_and_exclusive_lock(tmp_path: Path, smoke_tables) -> None:
    config, tables = smoke_tables
    path = score_table_path(tmp_path, config, "vp", 1.5)
    built = ensure_score_table(tmp_path, config, "vp", 1.5, resume=True)
    assert built.table_hash == tables[("vp", 1.5)].table_hash
    resumed = ensure_score_table(tmp_path, config, "vp", 1.5, resume=True)
    assert resumed.table_hash == built.table_hash
    with pytest.raises(ArtifactError, match="resume is disabled"):
        ensure_score_table(tmp_path, config, "vp", 1.5, resume=False)
    assert path.is_file()


def test_sensitivity_table_resolution_is_independently_doubled() -> None:
    config = load_config(ROOT / "configs" / "smoke" / "exp1.toml")
    refined = sensitivity_config(config)
    assert refined.score_table.time_points == 2 * (config.score_table.time_points - 1) + 1
    assert refined.score_table.space_points == 2 * (config.score_table.space_points - 1) + 1
    assert refined.score_table.stable_fft_points == 2 * config.score_table.stable_fft_points
    assert refined.score_table.vp_fft_points == 2 * config.score_table.vp_fft_points
