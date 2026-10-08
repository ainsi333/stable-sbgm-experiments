from __future__ import annotations

from pathlib import Path

from experiments.exp4.config import load_config


def test_smoke_config_is_hash_stable_and_source_independent() -> None:
    project = Path(__file__).resolve().parents[2]
    path = project / "configs" / "smoke" / "exp4.toml"
    first = load_config(path)
    second = load_config(path)

    assert first.resolved_hash == second.resolved_hash
    assert first.tier == "smoke"
    assert first.source.required is False
    assert first.precision == "float64"
    assert first.validation.domain_radii[-1] == first.stable_spectral.spatial_max
