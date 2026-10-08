from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from experiments.exp1.config import load_config as load_exp1_config
from experiments.exp1.storage import prepare_run as prepare_exp1_run
from experiments.exp2.config import load_config as load_exp2_config
from experiments.exp2.storage import prepare_run as prepare_exp2_run
from levy_experiments.authority import authority_manifest
from levy_experiments.config import load_config as load_exp3_config
from levy_experiments.errors import ArtifactError
from levy_experiments.storage import prepare_run as prepare_exp3_run
from levy_experiments.storage import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT_ENV = "LEVY_MANUSCRIPT_PATH"


def test_frozen_authority_hashes_match_audited_sources() -> None:
    authority = authority_manifest()
    assert set(authority) == {
        "current_manuscript",
        "execution_manuscript_snapshot",
        "independent_specification",
    }
    assert (
        authority["current_manuscript"]["sha256"]
        == "4713d4441e62491322371b02c42d11673fc615b8c06d626560bfd36aac9ca88c"
    )
    manuscript_path = os.environ.get(MANUSCRIPT_ENV)
    if manuscript_path:
        manuscript = Path(manuscript_path)
        assert manuscript.is_file()
        assert authority["current_manuscript"]["sha256"] == sha256_file(manuscript)
    assert (
        authority["execution_manuscript_snapshot"]["sha256"]
        == "a2ceb81155566b43030677c90f5a62437146449118dacc17f0ec2b963dce56bf"
    )
    assert authority["independent_specification"]["sha256"] == sha256_file(
        PROJECT_ROOT / "reports" / "phase0_independent_spec_latest.md"
    )
    for source in authority.values():
        assert re.fullmatch(r"[0-9a-f]{64}", source["sha256"])


def test_authority_manifest_returns_an_independent_copy() -> None:
    first = authority_manifest()
    first["current_manuscript"]["sha256"] = "mutated"
    assert authority_manifest()["current_manuscript"]["sha256"] != "mutated"


@pytest.mark.parametrize(
    ("name", "config_path", "load", "prepare"),
    (
        ("exp1", PROJECT_ROOT / "configs/smoke/exp1.toml", load_exp1_config, prepare_exp1_run),
        ("exp2", PROJECT_ROOT / "configs/smoke/exp2.toml", load_exp2_config, prepare_exp2_run),
        ("exp3", PROJECT_ROOT / "tests/data/exp3_tiny.toml", load_exp3_config, prepare_exp3_run),
    ),
)
def test_every_run_provenance_embeds_authority(
    tmp_path: Path, name: str, config_path: Path, load, prepare
) -> None:
    config = load(config_path)
    layout = prepare(tmp_path / name, config)
    provenance = json.loads((layout.run_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["authority"] == authority_manifest()


def test_changed_authority_in_existing_provenance_is_rejected(tmp_path: Path) -> None:
    config = load_exp3_config(PROJECT_ROOT / "tests/data/exp3_tiny.toml")
    layout = prepare_exp3_run(tmp_path, config)
    path = layout.run_dir / "provenance.json"
    provenance = json.loads(path.read_text(encoding="utf-8"))
    provenance["authority"]["current_manuscript"]["sha256"] = "0" * 64
    path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ArtifactError, match="provenance differs"):
        prepare_exp3_run(tmp_path, config)
