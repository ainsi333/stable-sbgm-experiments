from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from levy_experiments.analysis_exp3 import verify_aggregate_manifest
from levy_experiments.config import load_config
from levy_experiments.errors import ArtifactError
from levy_experiments.storage import run_layout

ROOT = Path(__file__).resolve().parents[2]


def _run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_tiny_end_to_end_and_deterministic_resume(tmp_path: Path) -> None:
    common = [
        "--config",
        "tests/data/exp3_tiny.toml",
        "--device",
        "cpu",
        "--precision",
        "float64",
        "--output-dir",
        str(tmp_path),
    ]
    _run(["scripts/run_exp3.py", *common])
    _run(["scripts/aggregate_exp3.py", *common])
    run_directories = list(tmp_path.glob("exp3-*"))
    assert len(run_directories) == 1
    run_dir = run_directories[0]
    pdf = run_dir / "figures/experiment3.pdf"
    png = run_dir / "figures/experiment3.png"
    before = (
        hashlib.sha256(pdf.read_bytes()).hexdigest(),
        hashlib.sha256(png.read_bytes()).hexdigest(),
    )
    _run(["scripts/run_exp3.py", *common, "--resume"])
    _run(["scripts/aggregate_exp3.py", *common, "--resume"])
    after = (
        hashlib.sha256(pdf.read_bytes()).hexdigest(),
        hashlib.sha256(png.read_bytes()).hexdigest(),
    )
    assert before == after
    assert (run_dir / "aggregates/metrics_per_seed.csv").is_file()
    diagnostics_path = run_dir / "aggregates/diagnostics.json"
    assert diagnostics_path.is_file()
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["hybrid_control"]["excluded_from_valid_comparisons"] is True
    assert diagnostics["hybrid_control"]["label"] == (
        "invalid hybrid: drift eta=0.5, noise eta=1"
    )
    assert diagnostics["hybrid_control"]["simultaneous_rejection"]["pass"] is True
    assert diagnostics["publication_gate_pass"] is False
    assert diagnostics["publication_gate_failures"]
    assert (run_dir / "aggregates/refinement_per_seed.csv").is_file()
    assert (run_dir / "aggregates/aggregate_manifest.json").is_file()

    config = load_config(ROOT / "tests/data/exp3_tiny.toml")
    layout = run_layout(tmp_path, config)
    manifest = verify_aggregate_manifest(config, layout)
    assert manifest["task_count"] == 18
    original_diagnostics = diagnostics_path.read_bytes()
    diagnostics_path.write_bytes(original_diagnostics + b" ")
    with pytest.raises(ArtifactError, match="aggregate artifact checksum mismatch"):
        verify_aggregate_manifest(config, layout)
    diagnostics_path.write_bytes(original_diagnostics)

    first_arrays = next((run_dir / "raw").glob("*/arrays.npz"))
    original_arrays = first_arrays.read_bytes()
    first_arrays.write_bytes(original_arrays + b"corruption")
    with pytest.raises(ArtifactError, match="raw task checksum mismatch"):
        verify_aggregate_manifest(config, layout)
