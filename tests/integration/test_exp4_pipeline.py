from __future__ import annotations

import json
from pathlib import Path

from experiments.exp4.config import load_config
from experiments.exp4.run import run_experiment


def test_tiny_exp4_pipeline_and_resume(tmp_path: Path) -> None:
    config_path = tmp_path / "exp4.toml"
    config_path.write_text(
        """
[experiment]
name = "experiment4"
tier = "smoke"
target_df = 4.0
alpha = 1.5
beta = 1.0
time_min = 0.1
time_max = 1.0
time_points = 3
precision = "float64"

[source]
required = false
run_dir = ""
config_hash = ""
code_hash = ""
stable_main_file = ""
stable_main_hash = ""
stable_hires_file = ""
stable_hires_hash = ""
vp_main_file = ""
vp_main_hash = ""
vp_hires_file = ""
vp_hires_hash = ""

[stable_spectral]
fft_half_width = 256.0
fft_points = 32768
spatial_max = 20.0
density_floor = 1.0e-13

[vp_spectral]
fft_half_width = 64.0
fft_points = 16384
spatial_max = 20.0
density_floor = 1.0e-13

[validation]
times = [0.1, 1.0]
domain_radii = [8.0, 20.0]
refinement_factor = 2
resolution_rtol = 2.0e-2
vp_tail_reference_rtol = 2.0e-1
stable_boundary_fraction = 1.0e-1
source_table_rtol = 2.0e-1
maximizer_inner_radius = 8.0
""".strip()
        + "\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    output = tmp_path / "outputs"
    run_dir = run_experiment(
        config,
        output_root=output,
        resume=False,
        dry_run=False,
        device="cpu",
        precision="float64",
        seed=17,
    )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["all_numerical_gates_pass"] is True
    assert (run_dir / "figures" / "experiment4_ct.pdf").stat().st_size > 1000
    assert (run_dir / "figures" / "experiment4_ct.png").stat().st_size > 1000
    assert (run_dir / "aggregates" / "analysis_arrays.npz").is_file()

    resumed = run_experiment(
        config,
        output_root=output,
        resume=True,
        dry_run=False,
        device="cpu",
        precision="float64",
        seed=17,
    )
    assert resumed == run_dir
