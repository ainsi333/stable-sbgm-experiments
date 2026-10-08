from __future__ import annotations

import subprocess
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_wheel_contains_all_experiments_and_imports_away_from_checkout(tmp_path: Path) -> None:
    wheel_directory = tmp_path / "wheel"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_directory),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = tuple(wheel_directory.glob("levy_experiments-*.whl"))
    assert len(wheels) == 1
    wheel = wheels[0]

    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        assert "levy_experiments/__init__.py" in names
        assert "experiments/exp1/__init__.py" in names
        assert "experiments/exp2/__init__.py" in names
        entry_points_name = next(
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        )
        entry_points = archive.read(entry_points_name).decode("utf-8")
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
        assert metadata.get_all("License-Expression") == ["MIT"]
        assert metadata.get_all("License-File") == ["LICENSE"]
        dist_info_directory = metadata_name.removesuffix("METADATA")
        license_name = f"{dist_info_directory}licenses/{metadata['License-File']}"
        assert license_name in names
        assert archive.read(license_name) == (PROJECT_ROOT / "LICENSE").read_bytes()
    for command in (
        "levy-exp1",
        "levy-exp1-aggregate",
        "levy-exp2",
        "levy-exp2-aggregate",
        "levy-exp3",
        "levy-exp3-aggregate",
    ):
        assert f"{command} = " in entry_points

    empty_working_directory = tmp_path / "empty"
    empty_working_directory.mkdir()
    wheel_path = str(wheel.resolve())
    probe = (
        "import sys; "
        f"sys.path.insert(0, {wheel_path!r}); "
        "import experiments.exp1.run; "
        "import experiments.exp1.aggregate; "
        "import experiments.exp2.run; "
        "import experiments.exp2.aggregate; "
        "import levy_experiments.cli_exp3"
    )
    subprocess.run(
        [sys.executable, "-I", "-c", probe],
        cwd=empty_working_directory,
        check=True,
        capture_output=True,
        text=True,
    )
