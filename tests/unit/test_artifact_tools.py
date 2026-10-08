"""Archive checks fail before writes and detect corruption of code as well as data."""

import argparse
import hashlib
from pathlib import Path

import pytest

from scripts import replot_saved_results, verify_artifact


@pytest.fixture
def archived(tmp_path):
    payload = b"frozen source\n"
    (tmp_path / "source.py").write_bytes(payload)
    (tmp_path / "SHA256SUMS").write_text(
        hashlib.sha256(payload).hexdigest() + "  source.py\n", encoding="utf-8"
    )
    return tmp_path


def test_all_packaged_hashes_checked(archived):
    assert verify_artifact.verify_checksums(archived) == 1
    (archived / "source.py").write_bytes(b"modified")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_artifact.verify_checksums(archived)


def test_unlisted_file_rejected(archived):
    (archived / "secret.txt").write_text("not distributed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unlisted"):
        verify_artifact.verify_checksums(archived)


@pytest.mark.parametrize("name", ["../outside", "/absolute", "C:/outside", "a\\b"])
def test_checksum_path_escape_rejected(archived, name):
    (archived / "SHA256SUMS").write_text("0" * 64 + f"  {name}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unsafe"):
        verify_artifact.verify_checksums(archived)


def test_replot_preflights_manifest_before_any_output(tmp_path, monkeypatch):
    manifest = tmp_path / "figure_manifest.json"
    manifest.write_text("preserved", encoding="utf-8")
    monkeypatch.setattr(replot_saved_results, "parse_args", lambda: argparse.Namespace(
        experiment="all", output_dir=tmp_path, data_root=Path("absent"), force=False,
    ))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        replot_saved_results.main()
    assert list(tmp_path.iterdir()) == [manifest]
    assert manifest.read_text("utf-8") == "preserved"
