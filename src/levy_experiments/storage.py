"""Atomic, hash-checked run storage for independent Experiment 3 tasks."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .authority import authority_manifest
from .config import Experiment3Config, ExperimentTask, enumerate_tasks
from .errors import ArtifactError


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_content_hash(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(json.dumps(values.shape).encode("ascii"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def compute_code_hash(root: Path | None = None) -> str:
    """Hash executable source independently of uncommitted Git state."""

    root = project_root() if root is None else root.resolve()
    candidates: list[Path] = []
    for relative in ("src", "scripts"):
        directory = root / relative
        if directory.exists():
            candidates.extend(path for path in directory.rglob("*.py") if path.is_file())
    candidates.append(root / "pyproject.toml")
    digest = hashlib.sha256()
    for path in sorted(candidates, key=lambda item: item.relative_to(root).as_posix()):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_revision(root: Path | None = None) -> dict[str, Any]:
    root = project_root() if root is None else root.resolve()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def dependency_versions() -> dict[str, str | None]:
    packages = ("jax", "jaxlib", "numpy", "scipy", "matplotlib", "levy-experiments")
    result: dict[str, str | None] = {}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


@dataclass(frozen=True)
class RunLayout:
    root: Path
    run_dir: Path
    raw_dir: Path
    failed_dir: Path
    aggregate_dir: Path
    figure_dir: Path
    log_dir: Path


def run_layout(output_root: str | Path, config: Experiment3Config) -> RunLayout:
    root = Path(output_root).resolve()
    run_dir = root / f"exp3-{config.resolved_hash[:12]}"
    return RunLayout(
        root=root,
        run_dir=run_dir,
        raw_dir=run_dir / "raw",
        failed_dir=run_dir / "failures",
        aggregate_dir=run_dir / "aggregates",
        figure_dir=run_dir / "figures",
        log_dir=run_dir / "logs",
    )


def _atomic_write_bytes(path: Path, payload: bytes, *, refuse_existing: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and path.exists():
        raise ArtifactError(f"Refusing to overwrite existing artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if refuse_existing and path.exists():
            raise ArtifactError(f"Concurrent writer created artifact: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_or_verify_bytes(path: Path, payload: bytes, *, resume: bool) -> str:
    """Write once, or verify byte identity when resuming."""

    if path.exists():
        if not resume:
            raise ArtifactError(f"Refusing to overwrite existing artifact: {path}")
        if path.read_bytes() != payload:
            raise ArtifactError(f"Recomputed artifact differs from existing output: {path}")
        return "verified"
    _atomic_write_bytes(path, payload)
    return "written"


def atomic_write_json(path: Path, payload: dict[str, Any], *, refuse_existing: bool = True) -> None:
    serialized = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    _atomic_write_bytes(path, serialized, refuse_existing=refuse_existing)


def _task_manifest_bytes(tasks: tuple[ExperimentTask, ...]) -> bytes:
    buffer = io.StringIO(newline="")
    fieldnames = [
        "index",
        "task_id",
        "target",
        "method",
        "horizon",
        "eta",
        "noise_eta",
        "steps",
        "seed",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for task in tasks:
        row = asdict(task)
        row["task_id"] = task.task_id
        writer.writerow({name: row[name] for name in fieldnames})
    return buffer.getvalue().encode("utf-8")


def prepare_run(output_root: str | Path, config: Experiment3Config) -> RunLayout:
    """Create or validate immutable run-level provenance files."""

    layout = run_layout(output_root, config)
    for directory in (
        layout.run_dir,
        layout.raw_dir,
        layout.failed_dir,
        layout.aggregate_dir,
        layout.figure_dir,
        layout.log_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    resolved_path = layout.run_dir / "resolved_config.json"
    resolved_payload = json.loads(
        json.dumps(
            config.canonical_dict(include_source_path=False)
            | {
                "resolved_hash": config.resolved_hash,
                "requested_config_source_hash": config.source_hash,
            },
            sort_keys=True,
        )
    )
    if resolved_path.exists():
        try:
            existing = json.loads(resolved_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"Corrupt resolved configuration: {resolved_path}") from exc
        if existing != resolved_payload:
            raise ArtifactError("Output directory contains an incompatible resolved configuration")
    else:
        atomic_write_json(resolved_path, resolved_payload)
    requested_path = layout.run_dir / "requested_config.toml"
    requested_bytes = Path(config.source_path).read_bytes()
    if requested_path.exists():
        if requested_path.read_bytes() != requested_bytes:
            raise ArtifactError(
                "requested_config.toml differs from the current configuration source"
            )
    else:
        _atomic_write_bytes(requested_path, requested_bytes)
    manifest_path = layout.run_dir / "tasks.csv"
    manifest = _task_manifest_bytes(enumerate_tasks(config))
    if manifest_path.exists():
        if manifest_path.read_bytes() != manifest:
            raise ArtifactError("task manifest is incompatible with the resolved configuration")
    else:
        _atomic_write_bytes(manifest_path, manifest)
    provenance_path = layout.run_dir / "provenance.json"
    provenance = {
        "experiment": 3,
        "package_version": __version__,
        "config_hash": config.resolved_hash,
        "code_hash": compute_code_hash(),
        "authority": authority_manifest(),
        "git": git_revision(),
        "python": sys.version,
        "python_executable": sys.executable,
        "initial_command": [sys.executable, *sys.argv],
        "working_directory": str(Path.cwd()),
        "platform": platform.platform(),
        "dependencies": dependency_versions(),
    }
    if provenance_path.exists():
        existing = json.loads(provenance_path.read_text(encoding="utf-8"))
        immutable_keys = (
            "experiment",
            "package_version",
            "config_hash",
            "code_hash",
            "authority",
            "dependencies",
        )
        if any(existing.get(key) != provenance.get(key) for key in immutable_keys):
            raise ArtifactError("run provenance differs from the current code or dependencies")
    else:
        atomic_write_json(provenance_path, provenance)
    return layout


def task_directory(layout: RunLayout, task: ExperimentTask) -> Path:
    return layout.raw_dir / task.task_id


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())


def _replace_directory_with_retry(source: Path, destination: Path) -> None:
    """Atomically commit a task despite short-lived Windows/OneDrive locks."""

    delay = 0.025
    last_error: PermissionError | None = None
    for _ in range(9):
        if destination.exists():
            raise ArtifactError(f"Concurrent writer completed task: {destination}")
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(delay)
            delay = min(0.5, delay * 2.0)
    raise ArtifactError(
        f"atomic task commit remained locked after bounded retries: {destination}"
    ) from last_error


def write_task_artifact(
    layout: RunLayout,
    task: ExperimentTask,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> Path:
    """Commit one task atomically by renaming a complete temporary directory."""

    destination = task_directory(layout, task)
    if destination.exists():
        raise ArtifactError(f"Refusing to overwrite completed task: {destination}")
    temporary = layout.raw_dir / f".{task.task_id}.{os.getpid()}.{time.time_ns()}.tmp"
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        normalized = {name: np.asarray(value) for name, value in arrays.items()}
        if any(
            np.issubdtype(value.dtype, np.number) and np.any(~np.isfinite(value))
            for value in normalized.values()
        ):
            raise ArtifactError(f"Task {task.task_id} contains a non-finite array")
        array_hashes = {
            name: {
                "sha256": array_content_hash(value),
                "shape": list(value.shape),
                "dtype": value.dtype.str,
            }
            for name, value in normalized.items()
        }
        arrays_path = temporary / "arrays.npz"
        _save_npz(arrays_path, normalized)
        full_metadata = metadata | {
            "task": asdict(task) | {"task_id": task.task_id},
            "array_content_hashes": array_hashes,
            "arrays_file_sha256": sha256_file(arrays_path),
        }
        atomic_write_json(temporary / "metadata.json", full_metadata)
        log_payload = (
            json.dumps(
                {
                    "status": "complete",
                    "task_id": task.task_id,
                    "compile_seconds": metadata.get("compile_seconds"),
                    "execution_seconds_excluding_compilation": metadata.get(
                        "execution_seconds_excluding_compilation"
                    ),
                    "throughput_particle_steps_per_second": metadata.get(
                        "throughput_particle_steps_per_second"
                    ),
                    "warnings": metadata.get("warnings", []),
                },
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        _atomic_write_bytes(temporary / "run.log", log_payload)
        complete = {
            "task_id": task.task_id,
            "config_hash": metadata["config_hash"],
            "code_hash": metadata["code_hash"],
            "arrays_file_sha256": full_metadata["arrays_file_sha256"],
            "metadata_file_sha256": sha256_file(temporary / "metadata.json"),
            "log_file_sha256": sha256_file(temporary / "run.log"),
            "status": "complete",
        }
        atomic_write_json(temporary / "COMPLETE.json", complete)
        _replace_directory_with_retry(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def write_failure(layout: RunLayout, task: ExperimentTask, payload: dict[str, Any]) -> Path:
    failure = layout.failed_dir / f"{task.task_id}-{time.time_ns()}.json"
    atomic_write_json(failure, payload | {"task": asdict(task), "status": "failed"})
    return failure


def load_task_artifact(
    layout: RunLayout,
    task: ExperimentTask,
    *,
    expected_config_hash: str,
    expected_code_hash: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    directory = task_directory(layout, task)
    complete_path = directory / "COMPLETE.json"
    metadata_path = directory / "metadata.json"
    arrays_path = directory / "arrays.npz"
    log_path = directory / "run.log"
    if not directory.is_dir() or not all(
        path.is_file() for path in (complete_path, metadata_path, arrays_path, log_path)
    ):
        raise ArtifactError(f"Task is absent or incomplete: {task.task_id}")
    try:
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Corrupt JSON for task {task.task_id}") from exc
    if complete.get("status") != "complete" or complete.get("task_id") != task.task_id:
        raise ArtifactError(f"Invalid completion marker for task {task.task_id}")
    if (
        complete.get("config_hash") != expected_config_hash
        or metadata.get("config_hash") != expected_config_hash
    ):
        raise ArtifactError(f"Configuration mismatch in task {task.task_id}")
    if (
        complete.get("code_hash") != expected_code_hash
        or metadata.get("code_hash") != expected_code_hash
    ):
        raise ArtifactError(f"Code-version mismatch in task {task.task_id}")
    if sha256_file(metadata_path) != complete.get("metadata_file_sha256"):
        raise ArtifactError(f"Metadata checksum mismatch in task {task.task_id}")
    if sha256_file(log_path) != complete.get("log_file_sha256"):
        raise ArtifactError(f"Log checksum mismatch in task {task.task_id}")
    arrays_file_hash = sha256_file(arrays_path)
    if arrays_file_hash != complete.get("arrays_file_sha256") or arrays_file_hash != metadata.get(
        "arrays_file_sha256"
    ):
        raise ArtifactError(f"NPZ checksum mismatch in task {task.task_id}")
    try:
        with np.load(arrays_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise ArtifactError(f"Corrupt NPZ in task {task.task_id}: {exc}") from exc
    expected_arrays = metadata.get("array_content_hashes", {})
    if set(arrays) != set(expected_arrays):
        raise ArtifactError(f"Array inventory mismatch in task {task.task_id}")
    for name, array in arrays.items():
        details = expected_arrays[name]
        if (
            array_content_hash(array) != details.get("sha256")
            or list(array.shape) != details.get("shape")
            or array.dtype.str != details.get("dtype")
        ):
            raise ArtifactError(f"Content mismatch for array {name} in task {task.task_id}")
    return arrays, metadata


def config_source_is_unchanged(config: Experiment3Config) -> bool:
    path = Path(config.source_path)
    return path.is_file() and sha256_file(path) == config.source_hash
