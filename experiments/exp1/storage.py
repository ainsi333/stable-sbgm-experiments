"""Atomic, content-checked storage and provenance for Experiment 1."""

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

from levy_experiments.authority import authority_manifest
from levy_experiments.errors import ArtifactError

from .config import (
    Experiment1Config,
    Experiment1Task,
    enumerate_tasks,
    validate_execution,
)

EXP1_SHARED_SOURCE_FILES = (
    "src/levy_experiments/__init__.py",
    "src/levy_experiments/authority.py",
    "src/levy_experiments/devices.py",
    "src/levy_experiments/errors.py",
    "src/levy_experiments/integrators.py",
    "src/levy_experiments/metrics/characteristic.py",
    "src/levy_experiments/random.py",
    "src/levy_experiments/stable.py",
)


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


def _code_files(root: Path) -> tuple[Path, ...]:
    required = [root / relative for relative in EXP1_SHARED_SOURCE_FILES]
    required.append(root / "pyproject.toml")
    local_root = root / "experiments" / "exp1"
    if not local_root.is_dir():
        raise ArtifactError(f"Experiment 1 source directory is missing: {local_root}")
    missing = [path for path in required if not path.is_file()]
    if missing:
        names = ", ".join(path.relative_to(root).as_posix() for path in missing)
        raise ArtifactError(f"Experiment 1 code-hash inputs are missing: {names}")
    local = [path for path in local_root.rglob("*.py") if path.is_file()]
    candidates = {path.resolve() for path in (*required, *local)}
    return tuple(sorted(candidates, key=lambda path: path.relative_to(root).as_posix()))


def compute_code_hash(root: Path | None = None) -> str:
    root = project_root() if root is None else root.resolve()
    digest = hashlib.sha256()
    for path in _code_files(root):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
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
                ["git", "status", "--porcelain", "--", "."],
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


def _nvidia_metadata() -> dict[str, Any] | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    query = "name,uuid,memory.total,driver_version"
    try:
        output = subprocess.run(
            [executable, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except subprocess.SubprocessError:
        return {"query_failed": True}
    devices = []
    for line in output.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 4:
            devices.append(
                {
                    "name": fields[0],
                    "uuid": fields[1],
                    "memory_mib": fields[2],
                    "driver": fields[3],
                }
            )
    return {"devices": devices}


def scheduler_metadata() -> dict[str, str | None]:
    names = (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_NODELIST",
        "CUDA_VISIBLE_DEVICES",
        "APPTAINER_CONTAINER",
        "EXP1_SIF_SHA256",
        "OCI_IMAGE_DIGEST",
    )
    return {name: os.environ.get(name) for name in names}


@dataclass(frozen=True)
class RunLayout:
    root: Path
    run_dir: Path
    raw_dir: Path
    failed_dir: Path
    aggregate_dir: Path
    figure_dir: Path
    log_dir: Path


def run_layout(output_root: str | Path, config: Experiment1Config) -> RunLayout:
    root = Path(output_root).resolve()
    run_dir = root / f"exp1-{config.resolved_hash[:12]}"
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


def atomic_write_json(path: Path, payload: dict[str, Any], *, refuse_existing: bool = True) -> None:
    try:
        serialized = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"JSON payload is not finite and serializable: {path}") from exc
    _atomic_write_bytes(path, serialized, refuse_existing=refuse_existing)


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Corrupt {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise ArtifactError(f"Invalid {description}: {path}")
    return payload


def _write_or_verify(path: Path, payload: bytes, description: str) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ArtifactError(f"Existing {description} is incompatible: {path}")
        return
    try:
        _atomic_write_bytes(path, payload)
    except ArtifactError:
        if not path.is_file() or path.read_bytes() != payload:
            raise


def _task_manifest_bytes(tasks: tuple[Experiment1Task, ...]) -> bytes:
    fields = (
        "index",
        "task_id",
        "kind",
        "model",
        "nu",
        "steps",
        "seed",
        "particles",
        "control_particles",
        "purpose",
        "coupling_steps",
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for task in tasks:
        row = asdict(task) | {"task_id": task.task_id}
        writer.writerow({name: row[name] for name in fields})
    return buffer.getvalue().encode("utf-8")


def _resolved_config_bytes(config: Experiment1Config) -> bytes:
    payload = config.canonical_dict(include_source_path=False) | {
        "resolved_hash": config.resolved_hash,
        "requested_config_source_hash": config.source_hash,
    }
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def prepare_run(output_root: str | Path, config: Experiment1Config) -> RunLayout:
    validate_execution(config)
    if not config_source_is_unchanged(config):
        raise ArtifactError("configuration source changed after it was loaded")
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
    _write_or_verify(
        layout.run_dir / "resolved_config.json",
        _resolved_config_bytes(config),
        "resolved configuration",
    )
    _write_or_verify(
        layout.run_dir / "requested_config.toml",
        Path(config.source_path).read_bytes(),
        "requested configuration",
    )
    _write_or_verify(
        layout.run_dir / "tasks.csv",
        _task_manifest_bytes(enumerate_tasks(config)),
        "task manifest",
    )
    provenance = {
        "experiment": 1,
        "config_hash": config.resolved_hash,
        "code_hash": compute_code_hash(),
        "authority": authority_manifest(),
        "git": git_revision(),
        "python": sys.version,
        "python_executable": sys.executable,
        "initial_command": [sys.executable, *sys.argv],
        "working_directory": str(Path.cwd()),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "dependencies": dependency_versions(),
        "nvidia": _nvidia_metadata(),
        "scheduler": scheduler_metadata(),
    }
    provenance_path = layout.run_dir / "provenance.json"
    if provenance_path.exists():
        existing = _read_json(provenance_path, "run provenance")
        immutable = ("experiment", "config_hash", "code_hash", "authority", "dependencies")
        if any(existing.get(name) != provenance[name] for name in immutable):
            raise ArtifactError("run provenance differs from current code or dependencies")
    else:
        atomic_write_json(provenance_path, provenance)
    return layout


def task_directory(layout: RunLayout, task: Experiment1Task) -> Path:
    return layout.raw_dir / task.task_id


def _normalize_arrays(
    task: Experiment1Task, arrays: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    if not arrays or not all(isinstance(name, str) and name for name in arrays):
        raise ArtifactError(f"Task {task.task_id} must contain named arrays")
    normalized = {name: np.asarray(value) for name, value in arrays.items()}
    for name, value in normalized.items():
        if value.dtype.hasobject:
            raise ArtifactError(f"Object array is forbidden in task {task.task_id}: {name}")
        if np.issubdtype(value.dtype, np.number) and np.any(~np.isfinite(value)):
            raise ArtifactError(f"Task {task.task_id} contains non-finite values: {name}")
    return normalized


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())


def _replace_directory(source: Path, destination: Path) -> None:
    delay = 0.025
    last_error: PermissionError | None = None
    for _ in range(9):
        if destination.exists():
            raise ArtifactError(f"Concurrent writer completed task: {destination}")
        try:
            os.replace(source, destination)
            return
        except FileExistsError as exc:
            raise ArtifactError(f"Concurrent writer completed task: {destination}") from exc
        except PermissionError as exc:
            last_error = exc
            time.sleep(delay)
            delay = min(0.5, delay * 2.0)
    raise ArtifactError(f"atomic task commit remained locked: {destination}") from last_error


def write_task_artifact(
    layout: RunLayout,
    task: Experiment1Task,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    *,
    resume: bool = False,
) -> Path:
    normalized = _normalize_arrays(task, arrays)
    destination = task_directory(layout, task)
    if destination.exists():
        if not resume:
            raise ArtifactError(f"Task already exists: {destination}")
        actual, _ = load_task_artifact(
            layout,
            task,
            expected_config_hash=str(metadata["config_hash"]),
            expected_code_hash=str(metadata["code_hash"]),
        )
        if set(actual) != set(normalized) or any(
            array_content_hash(actual[name]) != array_content_hash(normalized[name])
            for name in normalized
        ):
            raise ArtifactError(f"Resume content differs for task {task.task_id}")
        return destination

    temporary = layout.raw_dir / f".{task.task_id}.{os.getpid()}.{time.time_ns()}.tmp"
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        npz_path = temporary / "arrays.npz"
        _save_npz(npz_path, normalized)
        array_hashes = {name: array_content_hash(value) for name, value in normalized.items()}
        complete_metadata = dict(metadata) | {
            "task": asdict(task) | {"task_id": task.task_id},
            "array_hashes": array_hashes,
            "npz_sha256": sha256_file(npz_path),
        }
        metadata_bytes = (
            json.dumps(complete_metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        (temporary / "metadata.json").write_bytes(metadata_bytes)
        log_bytes = (
            json.dumps(
                {
                    "task_id": task.task_id,
                    "status": "complete",
                    "warnings": complete_metadata.get("warnings", []),
                },
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        (temporary / "run.log").write_bytes(log_bytes)
        complete = {
            "task_id": task.task_id,
            "status": "complete",
            "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
            "log_sha256": hashlib.sha256(log_bytes).hexdigest(),
            "npz_sha256": complete_metadata["npz_sha256"],
        }
        (temporary / "COMPLETE.json").write_text(
            json.dumps(complete, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _replace_directory(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    return destination


def load_task_artifact(
    layout: RunLayout,
    task: Experiment1Task,
    *,
    expected_config_hash: str,
    expected_code_hash: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    directory = task_directory(layout, task)
    complete_path = directory / "COMPLETE.json"
    metadata_path = directory / "metadata.json"
    log_path = directory / "run.log"
    npz_path = directory / "arrays.npz"
    if not all(path.is_file() for path in (complete_path, metadata_path, log_path, npz_path)):
        raise ArtifactError(f"Task is absent or incomplete: {task.task_id}")
    complete = _read_json(complete_path, f"completion marker for {task.task_id}")
    metadata = _read_json(metadata_path, f"metadata for {task.task_id}")
    if complete.get("status") != "complete" or complete.get("task_id") != task.task_id:
        raise ArtifactError(f"Invalid completion marker for task {task.task_id}")
    if metadata.get("task") != asdict(task) | {"task_id": task.task_id}:
        raise ArtifactError(f"Task identity mismatch in artifact {task.task_id}")
    if metadata.get("config_hash") != expected_config_hash:
        raise ArtifactError(f"Configuration mismatch in task {task.task_id}")
    if metadata.get("code_hash") != expected_code_hash:
        raise ArtifactError(f"Code-version mismatch in task {task.task_id}")
    if complete.get("metadata_sha256") != sha256_file(metadata_path):
        raise ArtifactError(f"Metadata checksum mismatch in task {task.task_id}")
    if complete.get("log_sha256") != sha256_file(log_path):
        raise ArtifactError(f"Log checksum mismatch in task {task.task_id}")
    if complete.get("npz_sha256") != sha256_file(npz_path):
        raise ArtifactError(f"NPZ checksum mismatch in task {task.task_id}")
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError, EOFError) as exc:
        raise ArtifactError(f"Corrupt NPZ in task {task.task_id}") from exc
    hashes = metadata.get("array_hashes")
    if not isinstance(hashes, dict) or set(hashes) != set(arrays):
        raise ArtifactError(f"Array inventory mismatch in task {task.task_id}")
    for name, value in arrays.items():
        if hashes[name] != array_content_hash(value):
            raise ArtifactError(f"Content mismatch for array {name} in task {task.task_id}")
    _normalize_arrays(task, arrays)
    return arrays, metadata


def write_failure(layout: RunLayout, task: Experiment1Task, payload: dict[str, Any]) -> Path:
    layout.failed_dir.mkdir(parents=True, exist_ok=True)
    path = layout.failed_dir / f"{task.task_id}-{time.time_ns()}.json"
    atomic_write_json(path, {"task_id": task.task_id, "status": "failed"} | payload)
    return path


def config_source_is_unchanged(config: Experiment1Config) -> bool:
    path = Path(config.source_path)
    return path.is_file() and sha256_file(path) == config.source_hash


__all__ = [
    "RunLayout",
    "array_content_hash",
    "atomic_write_json",
    "compute_code_hash",
    "config_source_is_unchanged",
    "load_task_artifact",
    "prepare_run",
    "run_layout",
    "sha256_file",
    "task_directory",
    "write_failure",
    "write_task_artifact",
]
