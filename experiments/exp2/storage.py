"""Atomic, hash-checked storage dedicated to Experiment 2.

The Experiment 2 hash deliberately covers its complete local implementation,
the shared numerical modules it imports, and ``pyproject.toml``.  It does not
change the Experiment 3 hash because no Experiment 3 source or hashing code is
modified here.
"""

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
    Experiment2Config,
    Experiment2Task,
    enumerate_tasks,
    validate_execution,
)

# This is the transitive shared-source surface currently used by Experiment 2.
# Experiment-specific sources are discovered recursively below so adding a new
# local runner, score implementation, analysis, or plotting module changes the
# hash without requiring this list to be edited.
EXP2_SHARED_SOURCE_FILES = (
    "src/levy_experiments/__init__.py",
    "src/levy_experiments/authority.py",
    "src/levy_experiments/config.py",
    "src/levy_experiments/devices.py",
    "src/levy_experiments/errors.py",
    "src/levy_experiments/integrators.py",
    "src/levy_experiments/random.py",
    "src/levy_experiments/stable.py",
    "src/levy_experiments/storage.py",
    "src/levy_experiments/theory.py",
    "src/levy_experiments/metrics/__init__.py",
    "src/levy_experiments/metrics/characteristic.py",
    "src/levy_experiments/metrics/mmd.py",
    "src/levy_experiments/metrics/wasserstein.py",
    "src/levy_experiments/scores/__init__.py",
    "src/levy_experiments/scores/discrete.py",
    "src/levy_experiments/scores/stable_table.py",
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
    required = [root / relative for relative in EXP2_SHARED_SOURCE_FILES]
    required.append(root / "pyproject.toml")
    missing = [path for path in required if not path.is_file()]
    if missing:
        listed = ", ".join(path.relative_to(root).as_posix() for path in missing)
        raise ArtifactError(f"Experiment 2 code-hash inputs are missing: {listed}")

    local_root = root / "experiments" / "exp2"
    if not local_root.is_dir():
        raise ArtifactError(f"Experiment 2 source directory is missing: {local_root}")
    local = [path for path in local_root.rglob("*.py") if path.is_file()]
    candidates = {path.resolve() for path in (*required, *local)}
    return tuple(sorted(candidates, key=lambda path: path.relative_to(root).as_posix()))


def compute_code_hash(root: Path | None = None) -> str:
    """Hash Exp2 code inputs independently of Git status and Exp3 outputs."""

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


@dataclass(frozen=True)
class RunLayout:
    root: Path
    run_dir: Path
    raw_dir: Path
    failed_dir: Path
    aggregate_dir: Path
    figure_dir: Path
    log_dir: Path


def run_layout(output_root: str | Path, config: Experiment2Config) -> RunLayout:
    root = Path(output_root).resolve()
    run_dir = root / f"exp2-{config.resolved_hash[:12]}"
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
        serialized = (
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"JSON payload is not finite and serializable: {path}") from exc
    _atomic_write_bytes(path, serialized, refuse_existing=refuse_existing)


def _read_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Corrupt {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise ArtifactError(f"Invalid {description}: {path}")
    return payload


def _write_or_verify_bytes(path: Path, payload: bytes, *, description: str) -> None:
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError as exc:
            raise ArtifactError(f"Cannot read existing {description}: {path}") from exc
        if existing != payload:
            raise ArtifactError(f"Existing {description} is incompatible: {path}")
        return
    try:
        _atomic_write_bytes(path, payload)
    except ArtifactError:
        # A concurrent array task may have written the same immutable run file.
        if not path.is_file() or path.read_bytes() != payload:
            raise


def _task_manifest_bytes(tasks: tuple[Experiment2Task, ...]) -> bytes:
    buffer = io.StringIO(newline="")
    fields = (
        "index",
        "task_id",
        "model",
        "horizon",
        "steps",
        "seed",
        "purposes",
        "initialization_arms",
        "reference_samples",
    )
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for task in tasks:
        row = asdict(task)
        row["task_id"] = task.task_id
        for name in ("purposes", "initialization_arms", "reference_samples"):
            row[name] = json.dumps(row[name], separators=(",", ":"))
        writer.writerow({name: row[name] for name in fields})
    return buffer.getvalue().encode("utf-8")


def _resolved_config_bytes(config: Experiment2Config) -> bytes:
    payload = config.canonical_dict(include_source_path=False) | {
        "resolved_hash": config.resolved_hash,
        "requested_config_source_hash": config.source_hash,
    }
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def prepare_run(output_root: str | Path, config: Experiment2Config) -> RunLayout:
    """Create or verify immutable run provenance before any final-scale write."""

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

    _write_or_verify_bytes(
        layout.run_dir / "resolved_config.json",
        _resolved_config_bytes(config),
        description="resolved configuration",
    )
    _write_or_verify_bytes(
        layout.run_dir / "requested_config.toml",
        Path(config.source_path).read_bytes(),
        description="requested configuration",
    )
    _write_or_verify_bytes(
        layout.run_dir / "tasks.csv",
        _task_manifest_bytes(enumerate_tasks(config)),
        description="task manifest",
    )

    code_hash = compute_code_hash()
    provenance_path = layout.run_dir / "provenance.json"
    provenance = {
        "experiment": 2,
        "config_hash": config.resolved_hash,
        "code_hash": code_hash,
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
        existing = _read_json(provenance_path, description="run provenance")
        immutable = ("experiment", "config_hash", "code_hash", "authority", "dependencies")
        if any(existing.get(name) != provenance[name] for name in immutable):
            raise ArtifactError("run provenance differs from current code or dependencies")
    else:
        atomic_write_json(provenance_path, provenance)
    return layout


def task_directory(layout: RunLayout, task: Experiment2Task) -> Path:
    return layout.raw_dir / task.task_id


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())


def _replace_directory_with_retry(source: Path, destination: Path) -> None:
    """Atomically commit a task despite bounded Windows/OneDrive locks."""

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
    raise ArtifactError(
        f"atomic task commit remained locked after bounded retries: {destination}"
    ) from last_error


def _normalize_arrays(
    task: Experiment2Task, arrays: dict[str, np.ndarray]
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


def _verify_resumed_arrays(
    task: Experiment2Task,
    expected: dict[str, np.ndarray],
    actual: dict[str, np.ndarray],
) -> None:
    if set(expected) != set(actual):
        raise ArtifactError(f"Resume array inventory differs for task {task.task_id}")
    for name in expected:
        if array_content_hash(expected[name]) != array_content_hash(actual[name]):
            raise ArtifactError(f"Resume content differs for array {name} in task {task.task_id}")


def _validate_task_hashes_against_run(
    layout: RunLayout, *, config_hash: str, code_hash: str
) -> None:
    resolved = _read_json(
        layout.run_dir / "resolved_config.json", description="resolved configuration"
    )
    provenance = _read_json(layout.run_dir / "provenance.json", description="run provenance")
    if resolved.get("resolved_hash") != config_hash:
        raise ArtifactError("Configuration mismatch between task and prepared run")
    if provenance.get("code_hash") != code_hash:
        raise ArtifactError("Code-version mismatch between task and prepared run")


def write_task_artifact(
    layout: RunLayout,
    task: Experiment2Task,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    *,
    resume: bool = False,
) -> Path:
    """Commit one task atomically, or verify exact content during resume."""

    normalized = _normalize_arrays(task, arrays)
    try:
        config_hash = str(metadata["config_hash"])
        code_hash = str(metadata["code_hash"])
    except KeyError as exc:
        raise ArtifactError("task metadata must contain config_hash and code_hash") from exc
    _validate_task_hashes_against_run(
        layout, config_hash=config_hash, code_hash=code_hash
    )
    destination = task_directory(layout, task)
    if destination.exists():
        if not resume:
            raise ArtifactError(f"Refusing to overwrite completed task: {destination}")
        existing_arrays, _ = load_task_artifact(
            layout,
            task,
            expected_config_hash=config_hash,
            expected_code_hash=code_hash,
        )
        _verify_resumed_arrays(task, normalized, existing_arrays)
        return destination

    layout.raw_dir.mkdir(parents=True, exist_ok=True)
    temporary = layout.raw_dir / f".{task.task_id}.{os.getpid()}.{time.time_ns()}.tmp"
    temporary.mkdir(parents=False, exist_ok=False)
    try:
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
            "config_hash": config_hash,
            "code_hash": code_hash,
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
            "config_hash": config_hash,
            "code_hash": code_hash,
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


def write_failure(
    layout: RunLayout, task: Experiment2Task, payload: dict[str, Any]
) -> Path:
    layout.failed_dir.mkdir(parents=True, exist_ok=True)
    failure = layout.failed_dir / f"{task.task_id}-{time.time_ns()}.json"
    atomic_write_json(failure, payload | {"task": asdict(task), "status": "failed"})
    return failure


def load_task_artifact(
    layout: RunLayout,
    task: Experiment2Task,
    *,
    expected_config_hash: str,
    expected_code_hash: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    directory = task_directory(layout, task)
    complete_path = directory / "COMPLETE.json"
    metadata_path = directory / "metadata.json"
    arrays_path = directory / "arrays.npz"
    log_path = directory / "run.log"
    required = (complete_path, metadata_path, arrays_path, log_path)
    if not directory.is_dir() or not all(path.is_file() for path in required):
        raise ArtifactError(f"Task is absent or incomplete: {task.task_id}")

    complete = _read_json(complete_path, description=f"completion marker for {task.task_id}")
    metadata = _read_json(metadata_path, description=f"metadata for {task.task_id}")
    if complete.get("status") != "complete" or complete.get("task_id") != task.task_id:
        raise ArtifactError(f"Invalid completion marker for task {task.task_id}")
    expected_task = asdict(task) | {"task_id": task.task_id}
    if metadata.get("task") != json.loads(json.dumps(expected_task)):
        raise ArtifactError(f"Task identity mismatch in artifact {task.task_id}")
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
    arrays = _normalize_arrays(task, arrays)
    expected_arrays = metadata.get("array_content_hashes")
    if not isinstance(expected_arrays, dict) or set(arrays) != set(expected_arrays):
        raise ArtifactError(f"Array inventory mismatch in task {task.task_id}")
    for name, array in arrays.items():
        details = expected_arrays[name]
        if not isinstance(details, dict) or (
            array_content_hash(array) != details.get("sha256")
            or list(array.shape) != details.get("shape")
            or array.dtype.str != details.get("dtype")
        ):
            raise ArtifactError(f"Content mismatch for array {name} in task {task.task_id}")
    return arrays, metadata


def config_source_is_unchanged(config: Experiment2Config) -> bool:
    path = Path(config.source_path)
    return path.is_file() and sha256_file(path) == config.source_hash
