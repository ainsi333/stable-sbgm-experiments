"""Run independent experiment tasks concurrently on a CPU-only workstation."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
import tomllib
from pathlib import Path

from levy_experiments.errors import ArtifactError
from levy_experiments.pilot_gate import verify_pilot_gate

MODULES = {
    1: ("experiments.exp1.run", "experiments.exp1.aggregate"),
    2: ("experiments.exp2.run", "experiments.exp2.aggregate"),
    3: ("levy_experiments.cli_exp3", "levy_experiments.cli_aggregate_exp3"),
}


def _environment(root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root)))
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS",
    ):
        environment[name] = "1"
    environment.setdefault(
        "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    )
    environment["JAX_ENABLE_X64"] = "true"
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    return environment


def _execute(command: list[str], *, root: Path, environment: dict[str, str]) -> str:
    result = subprocess.run(
        command, cwd=root, env=environment, text=True, capture_output=True, check=False
    )
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return result.stdout


def _is_publication_scale(config_path: Path) -> bool:
    try:
        payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError(f"cannot inspect configuration {config_path}") from exc
    experiment = payload.get("experiment")
    return isinstance(experiment, dict) and experiment.get("publication_scale") is True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=int, choices=MODULES, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pilot-output-dir",
        type=Path,
        default=None,
        help="output root containing the matching passed pilot (defaults to --output-dir)",
    )
    parser.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--precision", choices=("float32", "float64", "mixed"), default="float64")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-publication-scale", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    root = Path(__file__).resolve().parents[1]
    resolved_config = args.config.resolve()
    publication_scale = _is_publication_scale(resolved_config)
    if publication_scale and not args.dry_run:
        if not args.allow_publication_scale:
            parser.error("publication-scale execution requires --allow-publication-scale")
        try:
            receipt = verify_pilot_gate(
                experiment=args.experiment,
                final_config=resolved_config,
                output_root=args.output_dir,
                pilot_output_root=args.pilot_output_dir,
                precision=args.precision,
            )
        except ArtifactError as exc:
            parser.error(str(exc))
        print(
            "pilot_gate=passed "
            f"run_dir={receipt.run_dir} code_hash={receipt.code_hash}"
        )
    runner, aggregator = MODULES[args.experiment]
    common = [
        "--config",
        str(resolved_config),
        "--device",
        "cpu",
        "--precision",
        args.precision,
        "--output-dir",
        str(args.output_dir.resolve()),
    ]
    if args.resume:
        common.append("--resume")
    if args.allow_publication_scale:
        common.append("--allow-publication-scale")
    if args.pilot_output_dir is not None:
        common.extend(("--pilot-output-dir", str(args.pilot_output_dir.resolve())))
    environment = _environment(root)
    base = [sys.executable, "-m", runner, *common]
    count = int(
        _execute([*base, "--dry-run", "--print-task-count"], root=root, environment=environment)
    )
    if args.dry_run:
        print(f"experiment={args.experiment} tasks={count} jobs={args.jobs}")
        return 0
    print(_execute([*base, "--table-only"], root=root, environment=environment), end="")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                _execute,
                [*base, "--task-index", str(index)],
                root=root,
                environment=environment,
            ): index
            for index in range(count)
        }
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            index = futures[future]
            print(f"[{completed}/{count}] task_index={index}")
            print(future.result(), end="")
    aggregate = [sys.executable, "-m", aggregator, *common]
    print(_execute(aggregate, root=root, environment=environment), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
