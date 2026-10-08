"""Command-line driver for the target-specific tail experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from levy_experiments.devices import configure_runtime
from levy_experiments.errors import ExperimentError
from levy_experiments.pilot_gate import verify_pilot_gate

from .config import enumerate_tasks, load_config
from .runner import ensure_score_tables, run_task
from .storage import prepare_run, run_layout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run target-specific Student-tail diagnostics for stable and VP samplers"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "gpu", "auto"), default="auto")
    parser.add_argument("--precision", choices=("float32", "float64", "mixed"), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--pilot-output-dir",
        type=Path,
        default=None,
        help="output root containing the matching passed pilot (defaults to --output-dir)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-publication-scale", action="store_true")
    parser.add_argument(
        "--task-index",
        default="all",
        help="Deterministic Slurm array index, or 'all' for sequential execution",
    )
    parser.add_argument("--table-only", action="store_true")
    parser.add_argument("--print-task-count", action="store_true")
    return parser


def _select_tasks(tasks, requested: str):
    if requested == "all":
        return tasks
    try:
        index = int(requested)
    except ValueError as exc:
        raise ExperimentError("--task-index must be an integer or 'all'") from exc
    if not 0 <= index < len(tasks):
        raise ExperimentError(f"task index {index} is outside [0,{len(tasks) - 1}]")
    return (tasks[index],)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(
            args.config,
            precision_override=args.precision,
            seed_override=args.seed,
            allow_final=args.allow_publication_scale,
        )
        tasks = enumerate_tasks(config)
        if args.print_task_count:
            print(len(tasks))
            return 0
        selected = _select_tasks(tasks, args.task_index)
        layout = run_layout(args.output_dir, config)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "config_hash": config.resolved_hash,
                        "publication_scale": config.experiment.publication_scale,
                        "run_dir": str(layout.run_dir),
                        "selected_tasks": [
                            {**task.__dict__, "task_id": task.task_id} for task in selected
                        ],
                        "task_count": len(tasks),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if config.experiment.publication_scale:
            verify_pilot_gate(
                experiment=1,
                final_config=args.config,
                output_root=args.output_dir,
                pilot_output_root=args.pilot_output_dir,
                precision=args.precision,
            )

        device, runtime = configure_runtime(args.device, config.experiment.precision)
        layout = prepare_run(args.output_dir, config)
        tables = ensure_score_tables(layout, config, resume=args.resume)
        print(
            json.dumps(
                {
                    "event": "score_tables_ready",
                    "hashes": {key: table.table_hash for key, table in tables.items()},
                },
                sort_keys=True,
            )
        )
        if args.table_only:
            return 0
        for task in selected:
            result = run_task(
                config,
                task,
                layout,
                device=device,
                runtime=runtime,
                tables=tables,
                resume=args.resume,
            )
            print(
                json.dumps(
                    {"event": "task", "status": result["status"], "task_id": task.task_id},
                    sort_keys=True,
                )
            )
        return 0
    except ExperimentError as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
