"""Simulation CLI for Experiment 3."""

from __future__ import annotations

import argparse
import json

from .cli_common import add_common_arguments, resolved_config_from_args
from .config import enumerate_tasks
from .devices import configure_runtime
from .errors import ExperimentError
from .experiment3 import ensure_score_table, run_task
from .storage import prepare_run, run_layout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run exact stationary pairs and three-atom Popov tasks for Experiment 3"
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--task-index",
        default="all",
        help="Deterministic Slurm task index, or 'all' for a local sequential run",
    )
    parser.add_argument(
        "--table-only", action="store_true", help="Build and validate the score table, then stop"
    )
    parser.add_argument(
        "--print-task-count", action="store_true", help="Print only the deterministic task count"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = resolved_config_from_args(args)
        tasks = enumerate_tasks(config)
        if args.print_task_count:
            print(len(tasks))
            return 0
        if args.task_index == "all":
            selected = tasks
        else:
            try:
                index = int(args.task_index)
            except ValueError as exc:
                raise ExperimentError("--task-index must be an integer or 'all'") from exc
            if not 0 <= index < len(tasks):
                raise ExperimentError(f"task index {index} is outside [0,{len(tasks) - 1}]")
            selected = (tasks[index],)
        layout = run_layout(args.output_dir, config)
        if args.dry_run:
            payload = {
                "config_hash": config.resolved_hash,
                "publication_scale": config.experiment.publication_scale,
                "run_dir": str(layout.run_dir),
                "selected_tasks": [{**task.__dict__, "task_id": task.task_id} for task in selected],
                "task_count": len(tasks),
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        device, runtime = configure_runtime(args.device, config.experiment.precision)
        layout = prepare_run(args.output_dir, config)
        needs_table = args.table_only or any(
            task.method in {"popov_ei", "hybrid_ei_invalid"} for task in selected
        )
        table = None
        if needs_table:
            table, validation = ensure_score_table(layout, config)
            print(
                json.dumps(
                    {
                        "event": "score_table_ready",
                        "table_hash": table.table_hash,
                        "validation": validation,
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
                table=table,
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
