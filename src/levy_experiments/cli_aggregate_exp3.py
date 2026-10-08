"""Aggregation and rendering CLI for Experiment 3."""

from __future__ import annotations

import argparse
import json

from .analysis_exp3 import aggregate_experiment3
from .cli_common import add_common_arguments, resolved_config_from_args
from .config import enumerate_tasks
from .errors import ExperimentError
from .plotting import render_experiment3_figure
from .storage import prepare_run, run_layout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify, aggregate, and render a complete Experiment 3 run"
    )
    add_common_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = resolved_config_from_args(args)
        layout = run_layout(args.output_dir, config)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "config_hash": config.resolved_hash,
                        "expected_tasks": len(enumerate_tasks(config)),
                        "run_dir": str(layout.run_dir),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        layout = prepare_run(args.output_dir, config)
        rows, summaries, diagnostics = aggregate_experiment3(config, layout, resume=args.resume)
        figures = render_experiment3_figure(
            config, layout, rows, diagnostics, resume=args.resume
        )
        print(
            json.dumps(
                {
                    "diagnostics": diagnostics,
                    "figures": figures,
                    "metric_rows": len(rows),
                    "summary_rows": len(summaries),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except ExperimentError as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
