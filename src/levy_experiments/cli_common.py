"""Shared command-line contract for Experiment 3 executables."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import Precision, load_config
from .errors import ConfigurationError
from .pilot_gate import verify_pilot_gate


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path, help="Experiment 3 TOML file")
    parser.add_argument(
        "--device", choices=("cpu", "gpu", "auto"), default="auto", help="Requested backend"
    )
    parser.add_argument(
        "--precision",
        choices=("float32", "float64", "mixed"),
        default=None,
        help="Override the configured propagation precision",
    )
    parser.add_argument("--seed", type=int, default=None, help="Run only this scientific seed")
    parser.add_argument("--output-dir", required=True, type=Path, help="Root for immutable runs")
    parser.add_argument(
        "--pilot-output-dir",
        type=Path,
        default=None,
        help="output root containing the matching passed pilot (defaults to --output-dir)",
    )
    parser.add_argument("--resume", action="store_true", help="Verify and skip completed outputs")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print without writing")
    parser.add_argument(
        "--allow-publication-scale",
        action="store_true",
        help="Explicit guard required to execute a publication-scale configuration",
    )


def resolved_config_from_args(args: argparse.Namespace):
    precision: Precision | None = args.precision
    config = load_config(args.config, precision_override=precision, seed_override=args.seed)
    if (
        config.experiment.publication_scale
        and not args.dry_run
        and not args.allow_publication_scale
    ):
        raise ConfigurationError(
            "publication-scale execution is locked; pass --allow-publication-scale "
            "only after approval"
        )
    if config.experiment.publication_scale and not args.dry_run:
        verify_pilot_gate(
            experiment=3,
            final_config=args.config,
            output_root=args.output_dir,
            pilot_output_root=args.pilot_output_dir,
            precision=args.precision,
        )
    return config
