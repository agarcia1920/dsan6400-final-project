#!/usr/bin/env python3
"""Run all coordination scenarios across multiple random seeds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from soulcycle_network.experiment_runner import SCENARIOS, run_full_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run all coordination scenarios across multiple random seeds "
            "and write full seed-level outputs."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_PROJECT_ROOT / "data",
        help="Directory containing tracked input CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "outputs",
        help="Directory for generated seed-level experiment outputs.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=6400,
        help="First random seed in the experiment.",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=10,
        help="Number of consecutive seeds to run.",
    )
    parser.add_argument(
        "--n-weeks",
        type=int,
        default=52,
        help="Number of simulation weeks.",
    )
    parser.add_argument(
        "--n-riders",
        type=int,
        default=10_000,
        help="Number of simulated riders.",
    )
    args = parser.parse_args()

    if args.n_seeds < 1:
        parser.error("--n-seeds must be at least 1")
    if args.n_weeks < 1:
        parser.error("--n-weeks must be at least 1")
    if args.n_riders < 1:
        parser.error("--n-riders must be at least 1")

    seeds = list(range(args.base_seed, args.base_seed + args.n_seeds))

    print(
        f"Running {len(SCENARIOS)} scenarios across "
        f"{len(seeds)} seeds..."
    )

    master = run_full_experiment(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seeds=seeds,
        n_weeks=args.n_weeks,
        n_riders=args.n_riders,
    )

    master_path = args.output_dir / "longitudinal_metrics_master.csv"
    print(f"Wrote master table: {master_path}")
    print(f"Rows: {len(master):,}")


if __name__ == "__main__":
    main()