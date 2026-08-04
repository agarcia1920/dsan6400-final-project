#!/usr/bin/env python3
"""Run one scenario and seed and export its analysis files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from soulcycle_network.experiment_runner import (
    SCENARIOS,
    run_experiment_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one simulation scenario for one random seed "
            "and export its analysis files."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_PROJECT_ROOT / "data",
        help="Directory containing the model input data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "outputs",
        help="Directory for generated simulation outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=6400,
        help="Random seed used for the simulation.",
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS),
        default="baseline",
        help="Coordination scenario to simulate.",
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

    if args.n_weeks < 1:
        parser.error("--n-weeks must be at least 1")

    if args.n_riders < 1:
        parser.error("--n-riders must be at least 1")

    scenario = SCENARIOS[args.scenario]

    run_experiment_seed(
        data_dir=args.data_dir,
        scenario=scenario,
        seed=args.seed,
        output_dir=args.output_dir,
        n_weeks=args.n_weeks,
        n_riders=args.n_riders,
    )

    seed_output_dir = (
        args.output_dir
        / scenario.name
        / f"seed_{args.seed}"
    )
    summary_file = (
        seed_output_dir
        / "simulation_summary.csv"
    )

    if not summary_file.is_file():
        raise FileNotFoundError(
            "Simulation completed without producing "
            f"the expected summary: {summary_file}"
        )

    print(f"Wrote simulation outputs to {seed_output_dir}")


if __name__ == "__main__":
    main()