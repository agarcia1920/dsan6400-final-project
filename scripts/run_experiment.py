#!/usr/bin/env python3
# Every coordination scenario on a seed list; writes longitudinal_metrics_master.csv.

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
    parser = argparse.ArgumentParser(description="Run all scenarios across multiple seeds.")
    parser.add_argument("--data-dir", type=Path, default=_PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=_PROJECT_ROOT / "outputs")
    parser.add_argument("--base-seed", type=int, default=6400)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-weeks", type=int, default=52)
    parser.add_argument("--n-riders", type=int, default=10_000)
    args = parser.parse_args()

    if args.n_seeds < 1:
        parser.error("--n-seeds must be at least 1")
    if args.n_weeks < 1:
        parser.error("--n-weeks must be at least 1")
    if args.n_riders < 1:
        parser.error("--n-riders must be at least 1")

    seeds = list(range(args.base_seed, args.base_seed + args.n_seeds))
    print("Running", len(SCENARIOS), "scenarios across", len(seeds), "seeds...")

    master = run_full_experiment(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seeds=seeds,
        n_weeks=args.n_weeks,
        n_riders=args.n_riders,
    )

    master_path = args.output_dir / "longitudinal_metrics_master.csv"
    print("Wrote master table:", master_path)
    print("Rows:", len(master))


if __name__ == "__main__":
    main()
