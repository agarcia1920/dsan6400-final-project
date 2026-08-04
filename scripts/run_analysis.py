#!/usr/bin/env python3
"""Calibration, longitudinal rebuild, nulls, comparisons, and compact results."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import networkx as nx
import pandas as pd

from soulcycle_network.analysis.io import (
    familiarity_graph_from_seed,
    load_seed_exports,
    rebuild_longitudinal_from_attendance,
    rider_nodes_from_exports,
)
from soulcycle_network.analysis.metrics import summarize_graph
from soulcycle_network.analysis.models import (
    attractiveness_model,
    coattendance_from_attendance,
    compare_null_metrics,
    degree_ks_distance,
    fitness_model,
    matched_random_graph,
    nonlinear_preferential_attachment,
    null_shuffle_attendance_df,
    null_shuffle_attendance_market_df,
)
from soulcycle_network.calibration import default_calibration_seeds, run_calibration_batch
from soulcycle_network.network_formation import NetworkState


def _parse_seed_dir(path: Path) -> tuple[str, int] | None:
    match = re.match(r"seed_(\d+)$", path.name)
    if not match or path.parent.name == "outputs":
        return None
    return path.parent.name, int(match.group(1))


def task_calibration(results_dir: Path, data_dir: Path, base_seed: int, n_seeds: int) -> None:
    seeds = default_calibration_seeds(base_seed, n_seeds)
    results = run_calibration_batch(data_dir, seeds=seeds)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "calibration.csv"
    results.to_csv(out, index=False)
    print("Wrote", out)


def task_longitudinal(outputs: Path) -> None:
    frames: list[pd.DataFrame] = []
    for scenario_dir in sorted(outputs.iterdir()):
        if not scenario_dir.is_dir() or scenario_dir.name.startswith("."):
            continue
        for seed_dir in sorted(scenario_dir.iterdir()):
            parsed = _parse_seed_dir(seed_dir)
            if parsed is None:
                continue
            scenario, seed = parsed
            att_path = seed_dir / "attendance.csv"
            nodes_path = seed_dir / "node_attributes.csv"
            if not att_path.exists() or not nodes_path.exists():
                continue
            long_df = rebuild_longitudinal_from_attendance(
                pd.read_csv(att_path),
                pd.read_csv(nodes_path),
                scenario=scenario,
                seed=seed,
            )
            long_df.to_csv(seed_dir / "longitudinal_metrics.csv", index=False)
            frames.append(long_df)
            print("rebuilt", scenario, seed)
    if not frames:
        raise SystemExit("No seed folders under " + str(outputs))
    master_path = outputs / "longitudinal_metrics_master.csv"
    pd.concat(frames, ignore_index=True).to_csv(master_path, index=False)
    print("Wrote", master_path)


def _observed_state_from_pairs(pairs: pd.DataFrame) -> NetworkState:
    state = NetworkState()
    for row in pairs.itertuples(index=False):
        rider_1, rider_2 = sorted((row.rider_1, row.rider_2))
        key = (rider_1, rider_2)
        state.co_counts[key] = int(row.coattendance_count)
        state.tie_strength[key] = float(row.tie_strength)
    return state


def task_nulls(seed_dir: Path, output: Path, seed: int) -> None:
    exports = load_seed_exports(seed_dir)
    nodes = exports["node_attributes"]
    attendance = exports["attendance"].copy()
    pairs = exports["pair_history"]
    if "home_market" not in attendance.columns:
        attendance = attendance.merge(nodes[["rider_id", "home_market"]], on="rider_id", how="left")
    home_cluster = nodes.set_index("rider_id")["home_cluster"].to_dict()
    home_market = nodes.set_index("rider_id")["home_market"].to_dict()
    observed = _observed_state_from_pairs(pairs)
    frames = []
    for label, shuffled in (
        ("global_shuffle", null_shuffle_attendance_df(attendance, np.random.default_rng(seed))),
        ("market_shuffle", null_shuffle_attendance_market_df(attendance, np.random.default_rng(seed + 1))),
    ):
        null_state = coattendance_from_attendance(shuffled)
        comparison = compare_null_metrics(observed, null_state, home_cluster, home_market)
        comparison["null_model"] = label
        frames.append(comparison)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(output, index=False)
    print("Wrote", output)


def task_comparisons(seed_dir: Path, output: Path, seed: int, n_replicates: int) -> None:
    observed = familiarity_graph_from_seed(seed_dir, include_isolates=False)
    n, m = observed.number_of_nodes(), observed.number_of_edges()
    if n < 2 or m < 1:
        raise ValueError("Need a non-empty familiarity graph for comparisons.")
    m_links = max(1, int(round(m / n)))

    def row(name: str, parameterization: str, graph: nx.Graph, replicate: int | None) -> dict:
        return {
            "model": name,
            "parameterization": parameterization,
            "seed": seed,
            "replicate": replicate,
            "degree_ks_distance_from_observed": 0.0
            if name == "observed"
            else degree_ks_distance(observed, graph),
            **summarize_graph(graph),
        }

    rows = [row("observed", "", observed, None)]
    for replicate in range(n_replicates):
        rep_seed = seed + replicate
        for name, param, graph in (
            ("erdos_renyi", "matched_m", matched_random_graph(n=n, m=m, seed=rep_seed)),
            (
                "preferential_attachment",
                f"alpha=1.0;m_links={m_links}",
                nonlinear_preferential_attachment(n=n, m_links=m_links, alpha=1.0, seed=rep_seed + 10_000),
            ),
            (
                "attractiveness",
                f"A=1.0;m_links={m_links}",
                attractiveness_model(n=n, m_links=m_links, attractiveness=1.0, seed=rep_seed + 20_000),
            ),
            (
                "fitness",
                f"uniform;m_links={m_links}",
                fitness_model(n=n, m_links=m_links, seed=rep_seed + 30_000)[0],
            ),
        ):
            rows.append(row(name, param, graph, replicate))
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print("Wrote", output)


def task_results(outputs: Path, results_dir: Path, scenario: str, seed: int) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    master = outputs / "longitudinal_metrics_master.csv"
    if not master.is_file():
        raise FileNotFoundError(f"Missing {master}; run --task longitudinal first.")
    shutil.copy2(master, results_dir / "longitudinal_metrics.csv")
    cal = results_dir / "calibration.csv"
    if not cal.is_file():
        alt = outputs / "calibration_by_seed.csv"
        if alt.is_file():
            shutil.copy2(alt, cal)
    seed_dir = outputs / scenario / f"seed_{seed}"
    if not seed_dir.is_dir():
        raise FileNotFoundError(seed_dir)
    rider_nodes_from_exports(seed_dir).to_csv(results_dir / "rider_nodes.csv", index=False)
    for name in ("familiarity_edges.csv", "social_edges.csv"):
        shutil.copy2(seed_dir / name, results_dir / name)
    print("Wrote compact files under", results_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-simulation analysis tasks.")
    parser.add_argument(
        "--task",
        choices=["calibration", "longitudinal", "nulls", "comparisons", "results", "all"],
        default="all",
    )
    parser.add_argument("--data-dir", type=Path, default=_PROJECT_ROOT / "data")
    parser.add_argument("--outputs", type=Path, default=_PROJECT_ROOT / "outputs")
    parser.add_argument("--results", type=Path, default=_PROJECT_ROOT / "results")
    parser.add_argument("--seed-dir", type=Path, default=_PROJECT_ROOT / "outputs" / "baseline" / "seed_6400")
    parser.add_argument("--scenario", default="baseline")
    parser.add_argument("--seed", type=int, default=6400)
    parser.add_argument("--base-seed", type=int, default=6400)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-replicates", type=int, default=10)
    args = parser.parse_args()

    tasks = (
        ["calibration", "longitudinal", "nulls", "comparisons", "results"]
        if args.task == "all"
        else [args.task]
    )
    for task in tasks:
        if task == "calibration":
            task_calibration(args.results, args.data_dir, args.base_seed, args.n_seeds)
        elif task == "longitudinal":
            task_longitudinal(args.outputs)
        elif task == "nulls":
            task_nulls(args.seed_dir, args.results / "null_models.csv", args.seed)
        elif task == "comparisons":
            task_comparisons(args.seed_dir, args.results / "model_comparisons.csv", args.seed, args.n_replicates)
        elif task == "results":
            task_results(args.outputs, args.results, args.scenario, args.seed)


if __name__ == "__main__":
    main()
