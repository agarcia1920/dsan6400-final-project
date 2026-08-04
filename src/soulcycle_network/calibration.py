# Multi-seed calibration helpers for analysis notebooks and scripts.

from __future__ import annotations

import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from soulcycle_network.config import MIN_CLASSES_FOR_FAMILIARITY, RANDOM_SEED, TOTAL_SIMULATED_RIDERS, TOTAL_WEEKS
from soulcycle_network.network_formation import NetworkState, social_tie_pairs, to_graph
from soulcycle_network.simulation import SimulationContext, SimulationResult, run_default_simulation

CALIBRATION_METRICS: tuple[str, ...] = (
    "generated_mean_annual_ride_rate",
    "implied_population",
    "simulation_scale",
    "seat_occupancy_rate",
    "avg_attendance_per_week",
    "total_unmet_demand",
    "pair_count",
    "familiarity_pair_count",
    "social_tie_pair_count",
    "familiarity_mean_degree",
    "social_tie_mean_degree",
    "familiarity_largest_connected_component",
    "social_tie_largest_connected_component",
    "total_coordinated_bookings",
    "runtime_seconds",
)


def _graph_mean_degree(graph: nx.Graph) -> float:
    degrees = [deg for _, deg in graph.degree()]
    return float(sum(degrees) / len(degrees)) if degrees else 0.0


def _graph_largest_component(graph: nx.Graph) -> float:
    sizes = sorted((len(c) for c in nx.connected_components(graph)), reverse=True)
    return float(sizes[0]) if sizes else 0.0


def network_layer_stats(state: NetworkState) -> dict[str, float]:
    familiarity_graph = to_graph(state, MIN_CLASSES_FOR_FAMILIARITY)
    social_graph = nx.Graph()
    for a, b in social_tie_pairs(state):
        social_graph.add_edge(a, b)

    return {
        "familiarity_mean_degree": _graph_mean_degree(familiarity_graph),
        "social_tie_mean_degree": _graph_mean_degree(social_graph),
        "familiarity_largest_connected_component": _graph_largest_component(familiarity_graph),
        "social_tie_largest_connected_component": _graph_largest_component(social_graph),
    }


def metrics_from_simulation(
    ctx: SimulationContext,
    _result: SimulationResult,
    summary: dict[str, float],
    runtime_seconds: float,
    seed: int,
) -> dict[str, float | int]:
    row: dict[str, float | int] = {"seed": seed}
    row.update(network_layer_stats(_result.network_state))
    for key in CALIBRATION_METRICS:
        if key == "runtime_seconds":
            row[key] = float(runtime_seconds)
            continue
        if key in row:
            continue
        if key not in summary:
            raise KeyError("Missing metric " + key + " for seed " + str(seed) + ".")
        row[key] = summary[key]
    return row


def run_calibration_seed(
    data_dir: str | Path,
    seed: int,
    n_weeks: int = TOTAL_WEEKS,
    n_riders: int = TOTAL_SIMULATED_RIDERS,
) -> dict[str, float | int]:
    started = time.perf_counter()
    ctx, result, summary = run_default_simulation(
        data_dir,
        seed=seed,
        n_weeks=n_weeks,
        n_riders=n_riders,
    )
    runtime = time.perf_counter() - started
    return metrics_from_simulation(ctx, result, summary, runtime, seed)


def default_calibration_seeds(base_seed: int = RANDOM_SEED, n_seeds: int = 10) -> list[int]:
    return [base_seed + offset for offset in range(n_seeds)]


def run_calibration_batch(
    data_dir: str | Path,
    seeds: list[int] | None = None,
    n_weeks: int = TOTAL_WEEKS,
    n_riders: int = TOTAL_SIMULATED_RIDERS,
) -> pd.DataFrame:
    seed_list = default_calibration_seeds() if seeds is None else list(seeds)
    rows = [
        run_calibration_seed(data_dir, seed, n_weeks=n_weeks, n_riders=n_riders)
        for seed in seed_list
    ]
    return pd.DataFrame(rows)


def summarize_calibration_batch(results: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in CALIBRATION_METRICS if c in results.columns]
    stats = pd.DataFrame(
        {
            "mean": results[numeric_cols].mean(numeric_only=True),
            "std": results[numeric_cols].std(numeric_only=True, ddof=0),
            "min": results[numeric_cols].min(numeric_only=True),
            "max": results[numeric_cols].max(numeric_only=True),
        }
    )
    stats["cv"] = stats["std"] / stats["mean"].replace(0, np.nan)
    return stats
