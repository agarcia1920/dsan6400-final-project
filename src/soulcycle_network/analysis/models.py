"""Null models, canonical graph generators, and scenario comparison helpers."""

from __future__ import annotations

from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd

from soulcycle_network.analysis.metrics import FAMILIARITY_LAYER, build_layer_graph, graph_metrics
from soulcycle_network.network_formation import NetworkState
from soulcycle_network.simulation import SimulationContext, SimulationResult


# --- Attendance shuffles ---


def null_shuffle_attendance_df(attendance: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Shuffle session IDs within each (week, day). Keeps one class per rider per day."""
    if attendance.empty:
        return attendance.copy()

    out = attendance.copy()
    for _, day_df in out.groupby(["week_number", "day_of_week"], sort=False):
        indices = day_df.index.to_list()
        riders = day_df["rider_id"].to_list()
        if len(set(riders)) != len(riders):
            raise ValueError("Duplicate rider-day rows; fix attendance before nulls.")

        session_slots = day_df["session_id"].to_list()
        rng.shuffle(session_slots)
        for idx, session_id in zip(indices, session_slots):
            out.at[idx, "session_id"] = session_id
    return out


def null_shuffle_attendance_market_df(attendance: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Same as global shuffle, but separately within each home market."""
    if attendance.empty:
        return attendance.copy()
    if "home_market" not in attendance.columns:
        raise ValueError("Merge home_market from node_attributes first.")

    out = attendance.copy()
    for _, day_df in out.groupby(["week_number", "day_of_week", "home_market"], sort=False):
        indices = day_df.index.to_list()
        riders = day_df["rider_id"].to_list()
        if len(set(riders)) != len(riders):
            raise ValueError("Duplicate rider-day rows in a market group.")
        session_slots = day_df["session_id"].to_list()
        rng.shuffle(session_slots)
        for idx, session_id in zip(indices, session_slots):
            out.at[idx, "session_id"] = session_id
    return out


def null_randomize_attendance(
    ctx: SimulationContext,
    result: SimulationResult,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for week in result.week_results:
        for record in week.booking.records:
            rider = ctx.riders[record.rider_id]
            rows.append(
                {
                    "week_number": record.week_number,
                    "session_id": record.slot_id,
                    "rider_id": record.rider_id,
                    "studio_id": record.studio_id,
                    "day_of_week": record.day_of_week,
                    "home_market": rider.home_market,
                    "home_cluster": rider.home_cluster,
                }
            )
    return null_shuffle_attendance_df(pd.DataFrame(rows), rng)


def coattendance_from_attendance(attendance: pd.DataFrame) -> NetworkState:
    from soulcycle_network.network_formation import empty_network, update_from_enrollments

    state = empty_network()
    if attendance.empty:
        return state

    enrollments: dict[str, list[str]] = {}
    for row in attendance.itertuples(index=False):
        week = getattr(row, "week_number", None)
        if week is not None:
            key = "W" + str(int(week)).zfill(2) + "_" + str(row.session_id)
        else:
            key = str(row.session_id)
        enrollments.setdefault(key, []).append(row.rider_id)

    for session_id, rider_ids in enrollments.items():
        unique = list(dict.fromkeys(rider_ids))
        if len(unique) != len(rider_ids):
            counts = Counter(rider_ids)
            duplicates = [r for r, c in counts.items() if c > 1]
            raise ValueError("Session " + str(session_id) + " has duplicate riders " + str(duplicates) + ".")
        enrollments[session_id] = unique

    update_from_enrollments(state, enrollments)
    return state


def compare_null_metrics(
    observed_state: NetworkState,
    null_state: NetworkState,
    rider_cluster: dict[str, str],
    rider_market: dict[str, str],
) -> pd.DataFrame:
    obs_graph = build_layer_graph(observed_state, FAMILIARITY_LAYER)
    null_graph = build_layer_graph(null_state, FAMILIARITY_LAYER)
    obs = graph_metrics(obs_graph, rider_cluster, rider_market)
    nul = graph_metrics(null_graph, rider_cluster, rider_market)
    return pd.DataFrame(
        [{"metric": key, "observed": obs[key], "null": nul[key]} for key in obs]
    )


def degree_preserving_rewire(graph: nx.Graph, *, swaps_per_edge: int = 10, seed: int = 6400) -> nx.Graph:
    rewired = graph.copy()
    edge_count = rewired.number_of_edges()
    if edge_count < 2:
        return rewired
    try:
        nx.double_edge_swap(
            rewired,
            nswap=swaps_per_edge * edge_count,
            max_tries=100 * edge_count,
            seed=seed,
        )
    except nx.NetworkXAlgorithmError:
        pass
    return rewired


def null_degree_preserving_rewire(
    graph: nx.Graph,
    rng: np.random.Generator,
    n_swaps: int | None = None,
) -> nx.Graph:
    if graph.number_of_edges() < 2:
        return graph.copy()
    rewired = graph.copy()
    swaps = n_swaps if n_swaps is not None else rewired.number_of_edges() * 5
    try:
        nx.double_edge_swap(rewired, nswap=swaps, max_tries=swaps * 10)
    except nx.NetworkXAlgorithmError:
        pass
    return rewired


# --- Canonical comparison graphs ---


def matched_random_graph(*, n: int, m: int, seed: int) -> nx.Graph:
    probability = (2 * m / (n * (n - 1))) if n > 1 else 0.0
    return nx.gnp_random_graph(n=n, p=probability, seed=seed)


def nonlinear_preferential_attachment(*, n: int, m_links: int, alpha: float, seed: int) -> nx.Graph:
    if n < m_links + 1:
        raise ValueError("n must exceed the initial core size.")
    if m_links < 1:
        raise ValueError("m_links must be positive.")

    rng = np.random.default_rng(seed)
    graph = nx.complete_graph(m_links + 1)

    for new_node in range(m_links + 1, n):
        existing_nodes = np.array(list(graph.nodes()), dtype=int)
        degrees = np.array([graph.degree(int(node)) for node in existing_nodes], dtype=float)
        weights = np.power(np.maximum(degrees, 1.0), alpha)
        probabilities = weights / weights.sum()
        targets = rng.choice(
            existing_nodes,
            size=min(m_links, len(existing_nodes)),
            replace=False,
            p=probabilities,
        )
        graph.add_node(new_node)
        graph.add_edges_from((new_node, int(target)) for target in targets)
    return graph


def attractiveness_model(*, n: int, m_links: int, attractiveness: float, seed: int) -> nx.Graph:
    if attractiveness < 0:
        raise ValueError("attractiveness must be nonnegative.")

    rng = np.random.default_rng(seed)
    graph = nx.complete_graph(m_links + 1)

    for new_node in range(m_links + 1, n):
        existing = np.array(list(graph.nodes()), dtype=int)
        weights = np.array([graph.degree(int(node)) + attractiveness for node in existing], dtype=float)
        probabilities = weights / weights.sum()
        targets = rng.choice(existing, size=min(m_links, len(existing)), replace=False, p=probabilities)
        graph.add_node(new_node)
        graph.add_edges_from((new_node, int(target)) for target in targets)
    return graph


def fitness_model(*, n: int, m_links: int, seed: int) -> tuple[nx.Graph, dict[int, float]]:
    rng = np.random.default_rng(seed)
    graph = nx.complete_graph(m_links + 1)
    fitness: dict[int, float] = {int(node): float(rng.uniform(0.1, 1.0)) for node in graph.nodes()}

    for new_node in range(m_links + 1, n):
        fitness[new_node] = float(rng.uniform(0.1, 1.0))
        existing = np.array(list(graph.nodes()), dtype=int)
        weights = np.array(
            [fitness[int(node)] * max(graph.degree(int(node)), 1) for node in existing],
            dtype=float,
        )
        probabilities = weights / weights.sum()
        targets = rng.choice(existing, size=min(m_links, len(existing)), replace=False, p=probabilities)
        graph.add_node(new_node)
        graph.add_edges_from((new_node, int(target)) for target in targets)

    nx.set_node_attributes(graph, {k: v for k, v in fitness.items()}, "fitness")
    return graph, fitness


def degree_ks_distance(graph_a: nx.Graph, graph_b: nx.Graph) -> float:
    from scipy.stats import ks_2samp

    degree_a = [degree for _, degree in graph_a.degree()]
    degree_b = [degree for _, degree in graph_b.degree()]
    if not degree_a or not degree_b:
        return float("nan")
    return float(ks_2samp(degree_a, degree_b).statistic)


def paired_scenario_comparison(results: pd.DataFrame, *, metric: str, value_col: str = "edges") -> pd.DataFrame:
    """Paired deltas across coordination scenarios (same seed)."""
    if "metric" in results.columns and "value" in results.columns:
        metric_df = results.loc[results["metric"] == metric, ["seed", "scenario", "value"]]
    else:
        metric_df = results.loc[:, ["seed", "scenario", value_col]].rename(columns={value_col: "value"})

    wide = metric_df.pivot(index="seed", columns="scenario", values="value").reset_index()
    if "baseline" in wide.columns and "no_coordination" in wide.columns:
        wide["baseline_minus_none"] = wide["baseline"] - wide["no_coordination"]
    if "high_coordination" in wide.columns and "no_coordination" in wide.columns:
        wide["high_minus_none"] = wide["high_coordination"] - wide["no_coordination"]
    if "no_coordination" in wide.columns:
        denominator = wide["no_coordination"].replace(0, pd.NA)
        if "baseline_minus_none" in wide.columns:
            wide["baseline_percent_change"] = 100 * wide["baseline_minus_none"] / denominator
        if "high_minus_none" in wide.columns:
            wide["high_percent_change"] = 100 * wide["high_minus_none"] / denominator
    return wide
