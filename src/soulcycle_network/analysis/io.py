# Loads exported seed folders and rebuilds longitudinal metrics from attendance.csv when needed.

from __future__ import annotations
from pathlib import Path

import networkx as nx
import pandas as pd

from soulcycle_network.analysis.metrics import (
    ANALYSIS_SNAPSHOT_WEEKS,
    assign_activity_frequency_tier,
    build_rider_graph,
    snapshot_metrics_rows,
)
from soulcycle_network.analysis.models import coattendance_from_attendance
from soulcycle_network.network_formation import decay_ties, empty_network


def load_seed_exports(seed_dir: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(seed_dir)
    names = [
        "simulation_summary",
        "weekly_summary",
        "attendance",
        "node_attributes",
        "pair_history",
        "familiarity_edges",
        "social_edges",
        "longitudinal_metrics",
    ]
    out: dict[str, pd.DataFrame] = {}
    for name in names:
        path = root / (name + ".csv")
        if path.exists():
            out[name] = pd.read_csv(path)
    return out


def rider_nodes_from_exports(seed_dir: str | Path) -> pd.DataFrame:
    exports = load_seed_exports(seed_dir)
    nodes = exports["node_attributes"].copy()
    pairs = exports["pair_history"]
    nodes["coattendance_degree"] = 0
    if "rider_id" in nodes.columns:
        deg: dict[str, int] = {}
        for row in pairs.itertuples(index=False):
            deg[row.rider_1] = deg.get(row.rider_1, 0) + 1
            deg[row.rider_2] = deg.get(row.rider_2, 0) + 1
        nodes["coattendance_degree"] = nodes["rider_id"].map(deg).fillna(0).astype(int)
    rename = {"annual_ride_rate": "baseline_annual_ride_rate"}
    for old, new in rename.items():
        if old in nodes.columns and new not in nodes.columns:
            nodes = nodes.rename(columns={old: new})
    if "baseline_annual_ride_rate" in nodes.columns or "annual_ride_rate" in nodes.columns:
        tiered = assign_activity_frequency_tier(nodes)
        nodes["activity_frequency_tier"] = tiered["activity_frequency_tier"]
    return nodes


def familiarity_graph_from_seed(seed_dir: str | Path, include_isolates: bool = True) -> nx.Graph:
    exports = load_seed_exports(seed_dir)
    riders = exports["node_attributes"]["rider_id"].astype(str).tolist()
    records = exports["pair_history"].to_dict(orient="records")
    return build_rider_graph(riders, records, graph_type="familiarity", include_isolates=include_isolates)


def load_master_table(output_dir: str | Path) -> pd.DataFrame:
    path = Path(output_dir) / "longitudinal_metrics_master.csv"
    if not path.exists():
        raise FileNotFoundError("Missing " + str(path) + ". Run scripts/run_experiment.py first.")
    return normalize_longitudinal_columns(pd.read_csv(path))


def load_project_master(output_dir: str | Path) -> pd.DataFrame:
    return load_master_table(output_dir)


def normalize_longitudinal_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "gcc_nodes" not in out.columns and "largest_component" in out.columns:
        out["gcc_nodes"] = out["largest_component"]
    if "average_clustering" not in out.columns and "clustering" in out.columns:
        out["average_clustering"] = out["clustering"]
    if "gcc_average_shortest_path" not in out.columns:
        out["gcc_average_shortest_path"] = pd.NA
    if "mean_degree_population" not in out.columns and "mean_degree" in out.columns:
        out["mean_degree_population"] = out["mean_degree"]
    return out


def rebuild_longitudinal_from_attendance(
    attendance: pd.DataFrame,
    node_attributes: pd.DataFrame,
    *,
    scenario: str,
    seed: int,
    n_weeks: int | None = None,
    checkpoint_weeks: tuple[int, ...] = ANALYSIS_SNAPSHOT_WEEKS,
) -> pd.DataFrame:
    if attendance.empty:
        return pd.DataFrame()

    cluster = node_attributes.set_index("rider_id")["home_cluster"].astype(str).to_dict()
    market = node_attributes.set_index("rider_id")["home_market"].astype(str).to_dict()
    tiered = assign_activity_frequency_tier(node_attributes)
    rate_col = "baseline_annual_ride_rate" if "baseline_annual_ride_rate" in tiered.columns else "annual_ride_rate"
    tier_map = tiered.set_index("rider_id")["activity_frequency_tier"].astype(str).to_dict()
    baseline_rates = tiered.set_index("rider_id")[rate_col].astype(float).to_dict()
    max_week = int(attendance["week_number"].max()) if n_weeks is None else n_weeks

    state = empty_network()
    rows: list[dict[str, object]] = []
    checkpoints = set(checkpoint_weeks)

    for week in range(1, max_week + 1):
        decay_ties(state)
        week_att = attendance[attendance["week_number"] == week]
        if not week_att.empty:
            week_state = coattendance_from_attendance(week_att)
            for key, count in week_state.co_counts.items():
                state.co_counts[key] = state.co_counts.get(key, 0) + count
            for key, strength in week_state.tie_strength.items():
                state.tie_strength[key] = state.tie_strength.get(key, 0.0) + strength

        if week in checkpoints:
            rows.extend(
                snapshot_metrics_rows(
                    state,
                    scenario=scenario,
                    seed=seed,
                    week=week,
                    rider_cluster=cluster,
                    rider_market=market,
                    tier_map=tier_map,
                    baseline_rates=baseline_rates,
                    attendance=attendance,
                )
            )

    return pd.DataFrame(rows)
