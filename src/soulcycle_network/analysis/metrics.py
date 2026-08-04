# Graph summaries and longitudinal checkpoint metrics from network state or replayed attendance.

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import networkx as nx

import numpy as np
import pandas as pd

from soulcycle_network.config import (
    MIN_ACTIVE_TIE_STRENGTH_FOR_SOCIAL_TIE,
    MIN_CLASSES_FOR_FAMILIARITY,
    MIN_CLASSES_FOR_SOCIAL_TIE,
)
from soulcycle_network.network_formation import (
    NetworkState,
    familiarity_pairs,
    social_tie_pairs,
    to_graph,
)

ANALYSIS_SNAPSHOT_WEEKS: tuple[int, ...] = (1, 4, 8, 13, 26, 39, 52)
LINKING_STAGE_WEEKS: tuple[int, ...] = (13, 26, 52)
ACTIVITY_TIER_LABELS = ("low_frequency", "moderate_frequency", "high_frequency")


def isolate_gcc(graph: nx.Graph) -> nx.Graph:
    if graph.is_directed():
        raise TypeError("Expected an undirected graph.")
    if graph.number_of_nodes() == 0:
        return graph.copy()
    largest_nodes = max(nx.connected_components(graph), key=len)
    return graph.subgraph(largest_nodes).copy()


def build_rider_graph(
    rider_ids,
    pair_records,
    *,
    graph_type: str,
    familiarity_threshold: int = MIN_CLASSES_FOR_FAMILIARITY,
    social_threshold: int = MIN_CLASSES_FOR_SOCIAL_TIE,
    active_strength_threshold: float = MIN_ACTIVE_TIE_STRENGTH_FOR_SOCIAL_TIE,
    include_isolates: bool = True,
) -> nx.Graph:
    valid_types = {"coattendance", "familiarity", "active_social"}
    if graph_type not in valid_types:
        raise ValueError("graph_type must be one of " + str(sorted(valid_types)) + ".")

    graph = nx.Graph(graph_type=graph_type)
    if include_isolates:
        graph.add_nodes_from(rider_ids)

    for record in pair_records:
        rider_a = record["rider_1"]
        rider_b = record["rider_2"]
        co_count = int(record["coattendance_count"])
        strength = float(record["tie_strength"])
        include_edge = False
        if graph_type == "coattendance":
            include_edge = co_count >= 1
        elif graph_type == "familiarity":
            include_edge = co_count >= familiarity_threshold
        elif graph_type == "active_social":
            include_edge = co_count >= social_threshold and strength >= active_strength_threshold
        if include_edge:
            graph.add_edge(rider_a, rider_b, coattendance_count=co_count, tie_strength=strength)
    return graph


def categorical_assortativity(
    graph: nx.Graph,
    node_attributes: pd.DataFrame,
    *,
    attribute_column: str,
) -> float | None:
    attribute_map = node_attributes.set_index("rider_id")[attribute_column].astype(str).to_dict()
    labeled = graph.copy()
    nx.set_node_attributes(labeled, attribute_map, attribute_column)
    if labeled.number_of_edges() == 0:
        return None
    return float(nx.attribute_assortativity_coefficient(labeled, attribute_column))


def _baseline_rate_column(nodes: pd.DataFrame) -> str:
    if "baseline_annual_ride_rate" in nodes.columns:
        return "baseline_annual_ride_rate"
    if "annual_ride_rate" in nodes.columns:
        return "annual_ride_rate"
    raise ValueError("node table needs baseline_annual_ride_rate or annual_ride_rate.")


def assign_activity_frequency_tier(nodes: pd.DataFrame) -> pd.DataFrame:
    rate_col = _baseline_rate_column(nodes)
    rates = nodes[rate_col].astype(float)
    q33, q67 = float(rates.quantile(1 / 3)), float(rates.quantile(2 / 3))

    def tier(value: float) -> str:
        if value <= q33:
            return ACTIVITY_TIER_LABELS[0]
        if value <= q67:
            return ACTIVITY_TIER_LABELS[1]
        return ACTIVITY_TIER_LABELS[2]

    out = nodes.copy()
    out["activity_frequency_tier"] = rates.map(tier)
    out["activity_tertile_q33"] = q33
    out["activity_tertile_q67"] = q67
    return out


def _mean_abs_rate_diff_on_edges(graph: nx.Graph, rates: dict[str, float]) -> float | None:
    if graph.number_of_edges() == 0:
        return None
    diffs = [abs(rates.get(a, np.nan) - rates.get(b, np.nan)) for a, b in graph.edges()]
    diffs = [d for d in diffs if np.isfinite(d)]
    return float(np.mean(diffs)) if diffs else None


def _mean_degree_by_activity_tier(graph: nx.Graph, tier_map: dict[str, str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for label in ACTIVITY_TIER_LABELS:
        riders = [r for r, t in tier_map.items() if t == label and r in graph]
        if not riders:
            out["mean_degree_" + label] = None
            continue
        out["mean_degree_" + label] = float(sum(graph.degree(r) for r in riders) / len(riders))
    return out


def _coordinated_social_dyads(
    attendance: pd.DataFrame,
    state: NetworkState,
    through_week: int,
) -> int:
    att = attendance[attendance["week_number"] <= through_week]
    if att.empty:
        return 0
    coord = att[att["coordinated_booking"].astype(str).str.lower().isin({"true", "1", "yes"})]
    if coord.empty:
        return 0
    social = social_tie_pairs(state)
    count = 0
    for _, session_df in coord.groupby(["week_number", "session_id"], sort=False):
        riders = session_df["rider_id"].astype(str).tolist()
        for i in range(len(riders)):
            for j in range(i + 1, len(riders)):
                key = tuple(sorted((riders[i], riders[j])))
                if key in social:
                    count += 1
    return count


def dyad_linking_and_latent_counts(
    state: NetworkState,
    *,
    attendance: pd.DataFrame | None,
    through_week: int,
    rider_cluster: dict[str, str] | None = None,
) -> dict[str, int | float | None]:
    pairs = list(state.co_counts.items())
    n_co = sum(1 for _, c in pairs if c >= 1)
    social = social_tie_pairs(state)
    familiar = familiarity_pairs(state)
    share = (lambda n: float(n / n_co) if n_co else None)

    n_familiar = sum(1 for _, c in pairs if c >= MIN_CLASSES_FOR_FAMILIARITY)
    n_social = len(social)
    n_coord = _coordinated_social_dyads(attendance, state, through_week) if attendance is not None else None

    one_from_familiar = sum(1 for _, c in pairs if c == 2)
    one_from_social = sum(1 for key, c in pairs if c == MIN_CLASSES_FOR_SOCIAL_TIE - 1 and key not in social) + sum(
        1 for key, c in pairs if c == 5 and key not in social
    )
    repeated_not_social = sum(1 for key, c in pairs if c >= 2 and key not in social)
    riders_co = {r for pair in state.co_counts for r in pair}
    riders_social = {r for pair in social for r in pair}
    cross_cluster_familiar = sum(
        1 for a, b in familiar if rider_cluster and rider_cluster.get(a) != rider_cluster.get(b)
    )

    return {
        "coattending_dyads": n_co,
        "dyads_exactly_two_shared_classes": sum(1 for _, c in pairs if c == 2),
        "familiar_dyads": n_familiar,
        "near_social_dyads_four_or_five": sum(1 for _, c in pairs if c in (4, 5)),
        "active_social_dyads": n_social,
        "social_dyads_with_coordinated_class": n_coord,
        "share_familiar_of_coattending": share(n_familiar),
        "share_social_of_coattending": share(n_social),
        "latent_pairs_one_encounter_from_familiarity": one_from_familiar,
        "latent_familiar_one_encounter_from_social": one_from_social,
        "latent_repeated_co_not_social_dyads": repeated_not_social,
        "latent_riders_with_co_but_no_social_tie": len(riders_co - riders_social),
        "latent_cross_cluster_familiarity_ties": cross_cluster_familiar,
    }


def edge_share_by_attribute(
    graph: nx.Graph,
    node_attributes: pd.DataFrame,
    *,
    attribute_column: str,
) -> dict[str, float]:
    attr = node_attributes.set_index("rider_id")[attribute_column].astype(str).to_dict()
    if graph.number_of_edges() == 0:
        return {"same_attribute_edge_share": 0.0, "cross_attribute_edge_share": 0.0}
    same = sum(1 for a, b in graph.edges() if attr.get(a) == attr.get(b))
    total = graph.number_of_edges()
    return {
        "same_attribute_edge_share": float(same / total),
        "cross_attribute_edge_share": float((total - same) / total),
    }


def centrality_table(
    graph: nx.Graph,
    *,
    use_gcc: bool = True,
    betweenness_k: int | None = None,
    seed: int = 6400,
) -> pd.DataFrame:
    analysis_graph = isolate_gcc(graph) if use_gcc else graph.copy()
    if analysis_graph.number_of_nodes() == 0:
        return pd.DataFrame(
            columns=[
                "rider_id",
                "degree",
                "degree_centrality",
                "betweenness_centrality",
                "closeness_centrality",
            ]
        )
    degree = dict(analysis_graph.degree())
    degree_centrality = nx.degree_centrality(analysis_graph)
    betweenness = nx.betweenness_centrality(
        analysis_graph, k=betweenness_k, seed=seed, normalized=True
    )
    closeness = nx.closeness_centrality(analysis_graph)
    nodes = list(analysis_graph.nodes())
    return pd.DataFrame(
        {
            "rider_id": nodes,
            "degree": [degree[node] for node in nodes],
            "degree_centrality": [degree_centrality[node] for node in nodes],
            "betweenness_centrality": [betweenness[node] for node in nodes],
            "closeness_centrality": [closeness[node] for node in nodes],
        }
    )


def detect_louvain_communities(graph: nx.Graph, *, seed: int = 6400, weight: str | None = "tie_strength"):
    communities = nx.community.louvain_communities(graph, weight=weight, seed=seed)
    node_to_community = {
        node: community_id
        for community_id, community in enumerate(communities)
        for node in community
    }
    modularity = nx.community.modularity(graph, communities, weight=weight)
    return communities, node_to_community, float(modularity)


@dataclass(frozen=True)
class NetworkSummary:
    nodes: int
    edges: int
    connected_nodes: int
    isolates: int
    components: int
    density_full_population: float
    average_degree_full_population: float
    mean_degree_population: float
    mean_degree_connected_subgraph: float
    average_clustering: float
    transitivity: float
    degree_assortativity: float | None
    gcc_nodes: int
    gcc_edges: int
    gcc_share_of_connected_nodes: float
    gcc_diameter: int | None
    gcc_radius: int | None
    gcc_average_shortest_path: float | None
    gcc_clustering: float
    gcc_transitivity: float


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def summarize_graph(graph: nx.Graph) -> dict[str, int | float | None]:
    if graph.is_directed():
        raise TypeError("This summary currently expects an undirected graph.")

    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    isolate_count = nx.number_of_isolates(graph) if n > 0 else 0
    connected_nodes = n - isolate_count
    component_count = nx.number_connected_components(graph) if n > 0 else 0

    average_degree = (2 * m / n) if n > 0 else 0.0
    edge_nodes = [node for node, deg in graph.degree() if deg > 0]
    subgraph = graph.subgraph(edge_nodes).copy() if edge_nodes else graph.copy()
    mean_degree_connected = (2 * subgraph.number_of_edges() / subgraph.number_of_nodes()) if subgraph.number_of_nodes() > 0 else 0.0

    clustering = nx.average_clustering(graph) if n > 0 else 0.0
    transitivity = nx.transitivity(graph) if n > 0 else 0.0
    assortativity = _finite_or_none(nx.degree_assortativity_coefficient(graph)) if m > 0 else None

    gcc = isolate_gcc(graph)
    gcc_n = gcc.number_of_nodes()
    if gcc_n > 1:
        gcc_diameter = nx.diameter(gcc)
        gcc_radius = nx.radius(gcc)
        gcc_path = nx.average_shortest_path_length(gcc)
    elif gcc_n == 1:
        gcc_diameter = 0
        gcc_radius = 0
        gcc_path = 0.0
    else:
        gcc_diameter = None
        gcc_radius = None
        gcc_path = None

    gcc_clustering = nx.average_clustering(gcc) if gcc_n > 0 else 0.0
    gcc_transitivity = nx.transitivity(gcc) if gcc_n > 0 else 0.0

    summary = NetworkSummary(
        nodes=n,
        edges=m,
        connected_nodes=connected_nodes,
        isolates=isolate_count,
        components=component_count,
        density_full_population=nx.density(graph) if n > 1 else 0.0,
        average_degree_full_population=average_degree,
        mean_degree_population=average_degree,
        mean_degree_connected_subgraph=mean_degree_connected,
        average_clustering=clustering,
        transitivity=transitivity,
        degree_assortativity=assortativity,
        gcc_nodes=gcc_n,
        gcc_edges=gcc.number_of_edges(),
        gcc_share_of_connected_nodes=(gcc_n / connected_nodes if connected_nodes > 0 else 0.0),
        gcc_diameter=gcc_diameter,
        gcc_radius=gcc_radius,
        gcc_average_shortest_path=gcc_path,
        gcc_clustering=gcc_clustering,
        gcc_transitivity=gcc_transitivity,
    )
    return asdict(summary)


@dataclass(frozen=True)
class NetworkLayer:
    name: str
    min_co_count: int | None = None
    use_active_social: bool = False


FAMILIARITY_LAYER = NetworkLayer("familiarity", min_co_count=MIN_CLASSES_FOR_FAMILIARITY)
SOCIAL_LAYER = NetworkLayer("social", use_active_social=True)


def build_layer_graph(state: NetworkState, layer: NetworkLayer) -> nx.Graph:
    if layer.use_active_social:
        graph = nx.Graph()
        for a, b in social_tie_pairs(state):
            graph.add_edge(a, b, tie_strength=state.tie_strength.get((a, b), 0.0))
        return graph
    return to_graph(state, layer.min_co_count or MIN_CLASSES_FOR_FAMILIARITY)


def _same_cluster_edge_share(graph: nx.Graph, rider_cluster: dict[str, str]) -> float:
    if graph.number_of_edges() == 0:
        return 0.0
    same = sum(1 for a, b in graph.edges() if rider_cluster.get(a) == rider_cluster.get(b))
    return float(same / graph.number_of_edges())


def graph_metrics(
    graph: nx.Graph,
    rider_cluster: dict[str, str],
    rider_market: dict[str, str],
    louvain_seed: int = 6400,
) -> dict[str, float]:
    nodes = graph.number_of_nodes()
    edges = graph.number_of_edges()
    if nodes == 0:
        return {
            "nodes": 0.0,
            "edges": 0.0,
            "density": 0.0,
            "mean_degree": 0.0,
            "median_degree": 0.0,
            "clustering": 0.0,
            "components": 0.0,
            "largest_component": 0.0,
            "modularity": 0.0,
            "assortativity_cluster": float("nan"),
            "assortativity_market": float("nan"),
            "same_cluster_edge_share": 0.0,
            "isolates": 0.0,
        }

    degrees = [deg for _, deg in graph.degree()]
    components = sorted((len(c) for c in nx.connected_components(graph)), reverse=True)
    attr_graph = graph.copy()
    for node in attr_graph.nodes():
        attr_graph.nodes[node]["cluster"] = rider_cluster.get(node, "")
        attr_graph.nodes[node]["market"] = rider_market.get(node, "")

    modularity = 0.0
    try:
        communities = nx.community.louvain_communities(attr_graph, weight=None, seed=louvain_seed)
        modularity = float(nx.community.modularity(attr_graph, communities))
    except Exception:
        modularity = float("nan")

    return {
        "nodes": float(nodes),
        "edges": float(edges),
        "density": float(nx.density(graph)),
        "mean_degree": float(sum(degrees) / len(degrees)),
        "median_degree": float(np.median(degrees)),
        "clustering": float(nx.average_clustering(graph)),
        "components": float(len(components)),
        "largest_component": float(components[0]),
        "modularity": modularity,
        "assortativity_cluster": float(nx.attribute_assortativity_coefficient(attr_graph, "cluster")),
        "assortativity_market": float(nx.attribute_assortativity_coefficient(attr_graph, "market")),
        "same_cluster_edge_share": _same_cluster_edge_share(graph, rider_cluster),
        "isolates": float(sum(1 for d in degrees if d == 0)),
    }


def metrics_for_state(
    state: NetworkState,
    rider_cluster: dict[str, str],
    rider_market: dict[str, str],
    louvain_seed: int = 6400,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for layer in (FAMILIARITY_LAYER, SOCIAL_LAYER):
        graph = build_layer_graph(state, layer)
        out[layer.name] = graph_metrics(graph, rider_cluster, rider_market, louvain_seed=louvain_seed)
    return out


def snapshot_metrics_rows(
    state: NetworkState,
    *,
    scenario: str,
    seed: int,
    week: int,
    rider_cluster: dict[str, str],
    rider_market: dict[str, str],
    tier_map: dict[str, str] | None = None,
    baseline_rates: dict[str, float] | None = None,
    attendance: pd.DataFrame | None = None,
) -> list[dict[str, object]]:
    node_attributes = pd.DataFrame(
        {"rider_id": list(rider_cluster.keys()), "home_cluster": list(rider_cluster.values()), "home_market": [rider_market[rid] for rid in rider_cluster]}
    )
    if tier_map:
        node_attributes["activity_frequency_tier"] = node_attributes["rider_id"].map(tier_map)

    linking: dict[str, int | float | None] = {}
    if week in LINKING_STAGE_WEEKS:
        linking = dyad_linking_and_latent_counts(state, attendance=attendance, through_week=week, rider_cluster=rider_cluster)

    rows: list[dict[str, object]] = []
    for network_type, layer in (("familiarity", FAMILIARITY_LAYER), ("social", SOCIAL_LAYER)):
        graph = build_layer_graph(state, layer)
        metrics = summarize_graph(graph)
        row: dict[str, object] = {
            "scenario": scenario,
            "seed": seed,
            "week": week,
            "network_type": network_type,
            **metrics,
        }
        if graph.number_of_edges() > 0:
            row["market_assortativity"] = categorical_assortativity(graph, node_attributes, attribute_column="home_market")
            row["cluster_assortativity"] = categorical_assortativity(graph, node_attributes, attribute_column="home_cluster")
            shares = edge_share_by_attribute(graph, node_attributes, attribute_column="home_cluster")
            row["same_cluster_edge_share"] = shares["same_attribute_edge_share"]
            if tier_map is not None:
                row["activity_tier_assortativity"] = categorical_assortativity(
                    graph, node_attributes, attribute_column="activity_frequency_tier"
                )
                if baseline_rates:
                    row["mean_abs_baseline_rate_diff_on_edges"] = _mean_abs_rate_diff_on_edges(graph, baseline_rates)
                row.update(_mean_degree_by_activity_tier(graph, tier_map))
        if network_type == "familiarity" and linking:
            row.update(linking)
        rows.append(row)
    return rows
