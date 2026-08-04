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
from soulcycle_network.network_formation import NetworkState, social_tie_pairs, to_graph

ANALYSIS_SNAPSHOT_WEEKS: tuple[int, ...] = (1, 4, 8, 13, 26, 39, 52)


# --- Graph construction ---


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
    """Course-style summary: full graph + GCC path metrics."""
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
    mean_degree_connected = (
        (2 * subgraph.number_of_edges() / subgraph.number_of_nodes())
        if subgraph.number_of_nodes() > 0
        else 0.0
    )

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


# --- Layer graphs (used during simulation export and null comparisons) ---


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
) -> list[dict[str, object]]:
    import pandas as pd

    node_attributes = pd.DataFrame(
        {
            "rider_id": list(rider_cluster.keys()),
            "home_cluster": list(rider_cluster.values()),
            "home_market": [rider_market[rid] for rid in rider_cluster],
        }
    )
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
            row["market_assortativity"] = categorical_assortativity(
                graph, node_attributes, attribute_column="home_market"
            )
            row["cluster_assortativity"] = categorical_assortativity(
                graph, node_attributes, attribute_column="home_cluster"
            )
            shares = edge_share_by_attribute(graph, node_attributes, attribute_column="home_cluster")
            row["same_cluster_edge_share"] = shares["same_attribute_edge_share"]
        rows.append(row)
    return rows
