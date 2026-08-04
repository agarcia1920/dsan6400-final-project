# Co-attendance counts, tie strength, decay, and familiar vs social pair definitions.

from dataclasses import dataclass, field
import networkx as nx
from soulcycle_network.config import MIN_ACTIVE_TIE_STRENGTH_FOR_SOCIAL_TIE, MIN_CLASSES_FOR_FAMILIARITY, MIN_CLASSES_FOR_SOCIAL_TIE, TIE_DECAY_RATE
from soulcycle_network.riders import coerce_float

@dataclass
class NetworkState:
    #track co-attendance, tie strength (and tie decay)
    co_counts: dict[tuple[str, str], int] = field(default_factory=dict)
    tie_strength: dict[tuple[str, str], float] = field(default_factory=dict)

def pair_key(rider_a: str, rider_b: str) -> tuple[str, str]:
    #validate the input types and values
    if not isinstance(rider_a, str):
        raise TypeError("rider_a must be a string.")
    if not isinstance(rider_b, str):
        raise TypeError("rider_b must be a string.")

    #strip whitespace
    rider_a = rider_a.strip()
    rider_b = rider_b.strip()
    if not rider_a or not rider_b:
        raise ValueError("rider ids cannot be empty.")
    if rider_a == rider_b:
        raise ValueError("pair_key requires two distinct rider ids.")

    #sort the rider ids
    if rider_a < rider_b:
        return (rider_a, rider_b)
    return (rider_b, rider_a)

def empty_network() -> NetworkState:
    #create an empty network state
    return NetworkState()

def decay_ties(state: NetworkState, rate: float = TIE_DECAY_RATE) -> None:
    #validate the input types and values
    if not isinstance(state, NetworkState):
        raise TypeError("state must be a NetworkState.")
    rate = coerce_float(rate, "rate")
    if rate < 0 or rate > 1:
        raise ValueError("rate must be between 0 and 1.")

    stale: list[tuple[str, str]] = [] #list to store the stale ties
    for key, strength in state.tie_strength.items():
        new_strength = strength * rate #calculate the new strength
        if new_strength < 1e-9:
            stale.append(key) #add the key to the stale list
        else:
            state.tie_strength[key] = new_strength

    for key in stale:
        del state.tie_strength[key] #delete the key from the tie strength dictionary

def update_from_enrollments(state: NetworkState, enrollments: dict[str, list[str]]) -> None:
    #validate the input types and values
    if not isinstance(state, NetworkState):
        raise TypeError("state must be a NetworkState.")
    if not isinstance(enrollments, dict):
        raise TypeError("enrollments must be a dictionary.")

    for rider_ids in enrollments.values(): #iterate over the rider ids
        if not isinstance(rider_ids, list):
            raise TypeError("enrollment values must be lists.")
        n = len(rider_ids) #get the number of rider ids
        if n < 2:
            continue

        for i in range(n): #iterate over the rider ids
            for j in range(i + 1, n):
                key = pair_key(rider_ids[i], rider_ids[j])
                state.co_counts[key] = state.co_counts.get(key, 0) + 1 #increment the co-attendance count
                state.tie_strength[key] = state.tie_strength.get(key, 0.0) + 1.0

def familiarity_pairs(state: NetworkState) -> set[tuple[str, str]]:
    #validate the input types and values
    if not isinstance(state, NetworkState):
        raise TypeError("state must be a NetworkState.")

    out: set[tuple[str, str]] = set() #set to store the familiarity pairs
    for key, count in state.co_counts.items():
        if count >= MIN_CLASSES_FOR_FAMILIARITY:
            out.add(key) #add the key to the set
    return out

def social_tie_pairs(state: NetworkState) -> set[tuple[str, str]]:
    #validate the input types and values
    if not isinstance(state, NetworkState):
        raise TypeError("state must be a NetworkState.")

    out: set[tuple[str, str]] = set() #set to store the social tie pairs
    for key, count in state.co_counts.items():
        if count >= MIN_CLASSES_FOR_SOCIAL_TIE and state.tie_strength.get(key, 0.0) >= MIN_ACTIVE_TIE_STRENGTH_FOR_SOCIAL_TIE:
            out.add(key) #add the key to the set
    return out

def partners_for_rider(state: NetworkState, rider_id: str) -> list[tuple[str, float]]:
    #validate the input types and values
    if not isinstance(state, NetworkState):
        raise TypeError("state must be a NetworkState.")
    if not isinstance(rider_id, str):
        raise TypeError("rider_id must be a string.")

    rider_id = rider_id.strip() #strip whitespace
    if not rider_id:
        raise ValueError("rider_id cannot be empty.")

    social = social_tie_pairs(state) #get the social tie pairs
    partners: list[tuple[str, float]] = []

    for a, b in social: #iterate over the social tie pairs
        if a == rider_id:
            partners.append((b, state.tie_strength.get((a, b), 0.0))) #add the partner to the list
        elif b == rider_id:
            partners.append((a, state.tie_strength.get((a, b), 0.0))) #add the partner to the list  

    partners.sort(key=lambda x: (-x[1], x[0])) #sort the partners by strength and then by id    
    return partners #return the partners

def to_graph(state: NetworkState, min_co_count: int = MIN_CLASSES_FOR_FAMILIARITY) -> nx.Graph:
    #validate the input types and values
    if not isinstance(state, NetworkState):
        raise TypeError("state must be a NetworkState.")
    if isinstance(min_co_count, bool) or not isinstance(min_co_count, int):
        raise TypeError("min_co_count must be an integer.")
    if min_co_count <= 0:
        raise ValueError("min_co_count must be positive.")

    graph = nx.Graph() #create a graph
    for key, count in state.co_counts.items():
        if count < min_co_count:
            continue #skip if the co-attendance count is less than the minimum co-attendance count
        a, b = key #get the pair
        graph.add_edge(a, b, co_count=count, tie_strength=state.tie_strength.get(key, 0.0)) #add the edge to the graph
    return graph #return the graph

def clone_network_state(state: NetworkState) -> NetworkState:
    if not isinstance(state, NetworkState):
        raise TypeError("state must be a NetworkState.")
    return NetworkState(
        co_counts=dict(state.co_counts),
        tie_strength=dict(state.tie_strength),
    )

def summarize_network(state: NetworkState) -> dict[str, float]:
    #validate the input types and values
    if not isinstance(state, NetworkState):
        raise TypeError("state must be a NetworkState.")

    fam = familiarity_pairs(state) #get the familiarity pairs
    social = social_tie_pairs(state) #get the social tie pairs
    graph = to_graph(state, MIN_CLASSES_FOR_FAMILIARITY)
    degrees = [deg for _, deg in graph.degree()]
    components = sorted((len(c) for c in nx.connected_components(graph)), reverse=True)

    return {
        "pair_count": float(len(state.co_counts)), #get the number of pairs
        "familiarity_pair_count": float(len(fam)), #get the number of familiarity pairs
        "social_tie_pair_count": float(len(social)), #get the number of social tie pairs
        "graph_nodes": float(graph.number_of_nodes()),
        "graph_edges": float(graph.number_of_edges()), #get the number of edges
        "mean_degree": float(sum(degrees) / len(degrees)) if degrees else 0.0,
        "largest_connected_component": float(components[0]) if components else 0.0,
        "mean_tie_strength": float(sum(state.tie_strength.values()) / len(state.tie_strength)) if state.tie_strength else 0.0,
    }
