# Chooses which social-tie partners attempt to ride together during a week.

from dataclasses import dataclass
import numpy as np
from soulcycle_network.config import MAX_COORDINATED_CLASSES_PER_WEEK, MAX_COORDINATION_PARTNERS_PER_WEEK
from soulcycle_network.network_formation import NetworkState, social_tie_pairs
from soulcycle_network.riders import Rider

@dataclass
class CoordinationPair: #dataclass to store the coordination pairs
    rider_a: str
    rider_b: str

def plan_coordination(
    riders: dict[str, Rider],
    network_state: NetworkState,
    weekly_counts: dict[str, int],
    rng: np.random.Generator,
    max_partners_per_week: int | None = None,
    max_coordinated_classes_per_week: int | None = None,
) -> list[CoordinationPair]: #function to plan the coordination
    if not isinstance(riders, dict):
        raise TypeError("riders must be a dictionary.")
    if not isinstance(network_state, NetworkState):
        raise TypeError("network_state must be a NetworkState.")
    if not isinstance(weekly_counts, dict):
        raise TypeError("weekly_counts must be a dictionary.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")

    partner_cap = MAX_COORDINATION_PARTNERS_PER_WEEK if max_partners_per_week is None else max_partners_per_week
    class_cap = MAX_COORDINATED_CLASSES_PER_WEEK if max_coordinated_classes_per_week is None else max_coordinated_classes_per_week
    if isinstance(partner_cap, bool) or not isinstance(partner_cap, int) or partner_cap < 0:
        raise ValueError("max_partners_per_week must be a non-negative integer.")
    if isinstance(class_cap, bool) or not isinstance(class_cap, int) or class_cap < 0:
        raise ValueError("max_coordinated_classes_per_week must be a non-negative integer.")
    if partner_cap == 0 or class_cap == 0:
        return []

    #get the active riders
    active = {rider_id for rider_id, n in weekly_counts.items() if isinstance(n, int) and not isinstance(n, bool) and n > 0}
    social = social_tie_pairs(network_state) #get the social tie pairs
    if not social or not active:
        return [] #return an empty list if there are no social tie pairs or active riders

    pair_scores: dict[tuple[str, str], float] = {} #dictionary to store the pair scores
    for a, b in social:
        if a not in active or b not in active:
            continue
        key = (a, b)
        pair_scores[key] = network_state.tie_strength.get(key, 0.0) + 1.0

    if not pair_scores:
        return []

    used_partners: dict[str, int] = {rid: 0 for rid in active} #dictionary to store the used partners
    planned: list[CoordinationPair] = [] #list to store the planned coordination pairs
    ordered = sorted(pair_scores.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))

    for (a, b), _ in ordered: #iterate over the ordered pair scores
        if used_partners.get(a, 0) >= partner_cap:
            continue #skip if the partner has already been used too many times
        if used_partners.get(b, 0) >= partner_cap:
            continue #skip if the partner has already been used too many times
        if weekly_counts.get(a, 0) <= 0 or weekly_counts.get(b, 0) <= 0:
            continue #skip if the rider has no classes this week

        planned.append(CoordinationPair(rider_a=a, rider_b=b))
        used_partners[a] = used_partners.get(a, 0) + 1 #increment the used partner count
        used_partners[b] = used_partners.get(b, 0) + 1 #increment the used partner count

        if len(planned) >= class_cap * len(active):
            break #break if the number of planned coordination pairs is greater than the maximum number of coordination pairs per week

    rng.shuffle(planned)
    return planned #return the planned coordination pairs

def coordination_count_by_rider(pairs: list[CoordinationPair]) -> dict[str, int]:
    if not isinstance(pairs, list): #validate the input types and values
        raise TypeError("pairs must be a list.")

    counts: dict[str, int] = {} #dictionary to store the coordination counts
    for pair in pairs:
        if not isinstance(pair, CoordinationPair):
            raise TypeError("pairs must contain CoordinationPair objects.")
        counts[pair.rider_a] = counts.get(pair.rider_a, 0) + 1
        counts[pair.rider_b] = counts.get(pair.rider_b, 0) + 1
    return counts
