# Assignment helpers for placing synthetic instructors in clusters and studios.

import numpy as np
import pandas as pd
from soulcycle_network.instructor import Instructor
from soulcycle_network.studio import Studio

def largest_remainder_allocation(total: int, weights: pd.Series) -> dict[str, int]:
    if isinstance(total, bool) or not isinstance(total, int):
        raise TypeError("total must be an integer.")
    if total < 0:
        raise ValueError("total cannot be negative.")
    if not isinstance(weights, pd.Series):
        raise TypeError("weights must be a pandas Series.")
    if weights.empty:
        raise ValueError("weights cannot be empty.")

    numeric_weights = pd.to_numeric(weights, errors="raise").astype(float)
    if numeric_weights.isna().any():
        raise ValueError("weights cannot contain missing values.")
    if (numeric_weights < 0).any():
        raise ValueError("weights cannot contain negative values.")
    if numeric_weights.sum() <= 0:
        raise ValueError("weights must sum to a positive value.")

    exact_allocations = (numeric_weights / numeric_weights.sum()) * total
    integer_allocations = np.floor(exact_allocations).astype(int)
    remaining = total - int(integer_allocations.sum())
    remainders = (exact_allocations - integer_allocations).sort_values(ascending=False)

    for category in remainders.index[:remaining]:
        integer_allocations.loc[category] += 1

    if int(integer_allocations.sum()) != total:
        raise RuntimeError("Largest-remainder allocation did not preserve the total.")

    return integer_allocations.to_dict()

def allocate_home_clusters(market: str, instructor_count: int, studio_data: pd.DataFrame) -> dict[str, int]:
    #cluster weights use weekly bike supply, not slot counts
    if not isinstance(market, str):
        raise TypeError("market must be a string.")
    if isinstance(instructor_count, bool) or not isinstance(instructor_count, int):
        raise TypeError("instructor_count must be an integer.")
    if instructor_count <= 0:
        raise ValueError("instructor_count must be positive.")
    if not isinstance(studio_data, pd.DataFrame):
        raise TypeError("studio_data must be a pandas DataFrame.")

    market = market.strip()
    market_studios = studio_data.loc[studio_data["network_market"] == market].copy()
    if market_studios.empty:
        raise ValueError("No studios found for market '" + market + "'.")

    cluster_weights = market_studios.groupby("local_ridership_cluster")["weekly_bike_supply"].sum().sort_index()
    return largest_remainder_allocation(total=instructor_count, weights=cluster_weights)

def choose_regular_studios(market: str, home_cluster: str, requested_studio_count: int, studio_data: pd.DataFrame, rng: np.random.Generator) -> list[str]:
    if not isinstance(market, str):
        raise TypeError("market must be a string.")
    if not isinstance(home_cluster, str):
        raise TypeError("home_cluster must be a string.")
    if isinstance(requested_studio_count, bool) or not isinstance(requested_studio_count, int):
        raise TypeError("requested_studio_count must be an integer.")
    if requested_studio_count <= 0:
        raise ValueError("requested_studio_count must be positive.")
    if not isinstance(studio_data, pd.DataFrame):
        raise TypeError("studio_data must be a pandas DataFrame.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")

    market = market.strip()
    home_cluster = home_cluster.strip()
    market_studios = studio_data.loc[studio_data["network_market"] == market].copy()
    if market_studios.empty:
        raise ValueError("No studios found for market '" + market + "'.")

    requested_studio_count = min(requested_studio_count, len(market_studios))
    home_cluster_studios = market_studios.loc[market_studios["local_ridership_cluster"] == home_cluster].copy()
    if home_cluster_studios.empty:
        raise ValueError("No studios found in cluster '" + home_cluster + "' for market '" + market + "'.")

    home_probabilities = home_cluster_studios["weekly_bike_supply"] / home_cluster_studios["weekly_bike_supply"].sum()
    selected_home_index = rng.choice(home_cluster_studios.index.to_numpy(), p=home_probabilities.to_numpy())
    selected_studio_ids = [str(home_cluster_studios.loc[selected_home_index, "studio_id"])]

    remaining_count = requested_studio_count - 1
    if remaining_count <= 0:
        return selected_studio_ids

    remaining_studios = market_studios.loc[~market_studios["studio_id"].isin(selected_studio_ids)].copy()
    remaining_count = min(remaining_count, len(remaining_studios))
    if remaining_count <= 0:
        return selected_studio_ids

    remaining_probabilities = remaining_studios["weekly_bike_supply"] / remaining_studios["weekly_bike_supply"].sum()
    selected_indices = rng.choice(remaining_studios.index.to_numpy(), size=remaining_count, replace=False, p=remaining_probabilities.to_numpy())
    selected_studio_ids.extend(remaining_studios.loc[selected_indices, "studio_id"].astype(str).tolist())

    return selected_studio_ids

def allocate_classes_across_studios(baseline_class_count: int, regular_studio_ids: list[str], studio_data: pd.DataFrame, rng: np.random.Generator) -> dict[str, int]:
    if isinstance(baseline_class_count, bool) or not isinstance(baseline_class_count, int):
        raise TypeError("baseline_class_count must be an integer.")
    if baseline_class_count <= 0:
        raise ValueError("baseline_class_count must be positive.")
    if not isinstance(regular_studio_ids, list):
        raise TypeError("regular_studio_ids must be a list.")
    if not regular_studio_ids:
        raise ValueError("regular_studio_ids cannot be empty.")
    if not isinstance(studio_data, pd.DataFrame):
        raise TypeError("studio_data must be a pandas DataFrame.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")

    if len(regular_studio_ids) > baseline_class_count:
        regular_studio_ids = regular_studio_ids[:baseline_class_count]

    allocation = {studio_id: 1 for studio_id in regular_studio_ids}
    remaining_classes = baseline_class_count - len(regular_studio_ids)
    if remaining_classes <= 0:
        return allocation

    selected_studios = studio_data.set_index("studio_id").loc[regular_studio_ids]
    weights = selected_studios["weekly_bike_supply"].astype(float)
    probabilities = (weights / weights.sum()).to_numpy()
    extra_allocations = rng.multinomial(n=remaining_classes, pvals=probabilities)

    for studio_id, extra_classes in zip(regular_studio_ids, extra_allocations):
        allocation[studio_id] += int(extra_classes)

    if sum(allocation.values()) != baseline_class_count:
        raise RuntimeError("Studio class allocation did not preserve baseline_class_count.")

    return allocation

def summarize_studio_capacity_vs_demand(instructors: dict[str, Instructor], studios: dict[str, Studio]) -> pd.DataFrame:
    #compare total instructor demand at each studio against available recurring class slots
    demand_by_studio: dict[str, int] = {}

    for instructor in instructors.values():
        for studio_id, class_count in instructor.baseline_studio_allocations.items():
            demand_by_studio[studio_id] = demand_by_studio.get(studio_id, 0) + class_count

    rows: list[dict[str, object]] = []
    for studio_id, studio in studios.items():
        requested_classes = demand_by_studio.get(studio_id, 0)
        available_classes = studio.weekly_class_count
        rows.append({
            "studio_id": studio_id,
            "network_market": studio.network_market,
            "home_cluster": studio.local_ridership_cluster,
            "available_classes": available_classes,
            "requested_classes": requested_classes,
            "surplus_or_shortfall": available_classes - requested_classes,
            "overallocated": requested_classes > available_classes,
        })

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(by=["overallocated", "studio_id"], ascending=[False, True]).reset_index(drop=True)

    return summary

def count_overallocated_studios(instructors: dict[str, Instructor], studios: dict[str, Studio]) -> int:
    summary = summarize_studio_capacity_vs_demand(instructors, studios)
    if summary.empty:
        return 0
    return int(summary["overallocated"].sum())
