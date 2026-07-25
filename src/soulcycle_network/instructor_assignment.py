# Place instructors in clusters and studios.

import numpy as np
import pandas as pd
from soulcycle_network.instructor import Instructor
from soulcycle_network.studio import Studio

def split_by_weights(total: int, weights: pd.Series) -> dict[str, int]:
    if isinstance(total, bool) or not isinstance(total, int):
        raise TypeError("total must be an integer.")
    if not isinstance(weights, pd.Series):
        raise TypeError("weights must be a pandas Series.")

    numeric_weights = pd.to_numeric(weights, errors="raise").astype(float)
    if numeric_weights.sum() <= 0:
        raise ValueError("weights must sum to a positive value.")

    exact = (numeric_weights / numeric_weights.sum()) * total
    ints = np.floor(exact).astype(int)
    left = total - int(ints.sum())
    remainders = (exact - ints).sort_values(ascending=False)

    for key in remainders.index[:left]:
        ints.loc[key] += 1

    return ints.to_dict()

def market_class_supply(market: str, studio_data: pd.DataFrame) -> int:
    if not isinstance(market, str):
        raise TypeError("market must be a string.")
    if not isinstance(studio_data, pd.DataFrame):
        raise TypeError("studio_data must be a pandas DataFrame.")

    rows = studio_data.loc[studio_data["network_market"] == market.strip()]
    if rows.empty:
        raise ValueError("No studios found for market '" + market + "'.")
    return int(rows["weekly_class_count"].sum())

def calibrate_class_loads(raw_counts: pd.Series, target_total: int) -> pd.Series:
    #every instructor keeps at least one baseline class; the rest follow raw load shape
    if not isinstance(raw_counts, pd.Series):
        raise TypeError("raw_counts must be a pandas Series.")
    if isinstance(target_total, bool) or not isinstance(target_total, int):
        raise TypeError("target_total must be an integer.")

    counts = pd.to_numeric(raw_counts, errors="raise").astype(float)
    n = len(counts)
    if target_total < n:
        raise ValueError("target_total is too small to give every instructor at least one baseline class.")

    left = target_total - n
    w = (counts - 1).clip(lower=0)
    if w.sum() == 0:
        w = pd.Series(1.0, index=counts.index)

    extra = split_by_weights(left, w)
    out = pd.Series({iid: 1 + extra[iid] for iid in counts.index}, dtype=int)

    if int(out.sum()) != target_total:
        raise RuntimeError("Calibrated class counts do not match target_total.")
    return out

def allocate_clusters(market: str, n_instructors: int, studio_data: pd.DataFrame) -> dict[str, int]:
    #cluster weights use weekly bike supply, not slot counts
    if not isinstance(market, str):
        raise TypeError("market must be a string.")
    if isinstance(n_instructors, bool) or not isinstance(n_instructors, int):
        raise TypeError("n_instructors must be an integer.")
    if not isinstance(studio_data, pd.DataFrame):
        raise TypeError("studio_data must be a pandas DataFrame.")

    rows = studio_data.loc[studio_data["network_market"] == market.strip()].copy()
    if rows.empty:
        raise ValueError("No studios found for market '" + market + "'.")

    w = rows.groupby("local_ridership_cluster")["weekly_bike_supply"].sum().sort_index()
    return split_by_weights(n_instructors, w)

def pick_studios(market: str, home_cluster: str, n_studios: int, studio_data: pd.DataFrame, rng: np.random.Generator, cap_left: dict[str, int]) -> list[str]:
    if not isinstance(market, str):
        raise TypeError("market must be a string.")
    if not isinstance(home_cluster, str):
        raise TypeError("home_cluster must be a string.")
    if isinstance(n_studios, bool) or not isinstance(n_studios, int):
        raise TypeError("n_studios must be an integer.")
    if not isinstance(studio_data, pd.DataFrame):
        raise TypeError("studio_data must be a pandas DataFrame.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")
    if not isinstance(cap_left, dict):
        raise TypeError("cap_left must be a dictionary.")

    market = market.strip()
    home_cluster = home_cluster.strip()
    rows = studio_data.loc[studio_data["network_market"] == market].copy()
    rows = rows.loc[rows["studio_id"].apply(lambda sid: cap_left.get(str(sid), 0) > 0)].copy()

    if rows.empty:
        raise ValueError("No studios with remaining capacity found for market '" + market + "'.")

    n_studios = min(n_studios, len(rows))
    home_rows = rows.loc[rows["local_ridership_cluster"] == home_cluster].copy()
    if home_rows.empty:
        home_rows = rows.copy()

    p = home_rows["weekly_bike_supply"] / home_rows["weekly_bike_supply"].sum()
    home_idx = rng.choice(home_rows.index.to_numpy(), p=p.to_numpy())
    picked = [str(home_rows.loc[home_idx, "studio_id"])]

    left = n_studios - 1
    if left <= 0:
        return picked

    rest = rows.loc[~rows["studio_id"].isin(picked)].copy()
    left = min(left, len(rest))
    if left <= 0:
        return picked

    p = rest["weekly_bike_supply"] / rest["weekly_bike_supply"].sum()
    idx = rng.choice(rest.index.to_numpy(), size=left, replace=False, p=p.to_numpy())
    picked.extend(rest.loc[idx, "studio_id"].astype(str).tolist())
    return picked

def init_capacity(studio_data: pd.DataFrame) -> dict[str, int]:
    if not isinstance(studio_data, pd.DataFrame):
        raise TypeError("studio_data must be a pandas DataFrame.")

    cap_left: dict[str, int] = {}
    for _, row in studio_data.iterrows():
        cap_left[str(row["studio_id"])] = int(row["weekly_class_count"])
    return cap_left

def allocate_classes(n_classes: int, studio_ids: list[str], studio_data: pd.DataFrame, rng: np.random.Generator, market: str, cap_left: dict[str, int]) -> dict[str, int]:
    if isinstance(n_classes, bool) or not isinstance(n_classes, int):
        raise TypeError("n_classes must be an integer.")
    if not isinstance(studio_ids, list):
        raise TypeError("studio_ids must be a list.")
    if not isinstance(studio_data, pd.DataFrame):
        raise TypeError("studio_data must be a pandas DataFrame.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")
    if not isinstance(market, str):
        raise TypeError("market must be a string.")
    if not isinstance(cap_left, dict):
        raise TypeError("cap_left must be a dictionary.")

    rows = studio_data.loc[studio_data["network_market"] == market.strip()].copy()
    out: dict[str, int] = {}
    left = n_classes

    while left > 0:
        preferred = [sid for sid in studio_ids if cap_left.get(sid, 0) > 0]
        market_ids = [str(sid) for sid in rows["studio_id"] if cap_left.get(str(sid), 0) > 0]
        candidates = preferred if preferred else market_ids

        if not candidates:
            raise RuntimeError("No remaining studio capacity in market '" + market + "'.")

        w = studio_data.set_index("studio_id").loc[candidates]["weekly_bike_supply"].astype(float)
        p = (w / w.sum()).to_numpy()
        pick = candidates[int(rng.choice(len(candidates), p=p))]

        out[pick] = out.get(pick, 0) + 1
        cap_left[pick] -= 1
        left -= 1

    if sum(out.values()) != n_classes:
        raise RuntimeError("Studio class allocation did not preserve n_classes.")
    return out

def capacity_summary(instructors: dict[str, Instructor], studios: dict[str, Studio]) -> pd.DataFrame:
    if not isinstance(instructors, dict):
        raise TypeError("instructors must be a dictionary.")
    if not isinstance(studios, dict):
        raise TypeError("studios must be a dictionary.")

    demand: dict[str, int] = {}
    for instructor in instructors.values():
        for sid, n in instructor.baseline_studio_allocations.items():
            demand[sid] = demand.get(sid, 0) + n

    rows: list[dict[str, object]] = []
    for sid, studio in studios.items():
        req = demand.get(sid, 0)
        avail = studio.weekly_class_count
        rows.append({
            "studio_id": sid,
            "network_market": studio.network_market,
            "home_cluster": studio.local_ridership_cluster,
            "available_classes": avail,
            "requested_classes": req,
            "surplus_or_shortfall": avail - req,
            "overallocated": req > avail,
        })

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(by=["overallocated", "studio_id"], ascending=[False, True]).reset_index(drop=True)
    return summary

def count_overallocated(instructors: dict[str, Instructor], studios: dict[str, Studio]) -> int:
    summary = capacity_summary(instructors, studios)
    if summary.empty:
        return 0
    return int(summary["overallocated"].sum())
