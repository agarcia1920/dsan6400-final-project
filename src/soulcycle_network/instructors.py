# Instructor roster, baseline class assignment, and weekly on/off availability.

from __future__ import annotations

from soulcycle_network.studios import BaselineClassSlot, Studio

# Instructor objects for the SoulCycle network simulation.

from dataclasses import dataclass, field

MARKET_TIERS = {"Mega", "Large", "Medium", "Concentrated"}

@dataclass
class Instructor:
    instructor_id: str
    instructor_name: str
    network_market: str #the market this instructor mainly works in
    market_tier: str
    home_cluster: str | None = None #where in the market the instructor is based
    baseline_class_count: int | None = None #their normal weekly teaching load
    regular_studio_assignments: list[str] = field(default_factory=list) #studios they regularly teach at
    baseline_studio_allocations: dict[str, int] = field(default_factory=dict) #how many classes at each regular studio
    baseline_day_counts: dict[str, int] = field(default_factory=dict) #filled in later when we assign days
    baseline_slot_ids: list[str] = field(default_factory=list) #filled in later when we assign specific slots

    def __post_init__(self) -> None:
        #validate the input types and values
        if not isinstance(self.instructor_id, str):
            raise TypeError("instructor_id must be a string.")
        if not isinstance(self.instructor_name, str):
            raise TypeError("instructor_name must be a string.")
        if not isinstance(self.network_market, str):
            raise TypeError("network_market must be a string.")
        if not isinstance(self.market_tier, str):
            raise TypeError("market_tier must be a string.")
        
        #strip whitespace
        self.instructor_id = self.instructor_id.strip()
        self.instructor_name = self.instructor_name.strip()
        self.network_market = self.network_market.strip()
        self.market_tier = self.market_tier.strip()

        #validate the input values further
        if not self.instructor_id:
            raise ValueError("instructor_id cannot be empty.")
        if not self.instructor_name:
            raise ValueError("Instructor " + self.instructor_id + " must have an instructor_name.")
        if not self.network_market:
            raise ValueError("Instructor " + self.instructor_id + " must have a network_market.")
        if self.market_tier not in MARKET_TIERS:
            raise ValueError("Invalid market_tier '" + self.market_tier + "' for instructor " + self.instructor_id + ". Expected one of " + str(sorted(MARKET_TIERS)) + ".")

        #validate the home cluster
        if self.home_cluster is not None:
            if not isinstance(self.home_cluster, str):
                raise TypeError("home_cluster for " + self.instructor_id + " must be a string or None.")
            self.home_cluster = self.home_cluster.strip()
            if not self.home_cluster:
                raise ValueError("home_cluster for " + self.instructor_id + " cannot be an empty string.")

        #validate the baseline class count
        if self.baseline_class_count is not None:
            if isinstance(self.baseline_class_count, bool) or not isinstance(self.baseline_class_count, int):
                raise TypeError("baseline_class_count for " + self.instructor_id + " must be an integer or None.")
            if self.baseline_class_count <= 0:
                raise ValueError("baseline_class_count for " + self.instructor_id + " must be positive.")

        #validate the regular studio assignments
        if not isinstance(self.regular_studio_assignments, list):
            raise TypeError("regular_studios for " + self.instructor_id + " must be a list.")
        if not isinstance(self.baseline_studio_allocations, dict):
            raise TypeError("baseline_studio_allocation for " + self.instructor_id + " must be a dictionary.")
        if not isinstance(self.baseline_day_counts, dict):
            raise TypeError("baseline_day_counts for " + self.instructor_id + " must be a dictionary.")
        if not isinstance(self.baseline_slot_ids, list):
            raise TypeError("baseline_slot_ids for " + self.instructor_id + " must be a list.")

# Estimate instructor behavior from the sample file and load generator inputs.

from pathlib import Path
import numpy as np
import pandas as pd
from soulcycle_network.config import CLASS_LOAD_STUDIO_EFFECT

#mapping of markets to tiers
MARKET_TO_TIER = {
    "Greater NYC": "Mega",
    "DMV": "Large",
    "Southern California": "Large",
    "Northern California": "Large",
    "Boston": "Medium",
    "South Florida": "Medium",
    "Chicago": "Medium",
    "Philadelphia": "Medium",
    "Austin": "Concentrated",
    "Dallas": "Concentrated",
    "Houston": "Concentrated",
    "Atlanta": "Concentrated",
    "Seattle": "Concentrated",
    "Ann Arbor": "Concentrated",
}

#bounds on the number of studios an instructor can be assigned to in a given week
STUDIO_COUNT_BOUNDS = {
    "Mega": (1, 7),
    "Large": (1, 4),
    "Medium": (1, 3),
    "Concentrated": (1, 2),
}

#validate the columns of a dataframe
def validate_cols(df: pd.DataFrame, required: set[str], name: str) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if not isinstance(required, set):
        raise TypeError("required must be a set.")
    if not isinstance(name, str):
        raise TypeError("name must be a string.")

    missing = required - set(df.columns)
    if missing:
        raise ValueError("The following columns are missing from " + name + ": " + str(sorted(missing)))

#add the bike supply to the studio data
def add_bike_supply(studio_data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(studio_data, pd.DataFrame):
        raise TypeError("studio_data must be a pandas DataFrame.")

    studio_data = studio_data.copy()
    studio_data["rides_per_wk_a"] = pd.to_numeric(studio_data["rides_per_wk_a"], errors="raise")
    studio_data["rides_per_wk_b"] = pd.to_numeric(studio_data["rides_per_wk_b"], errors="raise")
    studio_data["bikes_per_ride_a"] = pd.to_numeric(studio_data["bikes_per_ride_a"], errors="raise")
    studio_data["bikes_per_ride_b"] = pd.to_numeric(studio_data["bikes_per_ride_b"], errors="raise")
    studio_data["weekly_bikes_a"] = studio_data["rides_per_wk_a"] * studio_data["bikes_per_ride_a"]
    studio_data["weekly_bikes_b"] = studio_data["rides_per_wk_b"] * studio_data["bikes_per_ride_b"]
    studio_data["weekly_bike_supply"] = studio_data["weekly_bikes_a"] + studio_data["weekly_bikes_b"]
    studio_data["weekly_class_count"] = studio_data["rides_per_wk_a"] + studio_data["rides_per_wk_b"]
    return studio_data

#calculate the tier parameters from the sample data
def tier_params_from_sample(sample_data: pd.DataFrame) -> dict[str, dict[str, float]]:
    #validate the input types and values
    if not isinstance(sample_data, pd.DataFrame):
        raise TypeError("sample_data must be a pandas DataFrame.")

    validate_cols(sample_data, {"market", "classes_jul20_26", "studios_taught_count"}, "instructor sample")

    sample_data = sample_data.copy() #copy the sample data
    sample_data["classes_jul20_26"] = pd.to_numeric(sample_data["classes_jul20_26"], errors="raise")
    sample_data["studios_taught_count"] = pd.to_numeric(sample_data["studios_taught_count"], errors="raise")
    sample_data["market_tier"] = sample_data["market"].map(MARKET_TO_TIER) #map the market to the tier

    if sample_data["market_tier"].isna().any(): #check if there are any missing market tiers
        missing = sorted(sample_data.loc[sample_data["market_tier"].isna(), "market"].dropna().unique())
        raise ValueError("The following markets do not have a market-tier mapping: " + str(missing))

    scheduled = sample_data.loc[sample_data["classes_jul20_26"] > 0].copy() #copy the scheduled instructors
    if scheduled.empty:
        raise ValueError("Instructor sample contains no scheduled instructors.")

    params: dict[str, dict[str, float]] = {} #initialize the parameters

    for tier, rows in scheduled.groupby("market_tier"):
        class_sd = float(rows["classes_jul20_26"].std(ddof=1)) #calculate the standard deviation of the class counts
        studio_sd = float(rows["studios_taught_count"].std(ddof=1))
        if np.isnan(class_sd):
            class_sd = 0.0
        if np.isnan(studio_sd):
            studio_sd = 0.0

        params[tier] = {
            "class_mean": float(rows["classes_jul20_26"].mean()),
            "class_sd": class_sd,
            "class_min_observed": int(rows["classes_jul20_26"].min()),
            "class_max_observed": int(rows["classes_jul20_26"].max()),
            "studio_mean": float(rows["studios_taught_count"].mean()),
            "studio_sd": studio_sd,
            "scheduled_sample_size": int(len(rows)),
        }

    missing_tiers = set(MARKET_TO_TIER.values()) - set(params)
    if missing_tiers:
        raise ValueError("The instructor sample does not contain scheduled instructors for tiers: " + str(sorted(missing_tiers)))

    return params

def draw_class_count(tier: str, tier_params: dict[str, dict[str, float]], rng: np.random.Generator) -> int:
    #validate the input types and values
    if not isinstance(tier, str):
        raise TypeError("tier must be a string.")
    if not isinstance(tier_params, dict):
        raise TypeError("tier_params must be a dictionary.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")

    p = tier_params[tier.strip()] #get the tier parameters
    mean = float(p["class_mean"])
    sd = float(p["class_sd"]) #get the standard deviation of the class counts
    lo = max(1, int(p["class_min_observed"]) - 1) #get the lower bound of the class counts
    hi = int(p["class_max_observed"]) + 1 #get the upper bound of the class counts

    if sd <= 0:
        return int(np.clip(round(mean), lo, hi))

    while True:
        draw = rng.normal(loc=mean, scale=sd) #draw a normal distribution
        if lo <= draw <= hi:
            return int(np.clip(round(draw), lo, hi))

def draw_studio_count(tier: str, n_classes: int, tier_params: dict[str, dict[str, float]], rng: np.random.Generator) -> int:
    #validate the input types and values
    if not isinstance(tier, str):
        raise TypeError("tier must be a string.")
    if isinstance(n_classes, bool) or not isinstance(n_classes, int):
        raise TypeError("n_classes must be an integer.")
    if not isinstance(tier_params, dict):
        raise TypeError("tier_params must be a dictionary.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")

    tier = tier.strip() #strip the tier
    p = tier_params[tier] #get the tier parameters
    class_mean = float(p["class_mean"])
    class_sd = float(p["class_sd"]) #get the standard deviation of the class counts
    studio_mean = float(p["studio_mean"]) #get the mean of the studio counts
    studio_sd = float(p["studio_sd"]) #get the standard deviation of the studio counts

    if class_sd > 0:
        z = (n_classes - class_mean) / class_sd
    else:
        z = 0.0


    adj_mean = studio_mean + CLASS_LOAD_STUDIO_EFFECT * z #calculate the adjusted mean of the studio counts
    lo, hi = STUDIO_COUNT_BOUNDS[tier] #get the lower and upper bounds of the studio counts

    if studio_sd <= 0:
        return int(np.clip(round(adj_mean), lo, hi))

    while True:
        draw = rng.normal(loc=adj_mean, scale=studio_sd) #draw a normal distribution
        if lo <= draw <= hi:
            return int(np.clip(round(draw), lo, hi))

def load_inputs(active_path: str | Path, sample_path: str | Path, studio_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int], dict[str, dict[str, float]]]:
    active_path = Path(active_path)
    sample_path = Path(sample_path)
    studio_path = Path(studio_path)

    for path in (active_path, sample_path, studio_path):
        if not path.exists():
            raise FileNotFoundError("Input file not found: " + str(path))

    active_data = pd.read_csv(active_path) #read the active instructor data
    sample_data = pd.read_csv(sample_path) #read the instructor sample data
    studio_data = pd.read_csv(studio_path) #read the studio data

    validate_cols(active_data, {"instructor_name", "network_markets", "market_tier"}, "active instructor file") #validate the columns of the active instructor data from the active instructor file
    validate_cols(sample_data, {"sample_id", "market", "classes_jul20_26", "studios_taught_count"}, "instructor sample file") #validate the columns of the instructor sample data from the instructor sample file
    validate_cols(studio_data, {"studio_id", "studio_name", "network_market", "market_tier", "local_ridership_cluster", "rides_per_wk_a", "bikes_per_ride_a", "rides_per_wk_b", "bikes_per_ride_b"}, "studio file") #validate the columns of the studio data from the studio file

    active_data = active_data.copy() #copy the active instructor data
    sample_data = sample_data.copy() #copy the instructor sample data
    studio_data = studio_data.copy() #copy the studio data

    #clean the data
    active_data["network_markets"] = active_data["network_markets"].astype(str).str.strip()
    active_data["market_tier"] = active_data["market_tier"].astype(str).str.strip()
    studio_data["network_market"] = studio_data["network_market"].astype(str).str.strip()
    studio_data["market_tier"] = studio_data["market_tier"].astype(str).str.strip()
    studio_data["local_ridership_cluster"] = studio_data["local_ridership_cluster"].astype(str).str.strip()
    studio_data["studio_id"] = studio_data["studio_id"].astype(str).str.strip()
    studio_data = add_bike_supply(studio_data)

    #validate the studio data
    if (studio_data["weekly_bike_supply"] <= 0).any():
        raise ValueError("Every studio must have positive weekly_bike_supply.")
    if studio_data["studio_id"].duplicated().any():
        dups = sorted(studio_data.loc[studio_data["studio_id"].duplicated(keep=False), "studio_id"].unique())
        raise ValueError("Duplicate studio IDs found: " + str(dups))

    market_counts = active_data["network_markets"].value_counts().sort_index().astype(int).to_dict() #get the number of active instructors for each market
    tier_params = tier_params_from_sample(sample_data) #get the tier parameters from the sample data

    missing = set(market_counts) - set(studio_data["network_market"]) #check if there are any missing markets
    if missing:
        raise ValueError("No studios were found for these active markets: " + str(sorted(missing)))

    return active_data, sample_data, studio_data, market_counts, tier_params

# Place instructors in clusters and studios.

import numpy as np
import pandas as pd
from soulcycle_network.studios import Studio

def split_by_weights(total: int, weights: pd.Series) -> dict[str, int]:
    # validate input types and values
    if isinstance(total, bool) or not isinstance(total, int):
        raise TypeError("total must be an integer.")
    if not isinstance(weights, pd.Series):
        raise TypeError("weights must be a pandas Series.")

    # convert weights to numeric and validate
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
    # validate input types and values
    if not isinstance(market, str):
        raise TypeError("market must be a string.")
    if not isinstance(studio_data, pd.DataFrame):
        raise TypeError("studio_data must be a pandas DataFrame.")

    # filter studios by market and sum weekly class counts
    rows = studio_data.loc[studio_data["network_market"] == market.strip()]
    if rows.empty:
        raise ValueError("No studios found for market '" + market + "'.")
    return int(rows["weekly_class_count"].sum())

def calibrate_class_loads(raw_counts: pd.Series, target_total: int) -> pd.Series:
    #every instructor keeps at least one baseline class; the rest follow raw load shape
    #validate input types and values
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
    # validate input types and values
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

    # strip whitespace
    market=market.strip()
    home_cluster = home_cluster.strip()
    rows = studio_data.loc[studio_data["network_market"] == market].copy()
    # filter studios by remaining capacity
    rows = rows.loc[rows["studio_id"].apply(lambda sid: cap_left.get(str(sid), 0) > 0)].copy()

    if rows.empty:
        raise ValueError("No studios with remaining capacity found for market '" + market + "'.")

    # select the home cluster studios
    home_rows = rows.loc[rows["local_ridership_cluster"] == home_cluster].copy()
    if home_rows.empty:
        home_rows = rows.copy()

    # select the home cluster studio
    p = home_rows["weekly_bike_supply"] / home_rows["weekly_bike_supply"].sum()
    home_idx = rng.choice(home_rows.index.to_numpy(), p=p.to_numpy())
    picked = [str(home_rows.loc[home_idx, "studio_id"])]

    # select the remaining studios
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
    # validate input types and values
    if not isinstance(studio_data, pd.DataFrame):
        raise TypeError("studio_data must be a pandas DataFrame.")

    cap_left: dict[str, int] = {}
    for _, row in studio_data.iterrows():
        cap_left[str(row["studio_id"])] = int(row["weekly_class_count"])
    return cap_left

def allocate_classes(n_classes: int, studio_ids: list[str], studio_data: pd.DataFrame, rng: np.random.Generator, market: str, cap_left: dict[str, int]) -> dict[str, int]:
    # validate input types and values
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

    # strip whitespace
    rows = studio_data.loc[studio_data["network_market"] == market.strip()].copy()
    out: dict[str, int] = {}
    left = n_classes

    while left > 0:
        # select the preferred studios
        preferred = [sid for sid in studio_ids if cap_left.get(sid, 0) > 0]
        market_ids = [str(sid) for sid in rows["studio_id"] if cap_left.get(str(sid), 0) > 0]
        candidates = preferred if preferred else market_ids
        # select the studio with the highest weekly bike supply

        if not candidates:
            raise RuntimeError("No remaining studio capacity in market '" + market + "'.")

        w = studio_data.set_index("studio_id").loc[candidates]["weekly_bike_supply"].astype(float)
        p = (w / w.sum()).to_numpy()
        pick = candidates[int(rng.choice(len(candidates), p=p))]

        out[pick] = out.get(pick, 0) + 1
        cap_left[pick] -= 1
        left -= 1

    # validate the total number of classes allocated
    if sum(out.values()) != n_classes:
        raise RuntimeError("Studio class allocation did not preserve n_classes.")
    return out

def capacity_summary(instructors: dict[str, Instructor], studios: dict[str, Studio]) -> pd.DataFrame:
    # validate input types and values
    if not isinstance(instructors, dict):
        raise TypeError("instructors must be a dictionary.")
    if not isinstance(studios, dict):
        raise TypeError("studios must be a dictionary.")

    # calculate the demand for each studio
    demand: dict[str, int] = {}
    for instructor in instructors.values():
        for sid, n in instructor.baseline_studio_allocations.items():
            demand[sid] = demand.get(sid, 0) + n

    # create a summary of the capacity and demand
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

    # create a summary of the capacity and demand
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(by=["overallocated", "studio_id"], ascending=[False, True]).reset_index(drop=True)
    return summary

def count_overallocated(instructors: dict[str, Instructor], studios: dict[str, Studio]) -> int:
    #return the number of studios that are overallocated
    summary = capacity_summary(instructors, studios)
    if summary.empty:
        return 0
    return int(summary["overallocated"].sum())

# Generate the synthetic instructor population.

from pathlib import Path
import numpy as np
import pandas as pd
from faker import Faker

def generate_names(n: int, fake: Faker) -> list[str]:
    # validate input types and values
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be an integer.")
    if not isinstance(fake, Faker):
        raise TypeError("fake must be a Faker object.")

    names: list[str] = []   # list to store the generated names
    seen: set[str] = set()   # set to store the names that have already been seen

    while len(names) < n:
        candidate = fake.name().strip()   # generate a name and strip whitespace
        if candidate and candidate not in seen:
            names.append(candidate)   # add the name to the list if it is not already in the set
            seen.add(candidate)   # add the name to the set

    return names

def generate_instructors(active_path: str | Path, sample_path: str | Path, studio_path: str | Path, rng: np.random.Generator, fake: Faker) -> dict[str, Instructor]:
    # validate input types and values
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")
    if not isinstance(fake, Faker):
        raise TypeError("fake must be a Faker object.")

    active_data, sample_data, studio_data, market_counts, tier_params = load_inputs(active_path, sample_path, studio_path)
    # calculate the total number of instructors needed
    total = int(sum(market_counts.values()))
    names = generate_names(total, fake)   # generate the names
    name_iter = iter(names)   # create an iterator over the names
    instructors: dict[str, Instructor] = {}   # dictionary to store the instructors
    n = 1   # counter for the number of instructors
    cap_left = init_capacity(studio_data)   # initialize the capacity left

    for market in sorted(market_counts):
        n_market = int(market_counts[market])
        tier = MARKET_TO_TIER[market]   # get the tier for the market
        cluster_alloc = allocate_clusters(market, n_market, studio_data)
        clusters: list[str] = []   # list to store the clusters
        for cluster, count in cluster_alloc.items():
            clusters.extend([cluster] * count)   # add the cluster to the list for each count
        rng.shuffle(clusters)   # shuffle the clusters

        roster: list[dict[str, str]] = []   # list to store the roster
        for cluster in clusters:
            roster.append({
                "instructor_id": "I" + str(n).zfill(4),   # generate a unique instructor id
                "instructor_name": next(name_iter),   # get the next name from the iterator
                "home_cluster": cluster,
            })
            n += 1   # increment the counter

        raw_counts = pd.Series({entry["instructor_id"]: draw_class_count(tier, tier_params, rng) for entry in roster})   # draw the class counts for each instructor
        target = market_class_supply(market, studio_data)   # calculate the target class supply for the market
        class_counts = calibrate_class_loads(raw_counts, target)
        rng.shuffle(roster)   # shuffle the roster

        for entry in roster:
            iid = entry["instructor_id"]   # get the instructor id
            n_classes = int(class_counts[iid])
            n_studios = draw_studio_count(tier, n_classes, tier_params, rng)   # draw the number of studios for the instructor
            n_studios = min(n_studios, n_classes)   # limit the number of studios to the number of classes
            studio_ids = pick_studios(market, entry["home_cluster"], n_studios, studio_data, rng, cap_left)   # pick the studios for the instructor
            alloc = allocate_classes(n_classes, studio_ids, studio_data, rng, market, cap_left)   # allocate the classes to the studios

            instructors[iid] = Instructor(
                instructor_id=iid, # set the instructor id
                instructor_name=entry["instructor_name"], # set the instructor name
                network_market=market, # set the network market
                market_tier=tier, # set the market tier
                home_cluster=entry["home_cluster"], # set the home cluster
                baseline_class_count=n_classes, # set the baseline class count
                regular_studio_assignments=list(alloc.keys()), # set the regular studio assignments
                baseline_studio_allocations=alloc, # set the baseline studio allocations
                baseline_day_counts={}, # set the baseline day counts
                baseline_slot_ids=[], # set the baseline slot ids
            )

    if len(instructors) != total:   # check if the number of instructors generated is the same as the total number of instructors needed
        raise RuntimeError("Generated " + str(len(instructors)) + " instructors, but expected " + str(total) + ".")

    got = pd.Series([i.network_market for i in instructors.values()]).value_counts().to_dict()   # get the number of instructors for each market
    if got != market_counts:
        raise RuntimeError("Generated instructor market counts do not match the active instructor population.")

    return instructors

def instructors_to_dataframe(instructors: dict[str, Instructor]) -> pd.DataFrame:
    # validate input types and values
    if not isinstance(instructors, dict):
        raise TypeError("instructors must be a dictionary.")

    rows: list[dict[str, object]] = []   # list to store the rows

    for instructor in instructors.values():
        if not isinstance(instructor, Instructor):   # check if the instructor is an Instructor object
            raise TypeError("instructors must contain Instructor objects.")

        parts = []
        for sid, n in instructor.baseline_studio_allocations.items():
            parts.append(sid + ":" + str(n))
        # add the instructor to the rows
        rows.append({
            "instructor_id": instructor.instructor_id,
            "instructor_name": instructor.instructor_name,
            "network_market": instructor.network_market,
            "market_tier": instructor.market_tier,
            "home_cluster": instructor.home_cluster,
            "baseline_class_count": instructor.baseline_class_count,
            "regular_studio_count": len(instructor.regular_studio_assignments),
            "regular_studio_ids": "; ".join(instructor.regular_studio_assignments),
            "baseline_studio_allocations": "; ".join(parts),
        })

    df = pd.DataFrame(rows) # create a dataframe from the rows
    if not df.empty:
        df = df.sort_values(by="instructor_id").reset_index(drop=True) # sort the dataframe by the instructor id
    return df

def save_instructors(instructors: dict[str, Instructor], output_path: str | Path) -> None:
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".csv":
        raise ValueError("Instructor output file must be a CSV.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    instructors_to_dataframe(instructors).to_csv(output_path, index=False)

# Assign each instructor to their baseline recurring slots.
# daily_slot_index is treated as a synchronized company-wide time-position proxy until explicit class times are modeled.

import numpy as np
from soulcycle_network.config import DAYS_OF_WEEK, DAY_INDEX

MAX_ASSIGN_TRIES = 50

def reset_baseline_assignment(instructors: dict[str, Instructor], class_slots: list[BaselineClassSlot]) -> None:
    for slot in class_slots:
        slot.usual_instructor = None
    for instructor in instructors.values():
        instructor.baseline_slot_ids = []
        instructor.baseline_day_counts = {}

def instructor_order(instructors: dict[str, Instructor], ids: list[str]) -> list[str]:
    ordered = list(ids)
    ordered.sort(key=lambda iid: (len(instructors[iid].regular_studio_assignments), instructors[iid].baseline_class_count), reverse=True)
    return ordered

def try_assign_baseline_slots(instructors: dict[str, Instructor], open_by_studio: dict[str, list[BaselineClassSlot]], ids: list[str], rng: np.random.Generator) -> None:
    for iid in ids:
        instructor = instructors[iid]
        if not isinstance(instructor, Instructor):
            raise TypeError("Value stored under " + iid + " must be an Instructor object.")

        assigned_slots: list[BaselineClassSlot] = []
        day_counts = {day: 0 for day in DAYS_OF_WEEK}
        busy_times: set[tuple[str, int]] = set()
        studio_allocs = list(instructor.baseline_studio_allocations.items())
        rng.shuffle(studio_allocs)

        for sid, n in studio_allocs:
            for _ in range(n):
                eligible = [slot for slot in open_by_studio.get(sid, []) if slot.usual_instructor is None and (slot.day_of_week, slot.daily_slot_index) not in busy_times]
                if not eligible:
                    raise RuntimeError("Studio " + sid + " does not have enough conflict-free slots for instructor " + iid + ".")

                pick = int(rng.choice(len(eligible)))
                slot = eligible[pick]
                slot.usual_instructor = instructor.instructor_id
                assigned_slots.append(slot)
                busy_times.add((slot.day_of_week, slot.daily_slot_index))
                day_counts[slot.day_of_week] += 1

        if len(assigned_slots) != instructor.baseline_class_count:
            raise RuntimeError("Instructor " + iid + " was assigned " + str(len(assigned_slots)) + " slots, but baseline_class_count is " + str(instructor.baseline_class_count) + ".")

        assigned_slots.sort(key=lambda slot: (DAY_INDEX[slot.day_of_week], slot.daily_slot_index, slot.slot_id))
        instructor.baseline_slot_ids = [slot.slot_id for slot in assigned_slots]
        instructor.baseline_day_counts = {day: n for day, n in day_counts.items() if n > 0}

def assign_baseline_slots(instructors: dict[str, Instructor], class_slots: list[BaselineClassSlot], rng: np.random.Generator) -> list[BaselineClassSlot]:
    if not isinstance(instructors, dict):
        raise TypeError("instructors must be a dictionary.")
    if not isinstance(class_slots, list):
        raise TypeError("class_slots must be a list.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")

    open_by_studio: dict[str, list[BaselineClassSlot]] = {}

    for slot in class_slots:
        if not isinstance(slot, BaselineClassSlot):
            raise TypeError("class_slots must contain BaselineClassSlot objects.")
        open_by_studio.setdefault(slot.studio_id, []).append(slot)

    for attempt in range(MAX_ASSIGN_TRIES):
        reset_baseline_assignment(instructors, class_slots)

        ids = list(instructors.keys())
        rng.shuffle(ids)
        ids = instructor_order(instructors, ids)

        try:
            try_assign_baseline_slots(instructors, open_by_studio, ids, rng)
            validate_baseline(instructors, class_slots)
            return class_slots
        except RuntimeError:
            if attempt == MAX_ASSIGN_TRIES - 1:
                raise RuntimeError("Could not assign conflict-free baseline slots after " + str(MAX_ASSIGN_TRIES) + " tries.")

    raise RuntimeError("Could not assign conflict-free baseline slots.")

def validate_baseline(instructors: dict[str, Instructor], class_slots: list[BaselineClassSlot]) -> None:
    if not isinstance(instructors, dict):
        raise TypeError("instructors must be a dictionary.")
    if not isinstance(class_slots, list):
        raise TypeError("class_slots must be a list.")

    lookup = {slot.slot_id: slot for slot in class_slots}
    assigned = 0
    seen: set[str] = set()

    for slot in class_slots:
        if not isinstance(slot, BaselineClassSlot):
            raise TypeError("class_slots must contain BaselineClassSlot objects.")
        if slot.usual_instructor is None:
            raise ValueError("Slot " + slot.slot_id + " does not have a usual instructor.")
        if slot.slot_id in seen:
            raise ValueError("Duplicate slot ID found during validation: " + slot.slot_id)
        seen.add(slot.slot_id)
        assigned += 1

    total = 0
    seen_slots: set[str] = set()

    for iid, instructor in instructors.items():
        if not isinstance(instructor, Instructor):
            raise TypeError("Value stored under " + iid + " must be an Instructor object.")
        if len(instructor.baseline_slot_ids) != instructor.baseline_class_count:
            raise ValueError("Instructor " + iid + " slot count does not match baseline_class_count.")
        if sum(instructor.baseline_day_counts.values()) != instructor.baseline_class_count:
            raise ValueError("Instructor " + iid + " day counts do not match baseline_class_count.")

        seen_times: set[tuple[str, int]] = set()
        for slot_id in instructor.baseline_slot_ids:
            if slot_id in seen_slots:
                raise ValueError("Slot " + slot_id + " was assigned to more than one instructor.")
            seen_slots.add(slot_id)

            slot = lookup[slot_id]
            time_key = (slot.day_of_week, slot.daily_slot_index)
            if time_key in seen_times:
                raise ValueError("Instructor " + iid + " has overlapping baseline slots at " + str(time_key) + ".")
            seen_times.add(time_key)

        total += instructor.baseline_class_count

    if assigned != len(class_slots):
        raise RuntimeError("Assigned slot count does not match class slot list length.")
    if total != assigned:
        raise RuntimeError("Total instructor baseline classes do not match assigned slot count.")

def day_load_totals(instructors: dict[str, Instructor]) -> dict[str, int]:
    if not isinstance(instructors, dict):
        raise TypeError("instructors must be a dictionary.")

    out = {day: 0 for day in DAYS_OF_WEEK}
    for instructor in instructors.values():
        for day, n in instructor.baseline_day_counts.items():
            out[day] += n
    return out

