"""Riders, parameters, and generation."""

from __future__ import annotations
# Rider objects for the SoulCycle network simulation.

from dataclasses import dataclass, field

import numpy as np

@dataclass
class Rider:
    rider_id: str
    rider_name: str
    home_market: str
    home_cluster: str
    baseline_annual_ride_rate: float #persistent annual ride propensity; weekly attendance is drawn from this
    preferred_studio_ids: list[str] = field(default_factory=list)
    preferred_instructor_ids: list[str] = field(default_factory=list)
    attended_session_ids: list[str] = field(default_factory=list) #filled in as the simulation runs
    attended_instructor_counts: dict[str, int] = field(default_factory=dict)
    attended_studio_counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        #validate the input types and values
        if not isinstance(self.rider_id, str):
            raise TypeError("rider_id must be a string.")
        if not isinstance(self.rider_name, str):
            raise TypeError("rider_name must be a string.")
        if not isinstance(self.home_market, str):
            raise TypeError("home_market must be a string.")
        if not isinstance(self.home_cluster, str):
            raise TypeError("home_cluster must be a string.")

        self.rider_id = self.rider_id.strip()
        self.rider_name = self.rider_name.strip()
        self.home_market = self.home_market.strip()
        self.home_cluster = self.home_cluster.strip()

        if not self.rider_id:
            raise ValueError("rider_id cannot be empty.")
        if not self.rider_name:
            raise ValueError("Rider " + self.rider_id + " must have a rider_name.")
        if not self.home_market:
            raise ValueError("Rider " + self.rider_id + " must have a home_market.")
        if not self.home_cluster:
            raise ValueError("Rider " + self.rider_id + " must have a home_cluster.")

        if isinstance(self.baseline_annual_ride_rate, bool) or not isinstance(
            self.baseline_annual_ride_rate,
            (int, float, np.integer, np.floating),
        ):
            raise TypeError("baseline_annual_ride_rate for " + self.rider_id + " must be a number.")
        self.baseline_annual_ride_rate = float(self.baseline_annual_ride_rate)
        if self.baseline_annual_ride_rate <= 0:
            raise ValueError("baseline_annual_ride_rate for " + self.rider_id + " must be positive.")

        if not isinstance(self.preferred_studio_ids, list):
            raise TypeError("preferred_studio_ids for " + self.rider_id + " must be a list.")
        if not isinstance(self.preferred_instructor_ids, list):
            raise TypeError("preferred_instructor_ids for " + self.rider_id + " must be a list.")
        if not isinstance(self.attended_session_ids, list):
            raise TypeError("attended_session_ids for " + self.rider_id + " must be a list.")

        for sid in self.preferred_studio_ids:
            if not isinstance(sid, str) or not sid.strip():
                raise ValueError("preferred_studio_ids for " + self.rider_id + " must contain non-empty strings only.")
        for iid in self.preferred_instructor_ids:
            if not isinstance(iid, str) or not iid.strip():
                raise ValueError("preferred_instructor_ids for " + self.rider_id + " must contain non-empty strings only.")
        for session_id in self.attended_session_ids:
            if not isinstance(session_id, str) or not session_id.strip():
                raise ValueError("attended_session_ids for " + self.rider_id + " must contain non-empty strings only.")

        if not isinstance(self.attended_instructor_counts, dict):
            raise TypeError("attended_instructor_counts for " + self.rider_id + " must be a dictionary.")
        if not isinstance(self.attended_studio_counts, dict):
            raise TypeError("attended_studio_counts for " + self.rider_id + " must be a dictionary.")

        for key, count in self.attended_instructor_counts.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("attended_instructor_counts keys for " + self.rider_id + " must be non-empty strings.")
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError("attended_instructor_counts values for " + self.rider_id + " must be integers.")
            if count < 0:
                raise ValueError("attended_instructor_counts for " + self.rider_id + " cannot be negative.")

        for key, count in self.attended_studio_counts.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("attended_studio_counts keys for " + self.rider_id + " must be non-empty strings.")
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError("attended_studio_counts values for " + self.rider_id + " must be integers.")
            if count < 0:
                raise ValueError("attended_studio_counts for " + self.rider_id + " cannot be negative.")

# Draw rider frequency parameters and weekly attendance counts.

import numpy as np
from soulcycle_network.config import DAYS_OF_WEEK, MAX_CLASSES_PER_DAY, MEAN_ANNUAL_RIDES, RIDER_FREQUENCY_PARAMETERS, TARGET_OCCUPANCY, TOTAL_SIMULATED_RIDERS

def coerce_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(name + " must be a number.")
    return float(value)

def draw_annual_ride_rate(rng: np.random.Generator, params: dict[str, float] | None = None) -> float:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")

    p = params if params is not None else RIDER_FREQUENCY_PARAMETERS #get the parameters
    if not isinstance(p, dict):
        raise TypeError("params must be a dictionary.")

    log_mean = float(p["log_mean"]) #get the log mean
    log_sd = float(p["log_sd"]) #get the log standard deviation
    minimum = float(p["minimum"])
    maximum = float(p["maximum"]) #get the maximum

    latent = rng.lognormal(mean=log_mean, sigma=log_sd)
    return float(np.clip(latent, minimum, maximum)) #return the annual ride rate

def draw_weekly_ride_count(rider: Rider, rng: np.random.Generator) -> int:
    if not isinstance(rider, Rider): #validate the input types and values
        raise TypeError("rider must be a Rider object.")
    if not isinstance(rng, np.random.Generator): #validate the input types and values
        raise TypeError("rng must be a NumPy Generator.")

    weekly_rate = rider.baseline_annual_ride_rate / 52.0 #calculate the weekly rate
    weekly_cap = len(DAYS_OF_WEEK) * MAX_CLASSES_PER_DAY
    return min(int(rng.poisson(weekly_rate)), weekly_cap) #return the weekly ride count

def mean_generated_annual_ride_rate(riders: dict[str, Rider]) -> float:
    if not isinstance(riders, dict):
        raise TypeError("riders must be a dictionary.")
    if not riders:
        raise ValueError("riders must not be empty.")
    rates = [rider.baseline_annual_ride_rate for rider in riders.values()]
    return float(np.mean(rates))

def estimate_implied_population(total_weekly_bike_supply: int, target_occupancy: float = TARGET_OCCUPANCY, mean_annual_rides: float = float(MEAN_ANNUAL_RIDES)) -> int:
    if isinstance(total_weekly_bike_supply, bool) or not isinstance(total_weekly_bike_supply, int): 
        raise TypeError("total_weekly_bike_supply must be an integer.")
    target_occupancy = coerce_float(target_occupancy, "target_occupancy")
    mean_annual_rides = coerce_float(mean_annual_rides, "mean_annual_rides")
    if total_weekly_bike_supply <= 0:
        raise ValueError("total_weekly_bike_supply must be positive.")
    if target_occupancy <= 0 or target_occupancy > 1:
        raise ValueError("target_occupancy must be between 0 and 1.")
    if mean_annual_rides <= 0:
        raise ValueError("mean_annual_rides must be positive.")

    return int(round((52 * total_weekly_bike_supply * target_occupancy) / mean_annual_rides)) #return the implied population

def simulation_scale(implied_population: int, total_simulated_riders: int = TOTAL_SIMULATED_RIDERS) -> float:
    if isinstance(implied_population, bool) or not isinstance(implied_population, int): 
        raise TypeError("implied_population must be an integer.")
    if isinstance(total_simulated_riders, bool) or not isinstance(total_simulated_riders, int):
        raise TypeError("total_simulated_riders must be an integer.")
    if implied_population <= 0:
        raise ValueError("implied_population must be positive.")
    if total_simulated_riders <= 0:
        raise ValueError("total_simulated_riders must be positive.")

    return float(total_simulated_riders) / float(implied_population) #return the simulation scale

def simulated_session_capacity(real_capacity: int, scale: float) -> int: #function to calculate the simulated session capacity
    if isinstance(real_capacity, bool) or not isinstance(real_capacity, int):
        raise TypeError("real_capacity must be an integer.")
    scale = coerce_float(scale, "scale")
    if real_capacity <= 0:
        raise ValueError("real_capacity must be positive.")
    if scale <= 0:
        raise ValueError("scale must be positive.")

    return max(1, int(round(real_capacity * scale)))

# Generating the synthetic rider population.

from pathlib import Path
import numpy as np
import pandas as pd
from faker import Faker
from soulcycle_network.config import TOTAL_SIMULATED_RIDERS
from soulcycle_network.instructors import Instructor
from soulcycle_network.instructors import split_by_weights
from soulcycle_network.instructors import generate_names
from soulcycle_network.instructors import add_bike_supply, validate_cols

def load_studio_data(studio_path: str | Path) -> pd.DataFrame: #function to load the studio data
    studio_path = Path(studio_path)
    if not studio_path.exists():
        raise FileNotFoundError("Input file not found: " + str(studio_path))

    studio_data = pd.read_csv(studio_path) #read the studio data from the file
    validate_cols(studio_data, {"studio_id", "network_market", "local_ridership_cluster", "rides_per_wk_a", "bikes_per_ride_a", "rides_per_wk_b", "bikes_per_ride_b"}, "studio file")

    studio_data = studio_data.copy() #copy the studio data  
    studio_data["network_market"] = studio_data["network_market"].astype(str).str.strip()
    studio_data["local_ridership_cluster"] = studio_data["local_ridership_cluster"].astype(str).str.strip()
    studio_data["studio_id"] = studio_data["studio_id"].astype(str).str.strip()
    studio_data = add_bike_supply(studio_data)

    if (studio_data["weekly_bike_supply"] <= 0).any():
        raise ValueError("Every studio must have positive weekly_bike_supply.")
    return studio_data

def allocate_rider_clusters(n_riders: int, studio_data: pd.DataFrame) -> dict[str, int]:
    if isinstance(n_riders, bool) or not isinstance(n_riders, int):
        raise TypeError("n_riders must be an integer.")
    if not isinstance(studio_data, pd.DataFrame):
        raise TypeError("studio_data must be a pandas DataFrame.")
    if n_riders <= 0:
        raise ValueError("n_riders must be positive.")

    #group the studio data by local rider cluster and sum the weekly bike supply
    w = studio_data.groupby("local_ridership_cluster")["weekly_bike_supply"].sum().sort_index()
    return split_by_weights(n_riders, w) #split the riders by the weights

def cluster_market_map(studio_data: pd.DataFrame) -> dict[str, str]:
    if not isinstance(studio_data, pd.DataFrame):
        raise TypeError("studio_data must be a pandas DataFrame.")

    out: dict[str, str] = {} #dictionary to store the cluster market map
    for cluster, rows in studio_data.groupby("local_ridership_cluster"):
        markets = sorted(rows["network_market"].unique()) #get the unique network markets
        if len(markets) != 1:
            raise ValueError("Cluster '" + str(cluster) + "' spans multiple markets: " + str(markets))
        out[str(cluster)] = str(markets[0])
    return out

def pick_preferred_studios(market: str, home_cluster: str, n_studios: int, studio_data: pd.DataFrame, rng: np.random.Generator) -> list[str]:
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


    market = market.strip()
    home_cluster = home_cluster.strip()
    rows = studio_data.loc[studio_data["network_market"] == market].copy()
    if rows.empty:
        raise ValueError("No studios found for market '" + market + "'.")


    n_studios = min(n_studios, len(rows))
    home_rows = rows.loc[rows["local_ridership_cluster"] == home_cluster].copy()
    if home_rows.empty:
        home_rows = rows.copy()

    #calculate the probability of picking a studio based on the weekly bike supply
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

def pick_preferred_instructors(market: str, studio_ids: list[str], instructors: dict[str, Instructor], rng: np.random.Generator, n_instructors: int) -> list[str]:
    if not isinstance(market, str):
        raise TypeError("market must be a string.")
    if not isinstance(studio_ids, list):
        raise TypeError("studio_ids must be a list.")
    if not isinstance(instructors, dict):
        raise TypeError("instructors must be a dictionary.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")
    if isinstance(n_instructors, bool) or not isinstance(n_instructors, int):
        raise TypeError("n_instructors must be an integer.")
    if n_instructors < 0:
        raise ValueError("n_instructors cannot be negative.")

    market = market.strip()
    studio_set = set(studio_ids)
    candidates: list[str] = []

    for instructor in instructors.values(): #iterate over the instructors
        if instructor.network_market != market:
            continue
        if studio_set.intersection(instructor.regular_studio_assignments): #add the instructor to the candidates if they are assigned to a studio in the market
            candidates.append(instructor.instructor_id)

    if not candidates:
        for instructor in instructors.values(): #iterate over the instructors   
            if instructor.network_market == market: #add the instructor to the candidates if they are assigned to a studio in the market
                candidates.append(instructor.instructor_id)

    if not candidates or n_instructors == 0:
        return [] #return an empty list if there are no candidates or the number of instructors is 0

    n_instructors = min(n_instructors, len(candidates)) #get the number of instructors to pick
    idx = rng.choice(len(candidates), size=n_instructors, replace=False) #pick the instructors
    return [candidates[int(i)] for i in idx] #return the instructors

def draw_preferred_studio_count(rng: np.random.Generator) -> int: #function to draw the preferred studio count
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")

    options = np.array([1, 2, 3], dtype=int)
    p = np.array([0.50, 0.35, 0.15], dtype=float)
    return int(rng.choice(options, p=p))

def draw_preferred_instructor_count(rng: np.random.Generator) -> int: #function to draw the preferred instructor count
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")

    return min(2, int(rng.poisson(1.0))) #return the preferred instructor count

def generate_riders(studio_path: str | Path, instructors: dict[str, Instructor], rng: np.random.Generator, fake: Faker, n_riders: int = TOTAL_SIMULATED_RIDERS) -> dict[str, Rider]: #function to generate the riders
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")
    if not isinstance(fake, Faker):
        raise TypeError("fake must be a Faker object.")
    if not isinstance(instructors, dict):
        raise TypeError("instructors must be a dictionary.")
    if isinstance(n_riders, bool) or not isinstance(n_riders, int):
        raise TypeError("n_riders must be an integer.")
    if n_riders <= 0:
        raise ValueError("n_riders must be positive.")

    studio_data = load_studio_data(studio_path) #load the studio data
    cluster_alloc = allocate_rider_clusters(n_riders, studio_data)
    cluster_markets = cluster_market_map(studio_data) #map the clusters to the markets

    clusters: list[str] = []
    for cluster, count in cluster_alloc.items(): #iterate over the cluster allocations
        clusters.extend([cluster] * count)
    rng.shuffle(clusters) #shuffle the clusters

    names = generate_names(n_riders, fake) #generate the names  
    riders: dict[str, Rider] = {} #dictionary to store the riders

    for i, cluster in enumerate(clusters, start=1):
        rider_id = "R" + str(i).zfill(6) #generate the rider id
        market = cluster_markets[cluster] #get the market for the cluster
        n_studios = draw_preferred_studio_count(rng) #draw the preferred studio count
        studio_ids = pick_preferred_studios(market, cluster, n_studios, studio_data, rng) #pick the preferred studios
        n_instructors = draw_preferred_instructor_count(rng) #draw the preferred instructor count   
        instructor_ids = pick_preferred_instructors(market, studio_ids, instructors, rng, n_instructors)

        riders[rider_id] = Rider(
            rider_id=rider_id,
            rider_name=names[i - 1],
            home_market=market,
            home_cluster=cluster,
            baseline_annual_ride_rate=draw_annual_ride_rate(rng),
            preferred_studio_ids=studio_ids,
            preferred_instructor_ids=instructor_ids,
        )

    if len(riders) != n_riders:
        raise RuntimeError("Generated " + str(len(riders)) + " riders, but expected " + str(n_riders) + ".")

    return riders

def riders_to_dataframe(riders: dict[str, Rider]) -> pd.DataFrame: #function to convert the riders to a dataframe
    if not isinstance(riders, dict):
        raise TypeError("riders must be a dictionary.")

    rows: list[dict[str, object]] = []
    for rider in riders.values(): #iterate over the riders
        if not isinstance(rider, Rider):
            raise TypeError("riders must contain Rider objects.")

        rows.append({ #add the rider to the rows
            "rider_id": rider.rider_id,
            "rider_name": rider.rider_name,
            "home_market": rider.home_market,
            "home_cluster": rider.home_cluster,
            "baseline_annual_ride_rate": rider.baseline_annual_ride_rate,
            "preferred_studio_count": len(rider.preferred_studio_ids),
            "preferred_studio_ids": "; ".join(rider.preferred_studio_ids),
            "preferred_instructor_ids": "; ".join(rider.preferred_instructor_ids),
        })

    df = pd.DataFrame(rows) #convert the rows to a dataframe
    if not df.empty:
        df = df.sort_values(by="rider_id").reset_index(drop=True) #sort the dataframe by the rider id
    return df

def save_riders(riders: dict[str, Rider], output_path: str | Path) -> None:
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".csv":
        raise ValueError("Rider output file must be a CSV.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    riders_to_dataframe(riders).to_csv(output_path, index=False)

