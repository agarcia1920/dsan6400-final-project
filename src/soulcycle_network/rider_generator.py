# Generating the synthetic rider population.

from pathlib import Path
import numpy as np
import pandas as pd
from faker import Faker
from soulcycle_network.config import TOTAL_SIMULATED_RIDERS
from soulcycle_network.instructor import Instructor
from soulcycle_network.instructor_assignment import split_by_weights
from soulcycle_network.instructor_generator import generate_names
from soulcycle_network.instructor_parameters import add_bike_supply, validate_cols
from soulcycle_network.rider import Rider
from soulcycle_network.rider_parameters import draw_annual_ride_rate

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
