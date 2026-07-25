# Estimate instructor behavior from the sample file and load generator inputs.

from pathlib import Path
import numpy as np
import pandas as pd
from soulcycle_network.config import CLASS_LOAD_STUDIO_EFFECT

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

STUDIO_COUNT_BOUNDS = {
    "Mega": (1, 7),
    "Large": (1, 4),
    "Medium": (1, 3),
    "Concentrated": (1, 2),
}

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

def tier_params_from_sample(sample_data: pd.DataFrame) -> dict[str, dict[str, float]]:
    if not isinstance(sample_data, pd.DataFrame):
        raise TypeError("sample_data must be a pandas DataFrame.")

    validate_cols(sample_data, {"market", "classes_jul20_26", "studios_taught_count"}, "instructor sample")

    sample_data = sample_data.copy()
    sample_data["classes_jul20_26"] = pd.to_numeric(sample_data["classes_jul20_26"], errors="raise")
    sample_data["studios_taught_count"] = pd.to_numeric(sample_data["studios_taught_count"], errors="raise")
    sample_data["market_tier"] = sample_data["market"].map(MARKET_TO_TIER)

    if sample_data["market_tier"].isna().any():
        missing = sorted(sample_data.loc[sample_data["market_tier"].isna(), "market"].dropna().unique())
        raise ValueError("The following markets do not have a market-tier mapping: " + str(missing))

    scheduled = sample_data.loc[sample_data["classes_jul20_26"] > 0].copy()
    if scheduled.empty:
        raise ValueError("Instructor sample contains no scheduled instructors.")

    params: dict[str, dict[str, float]] = {}

    for tier, rows in scheduled.groupby("market_tier"):
        class_sd = float(rows["classes_jul20_26"].std(ddof=1))
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
    if not isinstance(tier, str):
        raise TypeError("tier must be a string.")
    if not isinstance(tier_params, dict):
        raise TypeError("tier_params must be a dictionary.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")

    p = tier_params[tier.strip()]
    mean = float(p["class_mean"])
    sd = float(p["class_sd"])
    lo = max(1, int(p["class_min_observed"]) - 1)
    hi = int(p["class_max_observed"]) + 1

    if sd <= 0:
        return int(np.clip(round(mean), lo, hi))

    while True:
        draw = rng.normal(loc=mean, scale=sd)
        if lo <= draw <= hi:
            return int(np.clip(round(draw), lo, hi))

def draw_studio_count(tier: str, n_classes: int, tier_params: dict[str, dict[str, float]], rng: np.random.Generator) -> int:
    if not isinstance(tier, str):
        raise TypeError("tier must be a string.")
    if isinstance(n_classes, bool) or not isinstance(n_classes, int):
        raise TypeError("n_classes must be an integer.")
    if not isinstance(tier_params, dict):
        raise TypeError("tier_params must be a dictionary.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")

    tier = tier.strip()
    p = tier_params[tier]
    class_mean = float(p["class_mean"])
    class_sd = float(p["class_sd"])
    studio_mean = float(p["studio_mean"])
    studio_sd = float(p["studio_sd"])

    if class_sd > 0:
        z = (n_classes - class_mean) / class_sd
    else:
        z = 0.0

    adj_mean = studio_mean + CLASS_LOAD_STUDIO_EFFECT * z
    lo, hi = STUDIO_COUNT_BOUNDS[tier]

    if studio_sd <= 0:
        return int(np.clip(round(adj_mean), lo, hi))

    while True:
        draw = rng.normal(loc=adj_mean, scale=studio_sd)
        if lo <= draw <= hi:
            return int(np.clip(round(draw), lo, hi))

def load_inputs(active_path: str | Path, sample_path: str | Path, studio_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int], dict[str, dict[str, float]]]:
    active_path = Path(active_path)
    sample_path = Path(sample_path)
    studio_path = Path(studio_path)

    for path in (active_path, sample_path, studio_path):
        if not path.exists():
            raise FileNotFoundError("Input file not found: " + str(path))

    active_data = pd.read_csv(active_path)
    sample_data = pd.read_csv(sample_path)
    studio_data = pd.read_csv(studio_path)

    validate_cols(active_data, {"instructor_name", "network_markets", "market_tier"}, "active instructor file")
    validate_cols(sample_data, {"sample_id", "market", "classes_jul20_26", "studios_taught_count"}, "instructor sample file")
    validate_cols(studio_data, {"studio_id", "studio_name", "network_market", "market_tier", "local_ridership_cluster", "rides_per_wk_a", "bikes_per_ride_a", "rides_per_wk_b", "bikes_per_ride_b"}, "studio file")

    active_data = active_data.copy()
    sample_data = sample_data.copy()
    studio_data = studio_data.copy()

    active_data["network_markets"] = active_data["network_markets"].astype(str).str.strip()
    active_data["market_tier"] = active_data["market_tier"].astype(str).str.strip()
    studio_data["network_market"] = studio_data["network_market"].astype(str).str.strip()
    studio_data["market_tier"] = studio_data["market_tier"].astype(str).str.strip()
    studio_data["local_ridership_cluster"] = studio_data["local_ridership_cluster"].astype(str).str.strip()
    studio_data["studio_id"] = studio_data["studio_id"].astype(str).str.strip()
    studio_data = add_bike_supply(studio_data)

    if (studio_data["weekly_bike_supply"] <= 0).any():
        raise ValueError("Every studio must have positive weekly_bike_supply.")
    if studio_data["studio_id"].duplicated().any():
        dups = sorted(studio_data.loc[studio_data["studio_id"].duplicated(keep=False), "studio_id"].unique())
        raise ValueError("Duplicate studio IDs found: " + str(dups))

    market_counts = active_data["network_markets"].value_counts().sort_index().astype(int).to_dict()
    tier_params = tier_params_from_sample(sample_data)

    missing = set(market_counts) - set(studio_data["network_market"])
    if missing:
        raise ValueError("No studios were found for these active markets: " + str(sorted(missing)))

    return active_data, sample_data, studio_data, market_counts, tier_params
