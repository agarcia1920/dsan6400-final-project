# Parameter estimation and random draws for synthetic instructor behavior.

from pathlib import Path
import numpy as np
import pandas as pd
from soulcycle_network.config import CLASS_LOAD_STUDIO_EFFECT

#map each network market to its tier
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

#reasonable min and max number of regular studios by tier
STUDIO_COUNT_BOUNDS = {
    "Mega": (1, 7),
    "Large": (1, 4),
    "Medium": (1, 3),
    "Concentrated": (1, 2),
}

def validate_cols(dataframe: pd.DataFrame, required_cols: set[str], file_name: str) -> None:
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")
    if not isinstance(required_cols, set):
        raise TypeError("required_cols must be a set.")
    if not isinstance(file_name, str):
        raise TypeError("file_name must be a string.")

    missing_cols = required_cols - set(dataframe.columns)
    if missing_cols:
        raise ValueError("The following columns are missing from " + file_name + ": " + str(sorted(missing_cols)))

def add_weekly_bike_supply(studio_data: pd.DataFrame) -> pd.DataFrame:
    #weekly bike supply is rides per week times bikes per ride for each room
    studio_data = studio_data.copy()
    studio_data["rides_per_wk_a"] = pd.to_numeric(studio_data["rides_per_wk_a"], errors="raise")
    studio_data["rides_per_wk_b"] = pd.to_numeric(studio_data["rides_per_wk_b"], errors="raise")
    studio_data["bikes_per_ride_a"] = pd.to_numeric(studio_data["bikes_per_ride_a"], errors="raise")
    studio_data["bikes_per_ride_b"] = pd.to_numeric(studio_data["bikes_per_ride_b"], errors="raise")
    studio_data["weekly_bikes_a"] = studio_data["rides_per_wk_a"] * studio_data["bikes_per_ride_a"]
    studio_data["weekly_bikes_b"] = studio_data["rides_per_wk_b"] * studio_data["bikes_per_ride_b"]
    studio_data["weekly_bike_supply"] = studio_data["weekly_bikes_a"] + studio_data["weekly_bikes_b"]
    return studio_data

def calculate_tier_behavior_parameters(sample_data: pd.DataFrame) -> dict[str, dict[str, float]]:
    required_cols = {"market", "classes_jul20_26", "studios_taught_count"}
    validate_cols(dataframe=sample_data, required_cols=required_cols, file_name="instructor sample")

    sample_data = sample_data.copy()
    sample_data["classes_jul20_26"] = pd.to_numeric(sample_data["classes_jul20_26"], errors="raise")
    sample_data["studios_taught_count"] = pd.to_numeric(sample_data["studios_taught_count"], errors="raise")

    if (sample_data["classes_jul20_26"] < 0).any():
        raise ValueError("classes_jul20_26 cannot contain negative values.")
    if (sample_data["studios_taught_count"] < 0).any():
        raise ValueError("studios_taught_count cannot contain negative values.")

    sample_data["market_tier"] = sample_data["market"].map(MARKET_TO_TIER)

    if sample_data["market_tier"].isna().any():
        missing_markets = sorted(sample_data.loc[sample_data["market_tier"].isna(), "market"].dropna().unique())
        raise ValueError("The following markets do not have a market-tier mapping: " + str(missing_markets))

    scheduled_data = sample_data.loc[sample_data["classes_jul20_26"] > 0].copy()
    if scheduled_data.empty:
        raise ValueError("Instructor sample contains no scheduled instructors.")

    parameters: dict[str, dict[str, float]] = {}

    for market_tier, tier_data in scheduled_data.groupby("market_tier"):
        class_sd = float(tier_data["classes_jul20_26"].std(ddof=1))
        studio_sd = float(tier_data["studios_taught_count"].std(ddof=1))
        if np.isnan(class_sd):
            class_sd = 0.0
        if np.isnan(studio_sd):
            studio_sd = 0.0

        parameters[market_tier] = {
            "class_mean": float(tier_data["classes_jul20_26"].mean()),
            "class_sd": class_sd,
            "class_min_observed": int(tier_data["classes_jul20_26"].min()),
            "class_max_observed": int(tier_data["classes_jul20_26"].max()),
            "studio_mean": float(tier_data["studios_taught_count"].mean()),
            "studio_sd": studio_sd,
            "scheduled_sample_size": int(len(tier_data)),
        }

    expected_tiers = set(MARKET_TO_TIER.values())
    missing_tiers = expected_tiers - set(parameters)
    if missing_tiers:
        raise ValueError("The instructor sample does not contain scheduled instructors for tiers: " + str(sorted(missing_tiers)))

    return parameters

def draw_baseline_class_count(market_tier: str, tier_parameters: dict[str, dict[str, float]], rng: np.random.Generator) -> int:
    if not isinstance(market_tier, str):
        raise TypeError("market_tier must be a string.")
    if not isinstance(tier_parameters, dict):
        raise TypeError("tier_parameters must be a dictionary.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")

    market_tier = market_tier.strip()
    if market_tier not in tier_parameters:
        raise ValueError("No behavior parameters found for tier '" + market_tier + "'.")

    parameters = tier_parameters[market_tier]
    required_parameter_keys = {"class_mean", "class_sd", "class_min_observed", "class_max_observed"}
    missing_keys = required_parameter_keys - set(parameters)
    if missing_keys:
        raise ValueError("Parameters for " + market_tier + " are missing: " + str(sorted(missing_keys)))

    mean = float(parameters["class_mean"])
    standard_deviation = float(parameters["class_sd"])
    observed_minimum = int(parameters["class_min_observed"])
    observed_maximum = int(parameters["class_max_observed"])
    lower_bound = max(1, observed_minimum - 1)
    upper_bound = observed_maximum + 1

    if standard_deviation <= 0:
        return int(np.clip(round(mean), lower_bound, upper_bound))

    while True:
        latent_class_count = rng.normal(loc=mean, scale=standard_deviation)
        if lower_bound <= latent_class_count <= upper_bound:
            baseline_class_count = int(round(latent_class_count))
            return int(np.clip(baseline_class_count, lower_bound, upper_bound))

def draw_regular_studio_count(market_tier: str, baseline_class_count: int, tier_parameters: dict[str, dict[str, float]], rng: np.random.Generator) -> int:
    if not isinstance(market_tier, str):
        raise TypeError("market_tier must be a string.")
    if isinstance(baseline_class_count, bool) or not isinstance(baseline_class_count, int):
        raise TypeError("baseline_class_count must be an integer.")
    if baseline_class_count <= 0:
        raise ValueError("baseline_class_count must be positive.")
    if not isinstance(tier_parameters, dict):
        raise TypeError("tier_parameters must be a dictionary.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")

    market_tier = market_tier.strip()
    if market_tier not in tier_parameters:
        raise ValueError("No behavior parameters found for tier '" + market_tier + "'.")
    if market_tier not in STUDIO_COUNT_BOUNDS:
        raise ValueError("No studio-count bounds found for tier '" + market_tier + "'.")

    parameters = tier_parameters[market_tier]
    required_parameter_keys = {"class_mean", "class_sd", "studio_mean", "studio_sd"}
    missing_keys = required_parameter_keys - set(parameters)
    if missing_keys:
        raise ValueError("Parameters for " + market_tier + " are missing: " + str(sorted(missing_keys)))

    class_mean = float(parameters["class_mean"])
    class_sd = float(parameters["class_sd"])
    studio_mean = float(parameters["studio_mean"])
    studio_sd = float(parameters["studio_sd"])

    if class_sd > 0:
        standardized_class_load = (baseline_class_count - class_mean) / class_sd
    else:
        standardized_class_load = 0.0

    class_load_effect = CLASS_LOAD_STUDIO_EFFECT * standardized_class_load
    adjusted_studio_mean = studio_mean + class_load_effect
    lower_bound, upper_bound = STUDIO_COUNT_BOUNDS[market_tier]

    if studio_sd <= 0:
        studio_count = int(round(adjusted_studio_mean))
        return int(np.clip(studio_count, lower_bound, upper_bound))

    while True:
        latent_studio_count = rng.normal(loc=adjusted_studio_mean, scale=studio_sd)
        if lower_bound <= latent_studio_count <= upper_bound:
            studio_count = int(round(latent_studio_count))
            return int(np.clip(studio_count, lower_bound, upper_bound))

def load_generator_inputs(active_instructor_path: str | Path, instructor_sample_path: str | Path, studio_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int], dict[str, dict[str, float]]]:
    active_instructor_path = Path(active_instructor_path)
    instructor_sample_path = Path(instructor_sample_path)
    studio_path = Path(studio_path)

    for path in (active_instructor_path, instructor_sample_path, studio_path):
        if not path.exists():
            raise FileNotFoundError("Input file not found: " + str(path))
        if path.suffix.lower() != ".csv":
            raise ValueError("Input file must be a CSV: " + str(path))

    active_data = pd.read_csv(active_instructor_path)
    sample_data = pd.read_csv(instructor_sample_path)
    studio_data = pd.read_csv(studio_path)

    validate_cols(dataframe=active_data, required_cols={"instructor_name", "network_markets", "market_tier"}, file_name="active instructor file")
    validate_cols(dataframe=sample_data, required_cols={"sample_id", "market", "classes_jul20_26", "studios_taught_count"}, file_name="instructor sample file")
    validate_cols(dataframe=studio_data, required_cols={"studio_id", "studio_name", "network_market", "market_tier", "local_ridership_cluster", "rides_per_wk_a", "bikes_per_ride_a", "rides_per_wk_b", "bikes_per_ride_b"}, file_name="studio file")

    active_data = active_data.copy()
    sample_data = sample_data.copy()
    studio_data = studio_data.copy()

    required_active_fields = {"network_markets", "market_tier"}
    if active_data[list(required_active_fields)].isna().any().any():
        raise ValueError("Active instructor market and tier values cannot be missing.")

    active_data["network_markets"] = active_data["network_markets"].astype(str).str.strip()
    active_data["market_tier"] = active_data["market_tier"].astype(str).str.strip()
    studio_data["network_market"] = studio_data["network_market"].astype(str).str.strip()
    studio_data["market_tier"] = studio_data["market_tier"].astype(str).str.strip()
    studio_data["local_ridership_cluster"] = studio_data["local_ridership_cluster"].astype(str).str.strip()
    studio_data["studio_id"] = studio_data["studio_id"].astype(str).str.strip()

    studio_data = add_weekly_bike_supply(studio_data)

    if studio_data["weekly_bike_supply"].isna().any():
        raise ValueError("Studio weekly_bike_supply values cannot be missing.")
    if (studio_data["weekly_bike_supply"] <= 0).any():
        raise ValueError("Every studio must have positive weekly_bike_supply.")
    if studio_data["studio_id"].duplicated().any():
        duplicate_ids = sorted(studio_data.loc[studio_data["studio_id"].duplicated(keep=False), "studio_id"].unique())
        raise ValueError("Duplicate studio IDs found: " + str(duplicate_ids))

    market_counts = active_data["network_markets"].value_counts().sort_index().astype(int).to_dict()
    tier_parameters = calculate_tier_behavior_parameters(sample_data)

    active_markets = set(market_counts)
    studio_markets = set(studio_data["network_market"])
    missing_studio_markets = active_markets - studio_markets
    if missing_studio_markets:
        raise ValueError("No studios were found for these active markets: " + str(sorted(missing_studio_markets)))

    return active_data, sample_data, studio_data, market_counts, tier_parameters
