# Functions for generating fictitious instructor objects.
# We use the real active roster to preserve market-level population counts,
# the anonymized instructor sample to estimate teaching behavior,
# and the studio file to place instructors in clusters and studios.

from pathlib import Path
import numpy as np
import pandas as pd
from faker import Faker
from soulcycle_network.instructor import Instructor

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
#mega markets have more studios so instructors can teach at more of them
STUDIO_COUNT_BOUNDS = {
    "Mega": (1, 7),
    "Large": (1, 4),
    "Medium": (1, 3),
    "Concentrated": (1, 2),
}

def validate_cols(dataframe: pd.DataFrame, required_cols: set[str], file_name: str) -> None:
    #check that the required columns are present in a dataframe
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")
    if not isinstance(required_cols, set):
        raise TypeError("required_cols must be a set.")
    if not isinstance(file_name, str):
        raise TypeError("file_name must be a string.")

    missing_cols = required_cols - set(dataframe.columns)
    if missing_cols:
        raise ValueError("The following columns are missing from " + file_name + ": " + str(sorted(missing_cols)))

def largest_remainder_allocation(total: int, weights: pd.Series) -> dict[str, int]:
    #split an integer total across categories proportionally to their weights
    #this keeps the counts as whole numbers and makes sure they still add up to the total
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

    #first give each category its floor share
    exact_allocations = (numeric_weights / numeric_weights.sum()) * total
    integer_allocations = np.floor(exact_allocations).astype(int)
    remaining = total - int(integer_allocations.sum())

    #then hand out the leftover units to whoever had the biggest remainders
    remainders = (exact_allocations - integer_allocations).sort_values(ascending=False)
    for category in remainders.index[:remaining]:
        integer_allocations.loc[category] += 1

    if int(integer_allocations.sum()) != total:
        raise RuntimeError("Largest-remainder allocation did not preserve the total.")

    return integer_allocations.to_dict()

def generate_names(num_names: int, fake: Faker) -> list[str]:
    #generate unique fake names for the synthetic instructors
    if isinstance(num_names, bool) or not isinstance(num_names, int):
        raise TypeError("num_names must be an integer.")
    if num_names <= 0:
        raise ValueError("num_names must be positive.")
    if not isinstance(fake, Faker):
        raise TypeError("fake must be a Faker object.")

    names: list[str] = []
    seen_names: set[str] = set()

    #keep drawing names until we have enough unique ones
    while len(names) < num_names:
        candidate_name = fake.name().strip()
        if candidate_name and candidate_name not in seen_names:
            names.append(candidate_name)
            seen_names.add(candidate_name)

    return names

def calculate_tier_behavior_parameters(sample_data: pd.DataFrame) -> dict[str, dict[str, float]]:
    #estimate what a normal instructor looks like in each market tier
    #we only use instructors who actually had classes that week
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

    #ignore instructors with zero classes that week
    scheduled_data = sample_data.loc[sample_data["classes_jul20_26"] > 0].copy()
    if scheduled_data.empty:
        raise ValueError("Instructor sample contains no scheduled instructors.")

    parameters: dict[str, dict[str, float]] = {}

    for market_tier, tier_data in scheduled_data.groupby("market_tier"):
        class_sd = float(tier_data["classes_jul20_26"].std(ddof=1))
        studio_sd = float(tier_data["studios_taught_count"].std(ddof=1))

        #if there is only one person in a tier, std comes back as nan so we set it to 0
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

    #make sure we got parameters for every tier we expect to simulate
    expected_tiers = set(MARKET_TO_TIER.values())
    missing_tiers = expected_tiers - set(parameters)
    if missing_tiers:
        raise ValueError("The instructor sample does not contain scheduled instructors for tiers: " + str(sorted(missing_tiers)))

    return parameters

def draw_baseline_class_count(market_tier: str, tier_parameters: dict[str, dict[str, float]], rng: np.random.Generator) -> int:
    #draw one instructor's normal weekly class load from a tier-specific normal distribution
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

    #allow a little wiggle room beyond what we saw in the sample
    lower_bound = max(1, observed_minimum - 1)
    upper_bound = observed_maximum + 1

    #if there is no variation in the sample, just use the mean
    if standard_deviation <= 0:
        return int(np.clip(round(mean), lower_bound, upper_bound))

    #keep drawing until we get a value inside the allowed range
    while True:
        latent_class_count = rng.normal(loc=mean, scale=standard_deviation)
        if lower_bound <= latent_class_count <= upper_bound:
            baseline_class_count = int(round(latent_class_count))
            return int(np.clip(baseline_class_count, lower_bound, upper_bound))

def draw_regular_studio_count(market_tier: str, baseline_class_count: int, tier_parameters: dict[str, dict[str, float]], rng: np.random.Generator) -> int:
    #draw how many regular studios this instructor teaches at
    #heavier class loads tend to mean more studios
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

    #see whether this instructor is above or below the tier average for classes
    if class_sd > 0:
        standardized_class_load = (baseline_class_count - class_mean) / class_sd
    else:
        standardized_class_load = 0.0

    #shift the expected studio count up or down based on class load
    class_load_effect = 0.5 * standardized_class_load
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
    #load and validate all the input files we need to generate instructors
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
    validate_cols(dataframe=studio_data, required_cols={"studio_id", "studio_name", "network_market", "market_tier", "local_ridership_cluster", "slots_per_wk"}, file_name="studio file")

    active_data = active_data.copy()
    sample_data = sample_data.copy()
    studio_data = studio_data.copy()

    required_active_fields = {"network_markets", "market_tier"}
    if active_data[list(required_active_fields)].isna().any().any():
        raise ValueError("Active instructor market and tier values cannot be missing.")

    #clean up string fields so whitespace does not cause mismatches later
    active_data["network_markets"] = active_data["network_markets"].astype(str).str.strip()
    active_data["market_tier"] = active_data["market_tier"].astype(str).str.strip()
    studio_data["network_market"] = studio_data["network_market"].astype(str).str.strip()
    studio_data["market_tier"] = studio_data["market_tier"].astype(str).str.strip()
    studio_data["local_ridership_cluster"] = studio_data["local_ridership_cluster"].astype(str).str.strip()
    studio_data["studio_id"] = studio_data["studio_id"].astype(str).str.strip()
    studio_data["slots_per_wk"] = pd.to_numeric(studio_data["slots_per_wk"], errors="raise")

    if studio_data["slots_per_wk"].isna().any():
        raise ValueError("Studio slots_per_wk values cannot be missing.")
    if (studio_data["slots_per_wk"] <= 0).any():
        raise ValueError("Every studio must have positive slots_per_wk.")
    if studio_data["studio_id"].duplicated().any():
        duplicate_ids = sorted(studio_data.loc[studio_data["studio_id"].duplicated(keep=False), "studio_id"].unique())
        raise ValueError("Duplicate studio IDs found: " + str(duplicate_ids))

    #count how many instructors we need in each market from the active roster
    market_counts = active_data["network_markets"].value_counts().sort_index().astype(int).to_dict()
    tier_parameters = calculate_tier_behavior_parameters(sample_data)

    #make sure every market with instructors also has studios in the data
    active_markets = set(market_counts)
    studio_markets = set(studio_data["network_market"])
    missing_studio_markets = active_markets - studio_markets
    if missing_studio_markets:
        raise ValueError("No studios were found for these active markets: " + str(sorted(missing_studio_markets)))

    return active_data, sample_data, studio_data, market_counts, tier_parameters

def allocate_home_clusters(market: str, instructor_count: int, studio_data: pd.DataFrame) -> dict[str, int]:
    #split a market's instructors across home clusters
    #bigger clusters with more weekly slots get more instructors
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

    cluster_weights = market_studios.groupby("local_ridership_cluster")["slots_per_wk"].sum().sort_index()
    return largest_remainder_allocation(total=instructor_count, weights=cluster_weights)

def choose_regular_studios(market: str, home_cluster: str, requested_studio_count: int, studio_data: pd.DataFrame, rng: np.random.Generator) -> list[str]:
    #pick the studios where this instructor regularly teaches
    #the first studio always comes from their home cluster
    #any extra studios come from the rest of the market
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

    #pick one home-cluster studio, weighted by weekly slot supply
    home_probabilities = home_cluster_studios["slots_per_wk"] / home_cluster_studios["slots_per_wk"].sum()
    selected_home_index = rng.choice(home_cluster_studios.index.to_numpy(), p=home_probabilities.to_numpy())
    selected_studio_ids = [str(home_cluster_studios.loc[selected_home_index, "studio_id"])]

    remaining_count = requested_studio_count - 1
    if remaining_count <= 0:
        return selected_studio_ids

    #if they teach at more than one studio, pick the rest from the wider market
    remaining_studios = market_studios.loc[~market_studios["studio_id"].isin(selected_studio_ids)].copy()
    remaining_count = min(remaining_count, len(remaining_studios))
    if remaining_count <= 0:
        return selected_studio_ids

    remaining_probabilities = remaining_studios["slots_per_wk"] / remaining_studios["slots_per_wk"].sum()
    selected_indices = rng.choice(remaining_studios.index.to_numpy(), size=remaining_count, replace=False, p=remaining_probabilities.to_numpy())
    selected_studio_ids.extend(remaining_studios.loc[selected_indices, "studio_id"].astype(str).tolist())

    return selected_studio_ids

def allocate_classes_across_studios(baseline_class_count: int, regular_studio_ids: list[str], studio_data: pd.DataFrame, rng: np.random.Generator) -> dict[str, int]:
    #split an instructor's weekly classes across their regular studios
    #every regular studio gets at least one class
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

    #you cannot teach one class at more studios than you have classes
    if len(regular_studio_ids) > baseline_class_count:
        regular_studio_ids = regular_studio_ids[:baseline_class_count]

    #start by giving each studio one class
    allocation = {studio_id: 1 for studio_id in regular_studio_ids}
    remaining_classes = baseline_class_count - len(regular_studio_ids)
    if remaining_classes <= 0:
        return allocation

    #distribute the leftover classes randomly, weighted by studio slot supply
    selected_studios = studio_data.set_index("studio_id").loc[regular_studio_ids]
    weights = selected_studios["slots_per_wk"].astype(float)
    probabilities = (weights / weights.sum()).to_numpy()
    extra_allocations = rng.multinomial(n=remaining_classes, pvals=probabilities)

    for studio_id, extra_classes in zip(regular_studio_ids, extra_allocations):
        allocation[studio_id] += int(extra_classes)

    if sum(allocation.values()) != baseline_class_count:
        raise RuntimeError("Studio class allocation did not preserve baseline_class_count.")

    return allocation

def generate_instructors(active_instructor_path: str | Path, instructor_sample_path: str | Path, studio_path: str | Path, rng: np.random.Generator, fake: Faker) -> dict[str, Instructor]:
    #generate the full synthetic instructor population
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")
    if not isinstance(fake, Faker):
        raise TypeError("fake must be a Faker object.")

    active_data, sample_data, studio_data, market_counts, tier_parameters = load_generator_inputs(active_instructor_path=active_instructor_path, instructor_sample_path=instructor_sample_path, studio_path=studio_path)

    total_instructors = int(sum(market_counts.values()))
    fictitious_names = generate_names(num_names=total_instructors, fake=fake)
    name_iterator = iter(fictitious_names)
    instructors: dict[str, Instructor] = {}
    instructor_number = 1

    #work market by market so the population counts match the real roster
    for market in sorted(market_counts):
        market_instructor_count = int(market_counts[market])
        if market not in MARKET_TO_TIER:
            raise ValueError("No market tier mapping found for '" + market + "'.")

        market_tier = MARKET_TO_TIER[market]
        cluster_allocations = allocate_home_clusters(market=market, instructor_count=market_instructor_count, studio_data=studio_data)

        #expand the cluster allocation into a flat list of home clusters
        home_clusters: list[str] = []
        for home_cluster, cluster_count in cluster_allocations.items():
            home_clusters.extend([home_cluster] * cluster_count)
        rng.shuffle(home_clusters) #shuffle so cluster order does not follow a fixed pattern

        for home_cluster in home_clusters:
            instructor_id = "I" + str(instructor_number).zfill(4)
            instructor_name = next(name_iterator)
            baseline_class_count = draw_baseline_class_count(market_tier=market_tier, tier_parameters=tier_parameters, rng=rng)
            regular_studio_count = draw_regular_studio_count(market_tier=market_tier, baseline_class_count=baseline_class_count, tier_parameters=tier_parameters, rng=rng)
            regular_studio_count = min(regular_studio_count, baseline_class_count) #cannot teach at more studios than you have classes
            regular_studio_ids = choose_regular_studios(market=market, home_cluster=home_cluster, requested_studio_count=regular_studio_count, studio_data=studio_data, rng=rng)
            studio_allocations = allocate_classes_across_studios(baseline_class_count=baseline_class_count, regular_studio_ids=regular_studio_ids, studio_data=studio_data, rng=rng)

            instructor = Instructor(
                instructor_id=instructor_id,
                instructor_name=instructor_name,
                official_region=home_cluster, #for synthetic instructors we use home cluster here instead of a formal region
                network_market=market,
                market_tier=market_tier,
                home_cluster=home_cluster,
                baseline_class_count=baseline_class_count,
                regular_studio_assignments=regular_studio_ids,
                baseline_studio_allocations=studio_allocations,
                baseline_day_counts={}, #will fill these in when we assign days and slots
                baseline_slot_ids=[],
            )

            if instructor_id in instructors:
                raise RuntimeError("Duplicate instructor ID generated: " + instructor_id)

            instructors[instructor_id] = instructor
            instructor_number += 1

    if len(instructors) != total_instructors:
        raise RuntimeError("Generated " + str(len(instructors)) + " instructors, but expected " + str(total_instructors) + ".")

    generated_market_counts = pd.Series([instructor.network_market for instructor in instructors.values()]).value_counts().to_dict()
    if generated_market_counts != market_counts:
        raise RuntimeError("Generated instructor market counts do not match the active instructor population.")

    return instructors

def instructors_to_dataframe(instructors: dict[str, Instructor]) -> pd.DataFrame:
    #convert Instructor objects into a flat dataframe for saving or inspection
    if not isinstance(instructors, dict):
        raise TypeError("instructors must be a dictionary.")

    rows: list[dict[str, object]] = []

    for instructor_id, instructor in instructors.items():
        if not isinstance(instructor, Instructor):
            raise TypeError("Value stored under " + instructor_id + " must be an Instructor object.")

        allocation_parts = []
        for studio_id, class_count in instructor.baseline_studio_allocations.items():
            allocation_parts.append(studio_id + ":" + str(class_count))

        rows.append({
            "instructor_id": instructor.instructor_id,
            "instructor_name": instructor.instructor_name,
            "network_market": instructor.network_market,
            "market_tier": instructor.market_tier,
            "home_cluster": instructor.home_cluster,
            "baseline_class_count": instructor.baseline_class_count,
            "regular_studio_count": len(instructor.regular_studio_assignments),
            "regular_studio_ids": "; ".join(instructor.regular_studio_assignments),
            "baseline_studio_allocations": "; ".join(allocation_parts),
        })

    instructor_data = pd.DataFrame(rows)
    if not instructor_data.empty:
        instructor_data = instructor_data.sort_values(by="instructor_id").reset_index(drop=True)

    return instructor_data

def save_instructors(instructors: dict[str, Instructor], output_path: str | Path) -> None:
    #save generated instructor data to a csv file
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".csv":
        raise ValueError("Instructor output file must be a CSV.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    instructor_data = instructors_to_dataframe(instructors)
    instructor_data.to_csv(output_path, index=False)
