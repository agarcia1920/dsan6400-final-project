# Orchestration for generating the synthetic instructor population.

from pathlib import Path
import numpy as np
import pandas as pd
from faker import Faker
from soulcycle_network.instructor import Instructor
from soulcycle_network.instructor_assignment import allocate_classes_across_studios, allocate_home_clusters, choose_regular_studios
from soulcycle_network.instructor_parameters import MARKET_TO_TIER, draw_baseline_class_count, draw_regular_studio_count, load_generator_inputs

def generate_names(num_names: int, fake: Faker) -> list[str]:
    if isinstance(num_names, bool) or not isinstance(num_names, int):
        raise TypeError("num_names must be an integer.")
    if num_names <= 0:
        raise ValueError("num_names must be positive.")
    if not isinstance(fake, Faker):
        raise TypeError("fake must be a Faker object.")

    names: list[str] = []
    seen_names: set[str] = set()

    while len(names) < num_names:
        candidate_name = fake.name().strip()
        if candidate_name and candidate_name not in seen_names:
            names.append(candidate_name)
            seen_names.add(candidate_name)

    return names

def generate_instructors(active_instructor_path: str | Path, instructor_sample_path: str | Path, studio_path: str | Path, rng: np.random.Generator, fake: Faker) -> dict[str, Instructor]:
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

    for market in sorted(market_counts):
        market_instructor_count = int(market_counts[market])
        if market not in MARKET_TO_TIER:
            raise ValueError("No market tier mapping found for '" + market + "'.")

        market_tier = MARKET_TO_TIER[market]
        cluster_allocations = allocate_home_clusters(market=market, instructor_count=market_instructor_count, studio_data=studio_data)

        home_clusters: list[str] = []
        for home_cluster, cluster_count in cluster_allocations.items():
            home_clusters.extend([home_cluster] * cluster_count)
        rng.shuffle(home_clusters)

        for home_cluster in home_clusters:
            instructor_id = "I" + str(instructor_number).zfill(4)
            instructor_name = next(name_iterator)
            baseline_class_count = draw_baseline_class_count(market_tier=market_tier, tier_parameters=tier_parameters, rng=rng)
            regular_studio_count = draw_regular_studio_count(market_tier=market_tier, baseline_class_count=baseline_class_count, tier_parameters=tier_parameters, rng=rng)
            regular_studio_count = min(regular_studio_count, baseline_class_count)
            regular_studio_ids = choose_regular_studios(market=market, home_cluster=home_cluster, requested_studio_count=regular_studio_count, studio_data=studio_data, rng=rng)
            studio_allocations = allocate_classes_across_studios(baseline_class_count=baseline_class_count, regular_studio_ids=regular_studio_ids, studio_data=studio_data, rng=rng)

            instructor = Instructor(
                instructor_id=instructor_id,
                instructor_name=instructor_name,
                network_market=market,
                market_tier=market_tier,
                home_cluster=home_cluster,
                baseline_class_count=baseline_class_count,
                regular_studio_assignments=regular_studio_ids,
                baseline_studio_allocations=studio_allocations,
                baseline_day_counts={},
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
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".csv":
        raise ValueError("Instructor output file must be a CSV.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    instructor_data = instructors_to_dataframe(instructors)
    instructor_data.to_csv(output_path, index=False)
