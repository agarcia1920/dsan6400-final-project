# Generate the synthetic instructor population.

from pathlib import Path
import numpy as np
import pandas as pd
from faker import Faker
from soulcycle_network.instructor import Instructor
from soulcycle_network.instructor_assignment import allocate_classes, allocate_clusters, calibrate_class_loads, init_capacity, market_class_supply, pick_studios
from soulcycle_network.instructor_parameters import MARKET_TO_TIER, draw_class_count, draw_studio_count, load_inputs

def generate_names(n: int, fake: Faker) -> list[str]:
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be an integer.")
    if not isinstance(fake, Faker):
        raise TypeError("fake must be a Faker object.")

    names: list[str] = []
    seen: set[str] = set()

    while len(names) < n:
        candidate = fake.name().strip()
        if candidate and candidate not in seen:
            names.append(candidate)
            seen.add(candidate)

    return names

def generate_instructors(active_path: str | Path, sample_path: str | Path, studio_path: str | Path, rng: np.random.Generator, fake: Faker) -> dict[str, Instructor]:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")
    if not isinstance(fake, Faker):
        raise TypeError("fake must be a Faker object.")

    active_data, sample_data, studio_data, market_counts, tier_params = load_inputs(active_path, sample_path, studio_path)

    total = int(sum(market_counts.values()))
    names = generate_names(total, fake)
    name_iter = iter(names)
    instructors: dict[str, Instructor] = {}
    n = 1
    cap_left = init_capacity(studio_data)

    for market in sorted(market_counts):
        n_market = int(market_counts[market])
        tier = MARKET_TO_TIER[market]
        cluster_alloc = allocate_clusters(market, n_market, studio_data)

        clusters: list[str] = []
        for cluster, count in cluster_alloc.items():
            clusters.extend([cluster] * count)
        rng.shuffle(clusters)

        roster: list[dict[str, str]] = []
        for cluster in clusters:
            roster.append({
                "instructor_id": "I" + str(n).zfill(4),
                "instructor_name": next(name_iter),
                "home_cluster": cluster,
            })
            n += 1

        raw_counts = pd.Series({entry["instructor_id"]: draw_class_count(tier, tier_params, rng) for entry in roster})
        target = market_class_supply(market, studio_data)
        class_counts = calibrate_class_loads(raw_counts, target)
        rng.shuffle(roster)

        for entry in roster:
            iid = entry["instructor_id"]
            n_classes = int(class_counts[iid])
            n_studios = draw_studio_count(tier, n_classes, tier_params, rng)
            n_studios = min(n_studios, n_classes)
            studio_ids = pick_studios(market, entry["home_cluster"], n_studios, studio_data, rng, cap_left)
            alloc = allocate_classes(n_classes, studio_ids, studio_data, rng, market, cap_left)

            instructors[iid] = Instructor(
                instructor_id=iid,
                instructor_name=entry["instructor_name"],
                network_market=market,
                market_tier=tier,
                home_cluster=entry["home_cluster"],
                baseline_class_count=n_classes,
                regular_studio_assignments=list(alloc.keys()),
                baseline_studio_allocations=alloc,
                baseline_day_counts={},
                baseline_slot_ids=[],
            )

    if len(instructors) != total:
        raise RuntimeError("Generated " + str(len(instructors)) + " instructors, but expected " + str(total) + ".")

    got = pd.Series([i.network_market for i in instructors.values()]).value_counts().to_dict()
    if got != market_counts:
        raise RuntimeError("Generated instructor market counts do not match the active instructor population.")

    return instructors

def instructors_to_dataframe(instructors: dict[str, Instructor]) -> pd.DataFrame:
    if not isinstance(instructors, dict):
        raise TypeError("instructors must be a dictionary.")

    rows: list[dict[str, object]] = []

    for instructor in instructors.values():
        if not isinstance(instructor, Instructor):
            raise TypeError("instructors must contain Instructor objects.")

        parts = []
        for sid, n in instructor.baseline_studio_allocations.items():
            parts.append(sid + ":" + str(n))

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

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by="instructor_id").reset_index(drop=True)
    return df

def save_instructors(instructors: dict[str, Instructor], output_path: str | Path) -> None:
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".csv":
        raise ValueError("Instructor output file must be a CSV.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    instructors_to_dataframe(instructors).to_csv(output_path, index=False)
