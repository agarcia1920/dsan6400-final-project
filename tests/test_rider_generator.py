import numpy as np
import pandas as pd
from soulcycle_network.config import TOTAL_SIMULATED_RIDERS
from soulcycle_network.instructor_generator import generate_instructors
from soulcycle_network.rider_generator import allocate_rider_clusters, generate_riders, load_studio_data

def test_allocate_rider_clusters_matches_total(studios_csv_path):
    studio_data = load_studio_data(studios_csv_path)
    alloc = allocate_rider_clusters(TOTAL_SIMULATED_RIDERS, studio_data)
    assert sum(alloc.values()) == TOTAL_SIMULATED_RIDERS

def test_generate_riders_count_and_fields(studios_csv_path, active_instructors_csv_path, instructor_sample_csv_path, rng, fake):
    instructors = generate_instructors(active_instructors_csv_path, instructor_sample_csv_path, studios_csv_path, rng, fake)
    riders = generate_riders(studios_csv_path, instructors, rng, fake, n_riders=200)

    assert len(riders) == 200
    for rider in riders.values():
        assert rider.baseline_annual_ride_rate > 0
        assert rider.home_cluster
        assert rider.home_market
        assert len(rider.preferred_studio_ids) >= 1
        assert rider.preferred_studio_ids[0]

def test_generate_riders_cluster_allocation_follows_bike_supply(studios_csv_path, active_instructors_csv_path, instructor_sample_csv_path, rng, fake):
    studio_data = load_studio_data(studios_csv_path)
    supply = studio_data.groupby("local_ridership_cluster")["weekly_bike_supply"].sum()
    expected_share = supply / supply.sum()

    instructors = generate_instructors(active_instructors_csv_path, instructor_sample_csv_path, studios_csv_path, rng, fake)
    riders = generate_riders(studios_csv_path, instructors, rng, fake, n_riders=5000)

    got = pd.Series([r.home_cluster for r in riders.values()]).value_counts()
    got_share = got / got.sum()

    for cluster in expected_share.index:
        assert abs(float(got_share.get(cluster, 0.0)) - float(expected_share.loc[cluster])) < 0.05

def test_generate_riders_preferred_studios_in_home_market(studios_csv_path, active_instructors_csv_path, instructor_sample_csv_path, rng, fake):
    studio_data = load_studio_data(studios_csv_path)
    market_by_studio = studio_data.set_index("studio_id")["network_market"].to_dict()

    instructors = generate_instructors(active_instructors_csv_path, instructor_sample_csv_path, studios_csv_path, rng, fake)
    riders = generate_riders(studios_csv_path, instructors, rng, fake, n_riders=300)

    for rider in riders.values():
        for sid in rider.preferred_studio_ids:
            assert market_by_studio[sid] == rider.home_market
