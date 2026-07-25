from soulcycle_network.instructor_assignment import count_overallocated_studios, summarize_studio_capacity_vs_demand
from soulcycle_network.instructor_generator import generate_instructors
from soulcycle_network.instructor_parameters import add_weekly_bike_supply, calculate_tier_behavior_parameters, load_generator_inputs
from soulcycle_network.studio_loader import load_studios
import pandas as pd

def test_load_generator_inputs_adds_weekly_bike_supply(studios_csv_path, active_instructors_csv_path, instructor_sample_csv_path):
    _, _, studio_data, market_counts, tier_parameters = load_generator_inputs(active_instructors_csv_path, instructor_sample_csv_path, studios_csv_path)

    assert "weekly_bike_supply" in studio_data.columns
    assert (studio_data["weekly_bike_supply"] > 0).all()
    assert len(market_counts) > 0
    assert "Mega" in tier_parameters

def test_calculate_tier_behavior_parameters_uses_scheduled_sample(instructor_sample_csv_path):
    sample_data = pd.read_csv(instructor_sample_csv_path)
    parameters = calculate_tier_behavior_parameters(sample_data)

    assert parameters["Mega"]["class_mean"] > 0
    assert parameters["Mega"]["studio_mean"] > 0

def test_generate_instructors_matches_active_market_counts(studios_csv_path, active_instructors_csv_path, instructor_sample_csv_path, rng, fake):
    instructors = generate_instructors(active_instructors_csv_path, instructor_sample_csv_path, studios_csv_path, rng, fake)

    assert len(instructors) == 248
    assert instructors["I0001"].home_cluster is not None
    assert not hasattr(instructors["I0001"], "official_region")

def test_capacity_diagnostics_runs(studios_csv_path, active_instructors_csv_path, instructor_sample_csv_path, rng, fake):
    studios = load_studios(studios_csv_path)
    instructors = generate_instructors(active_instructors_csv_path, instructor_sample_csv_path, studios_csv_path, rng, fake)
    summary = summarize_studio_capacity_vs_demand(instructors, studios)

    assert len(summary) == len(studios)
    assert "available_classes" in summary.columns
    assert "requested_classes" in summary.columns
    assert count_overallocated_studios(instructors, studios) >= 0

def test_add_weekly_bike_supply_matches_manual_calculation(studios_csv_path):
    studio_data = pd.read_csv(studios_csv_path)
    enriched = add_weekly_bike_supply(studio_data)
    first_row = enriched.iloc[0]

    expected = (first_row["rides_per_wk_a"] * first_row["bikes_per_ride_a"]) + (first_row["rides_per_wk_b"] * first_row["bikes_per_ride_b"])
    assert first_row["weekly_bike_supply"] == expected
