from soulcycle_network.instructor_assignment import calibrate_class_loads, capacity_summary, count_overallocated
from soulcycle_network.instructor_generator import generate_instructors
from soulcycle_network.instructor_parameters import add_bike_supply, load_inputs, tier_params_from_sample
from soulcycle_network.studio_loader import load_studios
import pandas as pd

def test_load_inputs_adds_bike_supply(studios_csv_path, active_instructors_csv_path, instructor_sample_csv_path):
    _, _, studio_data, market_counts, tier_params = load_inputs(active_instructors_csv_path, instructor_sample_csv_path, studios_csv_path)

    assert "weekly_bike_supply" in studio_data.columns
    assert (studio_data["weekly_bike_supply"] > 0).all()
    assert len(market_counts) > 0
    assert "Mega" in tier_params

def test_tier_params_from_sample_uses_scheduled_sample(instructor_sample_csv_path):
    sample_data = pd.read_csv(instructor_sample_csv_path)
    params = tier_params_from_sample(sample_data)

    assert params["Mega"]["class_mean"] > 0
    assert params["Mega"]["studio_mean"] > 0

def test_generate_instructors_matches_active_market_counts(studios_csv_path, active_instructors_csv_path, instructor_sample_csv_path, rng, fake):
    instructors = generate_instructors(active_instructors_csv_path, instructor_sample_csv_path, studios_csv_path, rng, fake)

    assert len(instructors) == 248
    assert instructors["I0001"].home_cluster is not None
    assert not hasattr(instructors["I0001"], "official_region")

def test_capacity_summary_runs(studios_csv_path, active_instructors_csv_path, instructor_sample_csv_path, rng, fake):
    studios = load_studios(studios_csv_path)
    instructors = generate_instructors(active_instructors_csv_path, instructor_sample_csv_path, studios_csv_path, rng, fake)
    summary = capacity_summary(instructors, studios)

    assert len(summary) == len(studios)
    assert "available_classes" in summary.columns
    assert "requested_classes" in summary.columns
    assert summary["requested_classes"].sum() == summary["available_classes"].sum()
    assert count_overallocated(instructors, studios) == 0

def test_calibrate_class_loads_matches_target_total():
    raw_counts = pd.Series({"I0001": 8.0, "I0002": 6.0, "I0003": 4.0})
    calibrated = calibrate_class_loads(raw_counts, 10)

    assert int(calibrated.sum()) == 10
    assert (calibrated >= 1).all()

def test_generate_instructors_total_classes_match_network_supply(studios_csv_path, active_instructors_csv_path, instructor_sample_csv_path, rng, fake):
    studios = load_studios(studios_csv_path)
    instructors = generate_instructors(active_instructors_csv_path, instructor_sample_csv_path, studios_csv_path, rng, fake)

    total_requested = sum(instructor.baseline_class_count for instructor in instructors.values())
    total_available = sum(studio.weekly_class_count for studio in studios.values())

    assert total_requested == total_available
    assert total_available == 1615

def test_add_bike_supply_matches_manual_calculation(studios_csv_path):
    studio_data = pd.read_csv(studios_csv_path)
    enriched = add_bike_supply(studio_data)
    first_row = enriched.iloc[0]

    expected = (first_row["rides_per_wk_a"] * first_row["bikes_per_ride_a"]) + (first_row["rides_per_wk_b"] * first_row["bikes_per_ride_b"])
    assert first_row["weekly_bike_supply"] == expected
