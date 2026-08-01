import pytest
from soulcycle_network.rider import Rider

def test_rider_initialization():
    rider = Rider(
        rider_id="R000001",
        rider_name="Test Rider",
        home_market="DMV",
        home_cluster="DC-Arlington",
        baseline_annual_ride_rate=27.0,
        preferred_studio_ids=["GTWN", "14TH"],
        preferred_instructor_ids=["I0021"],
    )

    assert rider.rider_id == "R000001"
    assert rider.baseline_annual_ride_rate == 27.0
    assert rider.attended_session_ids == []
    assert rider.attended_instructor_counts == {}
    assert rider.attended_studio_counts == {}

def test_rider_rejects_non_positive_annual_rate():
    with pytest.raises(ValueError):
        Rider(
            rider_id="R000002",
            rider_name="Test Rider",
            home_market="DMV",
            home_cluster="DC-Arlington",
            baseline_annual_ride_rate=0.0,
        )

def test_rider_strips_text_fields():
    rider = Rider(
        rider_id=" R000003 ",
        rider_name=" Test Rider ",
        home_market=" DMV ",
        home_cluster=" DC-Arlington ",
        baseline_annual_ride_rate=12.0,
    )

    assert rider.rider_id == "R000003"
    assert rider.home_market == "DMV"
