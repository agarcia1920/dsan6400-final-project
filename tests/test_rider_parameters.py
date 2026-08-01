import numpy as np
from soulcycle_network.config import RIDER_FREQUENCY_PARAMETERS
from soulcycle_network.rider import Rider
from soulcycle_network.rider_parameters import draw_annual_ride_rate, draw_weekly_ride_count, estimate_implied_population, simulated_session_capacity, simulation_scale

def test_draw_annual_ride_rate_respects_bounds():
    rng = np.random.default_rng(6400)
    for _ in range(100):
        rate = draw_annual_ride_rate(rng)
        assert RIDER_FREQUENCY_PARAMETERS["minimum"] <= rate <= RIDER_FREQUENCY_PARAMETERS["maximum"]

def test_draw_annual_ride_rate_median_near_five():
    rng = np.random.default_rng(6400)
    rates = [draw_annual_ride_rate(rng) for _ in range(5000)]
    assert 3.0 < float(np.median(rates)) < 8.0
    assert float(np.percentile(rates, 90)) >= 20.0

def test_draw_weekly_ride_count_allows_zero_weeks():
    rider = Rider(
        rider_id="R000001",
        rider_name="Casual Rider",
        home_market="DMV",
        home_cluster="DC-Arlington",
        baseline_annual_ride_rate=2.0,
    )
    rng = np.random.default_rng(6400)
    counts = [draw_weekly_ride_count(rider, rng) for _ in range(200)]

    assert min(counts) == 0
    assert max(counts) <= 7

def test_estimate_implied_population_example():
    implied = estimate_implied_population(10000)
    assert implied == 30333

def test_simulation_scale_and_capacity():
    implied = 200000
    scale = simulation_scale(implied)
    assert scale == 0.05
    assert simulated_session_capacity(59, scale) == 3
