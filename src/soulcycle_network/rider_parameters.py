# Draw rider frequency parameters and weekly attendance counts.

import numpy as np
from soulcycle_network.config import DAYS_OF_WEEK, MAX_CLASSES_PER_DAY, MEAN_ANNUAL_RIDES, RIDER_FREQUENCY_PARAMETERS, TARGET_OCCUPANCY, TOTAL_SIMULATED_RIDERS
from soulcycle_network.rider import Rider

def draw_annual_ride_rate(rng: np.random.Generator, params: dict[str, float] | None = None) -> float:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")

    p = params if params is not None else RIDER_FREQUENCY_PARAMETERS #get the parameters
    if not isinstance(p, dict):
        raise TypeError("params must be a dictionary.")

    log_mean = float(p["log_mean"]) #get the log mean
    log_sd = float(p["log_sd"]) #get the log standard deviation
    minimum = float(p["minimum"])
    maximum = float(p["maximum"]) #get the maximum

    latent = rng.lognormal(mean=log_mean, sigma=log_sd)
    return float(np.clip(latent, minimum, maximum)) #return the annual ride rate

def draw_weekly_ride_count(rider: Rider, rng: np.random.Generator) -> int:
    if not isinstance(rider, Rider): #validate the input types and values
        raise TypeError("rider must be a Rider object.")
    if not isinstance(rng, np.random.Generator): #validate the input types and values
        raise TypeError("rng must be a NumPy Generator.")

    weekly_rate = rider.baseline_annual_ride_rate / 52.0 #calculate the weekly rate
    weekly_cap = len(DAYS_OF_WEEK) * MAX_CLASSES_PER_DAY
    return min(int(rng.poisson(weekly_rate)), weekly_cap) #return the weekly ride count

def estimate_implied_population(total_weekly_bike_supply: int, target_occupancy: float = TARGET_OCCUPANCY, mean_annual_rides: float = float(MEAN_ANNUAL_RIDES)) -> int:
    if isinstance(total_weekly_bike_supply, bool) or not isinstance(total_weekly_bike_supply, int): 
        raise TypeError("total_weekly_bike_supply must be an integer.")
    if not isinstance(target_occupancy, float):
        raise TypeError("target_occupancy must be a float.")
    if isinstance(mean_annual_rides, bool) or not isinstance(mean_annual_rides, (int, float)):
        raise TypeError("mean_annual_rides must be a number.")
    if total_weekly_bike_supply <= 0:
        raise ValueError("total_weekly_bike_supply must be positive.")
    if target_occupancy <= 0 or target_occupancy > 1:
        raise ValueError("target_occupancy must be between 0 and 1.")
    if mean_annual_rides <= 0:
        raise ValueError("mean_annual_rides must be positive.")

    return int(round((52 * total_weekly_bike_supply * target_occupancy) / mean_annual_rides)) #return the implied population

def simulation_scale(implied_population: int, total_simulated_riders: int = TOTAL_SIMULATED_RIDERS) -> float:
    if isinstance(implied_population, bool) or not isinstance(implied_population, int): 
        raise TypeError("implied_population must be an integer.")
    if isinstance(total_simulated_riders, bool) or not isinstance(total_simulated_riders, int):
        raise TypeError("total_simulated_riders must be an integer.")
    if implied_population <= 0:
        raise ValueError("implied_population must be positive.")
    if total_simulated_riders <= 0:
        raise ValueError("total_simulated_riders must be positive.")

    return float(total_simulated_riders) / float(implied_population) #return the simulation scale

def simulated_session_capacity(real_capacity: int, scale: float) -> int: #function to calculate the simulated session capacity
    if isinstance(real_capacity, bool) or not isinstance(real_capacity, int):
        raise TypeError("real_capacity must be an integer.")
    if not isinstance(scale, float):
        raise TypeError("scale must be a float.")
    if real_capacity <= 0:
        raise ValueError("real_capacity must be positive.")
    if scale <= 0:
        raise ValueError("scale must be positive.")

    return max(1, int(round(real_capacity * scale)))
