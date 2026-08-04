# Run the full 52-week rider attendance and network simulation.
#importing the necessary libraries and modules
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from soulcycle_network.instructors import assign_baseline_slots
from soulcycle_network.studios import BaselineClassSlot
from soulcycle_network.studios import create_network_class_slots
from soulcycle_network.config import RANDOM_SEED, TOTAL_SIMULATED_RIDERS, TOTAL_WEEKS
from soulcycle_network.instructors import Instructor
from soulcycle_network.instructors import generate_instructors
from soulcycle_network.network_formation import NetworkState, decay_ties, empty_network, summarize_network, update_from_enrollments
from soulcycle_network.riders import Rider
from soulcycle_network.coordination import plan_coordination
from soulcycle_network.riders import generate_riders, load_studio_data
from soulcycle_network.riders import estimate_implied_population, mean_generated_annual_ride_rate, simulation_scale
from soulcycle_network.studios import Studio
from soulcycle_network.studios import load_studios
from soulcycle_network.studios import create_all_weekly_schedules
from soulcycle_network.weekly import WeeklyBookingResult, book_week, draw_weekly_counts, summarize_booking
from soulcycle_network.weekly import WeeklyScheduleResult, create_weekly_schedule, summarize_weekly_simulation

@dataclass
class WeekResult: #class to store the week result
    week_number: int
    schedule: WeeklyScheduleResult
    booking: WeeklyBookingResult

@dataclass
class SimulationContext: #class to store the simulation context
    studios: dict[str, Studio]
    instructors: dict[str, Instructor]
    baseline_slots: list[BaselineClassSlot]
    riders: dict[str, Rider]
    scale: float
    implied_population: int
    generated_mean_annual_ride_rate: float
    studio_markets: dict[str, str]
    cluster_studios: dict[str, set[str]]

@dataclass
class SimulationResult: #class to store the simulation result
    week_results: list[WeekResult] = field(default_factory=list)
    network_state: NetworkState = field(default_factory=empty_network)
    scale: float = 0.0
    implied_population: int = 0

def build_cluster_studios(studio_data) -> dict[str, set[str]]: #function to build the cluster studios
    out: dict[str, set[str]] = {}
    for _, row in studio_data.iterrows(): #iterate over the studio data
        cluster = str(row["local_ridership_cluster"]).strip()
        sid = str(row["studio_id"]).strip()
        out.setdefault(cluster, set()).add(sid) #add the studio to the cluster
    return out

def init_simulation(studio_path: str | Path, active_path: str | Path, sample_path: str | Path, rng: np.random.Generator, fake, n_riders: int = TOTAL_SIMULATED_RIDERS) -> SimulationContext: #function to initialize the simulation
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")
    if isinstance(n_riders, bool) or not isinstance(n_riders, int):
        raise TypeError("n_riders must be an integer.")
    if n_riders <= 0:
        raise ValueError("n_riders must be positive.")

    studios = load_studios(studio_path) #load the studios
    create_all_weekly_schedules(studios, rng)
    baseline_slots = create_network_class_slots(studios) #create the baseline slots
    instructors = generate_instructors(active_path, sample_path, studio_path, rng, fake)
    assign_baseline_slots(instructors, baseline_slots, rng) #assign the baseline slots to the instructors
    riders = generate_riders(studio_path, instructors, rng, fake, n_riders=n_riders)

    studio_data = load_studio_data(studio_path) #load the studio data
    total_supply = int(studio_data["weekly_bike_supply"].sum())
    generated_mean = mean_generated_annual_ride_rate(riders)
    implied = estimate_implied_population(total_supply, mean_annual_rides=generated_mean) #estimate the implied population
    scale = simulation_scale(implied, n_riders)

    studio_markets = studio_data.set_index("studio_id")["network_market"].astype(str).to_dict()
    cluster_studios = build_cluster_studios(studio_data) #build the cluster studios from the studio data

    return SimulationContext( #return the simulation context
        studios=studios,
        instructors=instructors,
        baseline_slots=baseline_slots,
        riders=riders,
        scale=scale,
        implied_population=implied,
        generated_mean_annual_ride_rate=generated_mean,
        studio_markets=studio_markets,
        cluster_studios=cluster_studios,
    )

def run_week(ctx: SimulationContext, week_number: int, network_state: NetworkState, rng: np.random.Generator) -> WeekResult:
    if not isinstance(ctx, SimulationContext):
        raise TypeError("ctx must be a SimulationContext.")
    if isinstance(week_number, bool) or not isinstance(week_number, int):
        raise TypeError("week_number must be an integer.")
    if not isinstance(network_state, NetworkState):
        raise TypeError("network_state must be a NetworkState.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")
    if week_number <= 0:
        raise ValueError("week_number must be positive.")

    decay_ties(network_state) #decay the ties in the network state prior to running the week

    schedule = create_weekly_schedule(week_number, ctx.instructors, ctx.baseline_slots, rng) #create the weekly schedule
    weekly_counts = draw_weekly_counts(ctx.riders, rng) #draw the weekly counts
    coordination_pairs = plan_coordination(ctx.riders, network_state, weekly_counts, rng) #plan the coordination pairs
    booking = book_week(schedule, ctx.riders, ctx.scale, ctx.studio_markets, ctx.cluster_studios, weekly_counts, coordination_pairs, rng) #book the week
    update_from_enrollments(network_state, booking.enrollments) #update the network state from the bookings

    return WeekResult(week_number=week_number, schedule=schedule, booking=booking) #return the week result

def run_simulation(ctx: SimulationContext, rng: np.random.Generator, n_weeks: int = TOTAL_WEEKS) -> SimulationResult: #function to run the simulation
    if not isinstance(ctx, SimulationContext):
        raise TypeError("ctx must be a SimulationContext.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")
    if isinstance(n_weeks, bool) or not isinstance(n_weeks, int):
        raise TypeError("n_weeks must be an integer.")
    if n_weeks <= 0:
        raise ValueError("n_weeks must be positive.")

    network_state = empty_network()
    week_results: list[WeekResult] = [] #list to store the week results

    for week_number in range(1, n_weeks + 1):
        week_results.append(run_week(ctx, week_number, network_state, rng)) #run the week and add the result to the list

    return SimulationResult(
        week_results=week_results,
        network_state=network_state,
        scale=ctx.scale,
        implied_population=ctx.implied_population,
    )

def summarize_simulation(result: SimulationResult, n_instructors: int, generated_mean_annual_ride_rate: float | None = None) -> dict[str, float]: #function to summarize the simulation
    if not isinstance(result, SimulationResult):
        raise TypeError("result must be a SimulationResult.")
    if isinstance(n_instructors, bool) or not isinstance(n_instructors, int):
        raise TypeError("n_instructors must be an integer.")
    
    #summarize the weekly simulation stats
    schedule_stats = summarize_weekly_simulation([w.schedule for w in result.week_results], n_instructors)
    booking_stats = {
        "total_attendance": float(sum(len(w.booking.records) for w in result.week_results)),
        "total_unmet_demand": float(sum(w.booking.unmet_demand for w in result.week_results)),
        "total_coordinated_bookings": float(sum(w.booking.coordinated_bookings for w in result.week_results)),
    }
    if result.week_results:
        weekly_attendance = [len(w.booking.records) for w in result.week_results]
        booking_stats["avg_attendance_per_week"] = float(np.mean(weekly_attendance))
        booking_stats["max_attendance_in_week"] = float(max(weekly_attendance))
        total_seats = sum(w.booking.total_sim_seats for w in result.week_results)
        filled_seats = sum(w.booking.seats_filled for w in result.week_results)
        booking_stats["seat_occupancy_rate"] = float(filled_seats / total_seats) if total_seats > 0 else 0.0
    #summarize the network stats
    network_stats = summarize_network(result.network_state)
    out = dict(schedule_stats)
    out.update(booking_stats)
    out.update(network_stats)
    out["simulation_scale"] = result.scale
    out["implied_population"] = float(result.implied_population)
    if generated_mean_annual_ride_rate is not None:
        out["generated_mean_annual_ride_rate"] = float(generated_mean_annual_ride_rate)
    return out

def default_paths(data_dir: str | Path) -> tuple[Path, Path, Path]: #function to get the default paths
    data_dir = Path(data_dir)
    return (data_dir / "studios.csv", data_dir / "active_instructors_final.csv", data_dir / "instructors_sample.csv")

def run_default_simulation(data_dir: str | Path, seed: int = RANDOM_SEED, n_weeks: int = TOTAL_WEEKS, n_riders: int = TOTAL_SIMULATED_RIDERS): # function to run the default simulation
    from faker import Faker

    studio_path, active_path, sample_path = default_paths(data_dir)
    rng = np.random.default_rng(seed)
    fake = Faker()
    Faker.seed(seed)

    ctx = init_simulation(studio_path, active_path, sample_path, rng, fake, n_riders=n_riders)
    result = run_simulation(ctx, rng, n_weeks=n_weeks)
    summary = summarize_simulation(result, len(ctx.instructors), ctx.generated_mean_annual_ride_rate)
    return ctx, result, summary
