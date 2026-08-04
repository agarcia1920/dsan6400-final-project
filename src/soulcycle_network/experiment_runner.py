# Runs coordination scenarios over seeds, stores network snapshots, and writes per-seed CSV exports.

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker
from numpy.random import Generator

from soulcycle_network.analysis.metrics import ANALYSIS_SNAPSHOT_WEEKS, assign_activity_frequency_tier, snapshot_metrics_rows
from soulcycle_network.config import RANDOM_SEED, TOTAL_SIMULATED_RIDERS, TOTAL_WEEKS
from soulcycle_network.coordination import plan_coordination
from soulcycle_network.exports import PairWeekTracker, export_seed_outputs, rider_lookup
from soulcycle_network.network_formation import NetworkState, clone_network_state, decay_ties, empty_network, update_from_enrollments
from soulcycle_network.simulation import SimulationContext, SimulationResult, WeekResult, default_paths, init_simulation, summarize_simulation
from soulcycle_network.weekly import book_week, create_weekly_schedule, draw_weekly_counts


@dataclass(frozen=True)
class CoordinationScenario:
    name: str
    max_coordination_partners: int
    max_coordinated_classes: int


SCENARIOS: dict[str, CoordinationScenario] = {
    "no_coordination": CoordinationScenario("no_coordination", 0, 0),
    "baseline": CoordinationScenario("baseline", 2, 2),
    "high_coordination": CoordinationScenario("high_coordination", 4, 4),
}


def run_tracked_simulation(
    ctx: SimulationContext,
    rng: Generator,
    scenario: CoordinationScenario,
    n_weeks: int = TOTAL_WEEKS,
    snapshot_weeks: tuple[int, ...] = ANALYSIS_SNAPSHOT_WEEKS,
) -> tuple[SimulationResult, PairWeekTracker, dict[int, NetworkState]]:
    if n_weeks < 1:
        raise ValueError("n_weeks must be at least 1.")

    active_snapshot_weeks = {week for week in snapshot_weeks if week <= n_weeks}
    network_state = empty_network()
    tracker = PairWeekTracker()
    snapshots: dict[int, NetworkState] = {}
    week_results: list[WeekResult] = []

    for week_number in range(1, n_weeks + 1):
        week_result = _run_week_with_scenario(ctx, week_number, network_state, rng, scenario)
        week_results.append(week_result)
        tracker.update(network_state, week_number)
        if week_number in active_snapshot_weeks:
            snapshots[week_number] = clone_network_state(network_state)

    result = SimulationResult(
        week_results=week_results,
        network_state=network_state,
        scale=ctx.scale,
        implied_population=ctx.implied_population,
    )
    return result, tracker, snapshots


def _run_week_with_scenario(
    ctx: SimulationContext,
    week_number: int,
    network_state: NetworkState,
    rng: Generator,
    scenario: CoordinationScenario,
) -> WeekResult:
    decay_ties(network_state)
    schedule = create_weekly_schedule(week_number, ctx.instructors, ctx.baseline_slots, rng)
    weekly_counts = draw_weekly_counts(ctx.riders, rng)
    coordination_pairs = plan_coordination(
        ctx.riders,
        network_state,
        weekly_counts,
        rng,
        max_partners_per_week=scenario.max_coordination_partners,
        max_coordinated_classes_per_week=scenario.max_coordinated_classes,
    )
    booking = book_week(
        schedule,
        ctx.riders,
        ctx.scale,
        ctx.studio_markets,
        ctx.cluster_studios,
        weekly_counts,
        coordination_pairs,
        rng,
    )
    update_from_enrollments(network_state, booking.enrollments)
    return WeekResult(week_number=week_number, schedule=schedule, booking=booking)


def build_longitudinal_table(
    scenario: str,
    seed: int,
    snapshots: dict[int, NetworkState],
    ctx: SimulationContext,
) -> pd.DataFrame:
    rider_cluster, rider_market = rider_lookup(ctx)
    nodes_frame = pd.DataFrame(
        {
            "rider_id": list(ctx.riders.keys()),
            "baseline_annual_ride_rate": [ctx.riders[rid].baseline_annual_ride_rate for rid in ctx.riders],
        }
    )
    tiered = assign_activity_frequency_tier(nodes_frame)
    tier_map = tiered.set_index("rider_id")["activity_frequency_tier"].astype(str).to_dict()
    baseline_rates = tiered.set_index("rider_id")["baseline_annual_ride_rate"].astype(float).to_dict()
    rows: list[dict[str, object]] = []
    for week in sorted(snapshots):
        rows.extend(
            snapshot_metrics_rows(
                snapshots[week],
                scenario=scenario,
                seed=seed,
                week=week,
                rider_cluster=rider_cluster,
                rider_market=rider_market,
                tier_map=tier_map,
                baseline_rates=baseline_rates,
            )
        )
    return pd.DataFrame(rows)


def run_experiment_seed(
    data_dir: str | Path,
    scenario: CoordinationScenario,
    seed: int,
    output_dir: str | Path,
    n_weeks: int = TOTAL_WEEKS,
    n_riders: int = TOTAL_SIMULATED_RIDERS,
) -> pd.DataFrame:
    studio_path, active_path, sample_path = default_paths(data_dir)
    rng = np.random.default_rng(seed)
    fake = Faker()
    Faker.seed(seed)
    started = time.perf_counter()

    ctx = init_simulation(studio_path, active_path, sample_path, rng, fake, n_riders=n_riders)
    result, tracker, snapshots = run_tracked_simulation(ctx, rng, scenario, n_weeks=n_weeks)
    summary = summarize_simulation(result, len(ctx.instructors), ctx.generated_mean_annual_ride_rate)
    summary["runtime_seconds"] = time.perf_counter() - started
    longitudinal = build_longitudinal_table(scenario.name, seed, snapshots, ctx)
    export_seed_outputs(output_dir, scenario.name, seed, ctx, result, tracker, summary, longitudinal)
    return longitudinal


def run_full_experiment(
    data_dir: str | Path,
    output_dir: str | Path,
    seeds: list[int] | None = None,
    scenarios: dict[str, CoordinationScenario] | None = None,
    n_weeks: int = TOTAL_WEEKS,
    n_riders: int = TOTAL_SIMULATED_RIDERS,
) -> pd.DataFrame:
    seed_list = seeds if seeds is not None else [RANDOM_SEED + i for i in range(10)]
    scenario_map = scenarios if scenarios is not None else SCENARIOS
    if not seed_list:
        raise ValueError("At least one seed is required.")
    if not scenario_map:
        raise ValueError("At least one scenario is required.")

    frames: list[pd.DataFrame] = []
    for scenario in scenario_map.values():
        for seed in seed_list:
            frames.append(
                run_experiment_seed(data_dir, scenario, seed, output_dir, n_weeks=n_weeks, n_riders=n_riders)
            )

    master = pd.concat(frames, ignore_index=True)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    master_path = output_root / "longitudinal_metrics_master.csv"
    master.to_csv(master_path, index=False)
    return master


def load_master_table(output_dir: str | Path) -> pd.DataFrame:
    # Some notebooks import load_master_table from experiment_runner instead of analysis.io.
    from soulcycle_network.analysis.io import load_master_table as _load

    return _load(output_dir)
