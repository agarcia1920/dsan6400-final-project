import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from soulcycle_network.simulation import run_default_simulation

DIAGNOSTIC_KEYS = [
    "generated_mean_annual_ride_rate",
    "implied_population",
    "simulation_scale",
    "avg_attendance_per_week",
    "seat_occupancy_rate",
    "total_unmet_demand",
    "familiarity_pair_count",
    "social_tie_pair_count",
    "graph_nodes",
    "graph_edges",
    "mean_degree",
    "largest_connected_component",
]

def main() -> None:
    started = time.perf_counter()
    ctx, result, summary = run_default_simulation(PROJECT_ROOT / "data")
    elapsed = time.perf_counter() - started

    print("Simulation complete")
    print("Riders:", len(ctx.riders))
    print("Instructors:", len(ctx.instructors))
    print("Runtime (seconds):", round(elapsed, 2))
    print()
    print("Calibration diagnostics:")
    for key in DIAGNOSTIC_KEYS:
        print(" ", key + ":", summary.get(key, ctx.generated_mean_annual_ride_rate if key == "generated_mean_annual_ride_rate" else "n/a"))
    print()
    print("Full summary:")
    for key in sorted(summary):
        print(" ", key + ":", summary[key])

if __name__ == "__main__":
    main()
