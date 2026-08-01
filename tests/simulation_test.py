import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from soulcycle_network.simulation import run_default_simulation

def main() -> None:
    ctx, result, summary = run_default_simulation(PROJECT_ROOT / "data")

    print("Simulation complete")
    print("Riders:", len(ctx.riders))
    print("Instructors:", len(ctx.instructors))
    print("Scale:", round(ctx.scale, 4))
    print("Implied population:", ctx.implied_population)
    print()
    print("Summary:")
    for key in sorted(summary):
        print(" ", key + ":", summary[key])

if __name__ == "__main__":
    main()
