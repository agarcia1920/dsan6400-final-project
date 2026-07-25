# Print studio capacity vs instructor demand.
# Run from the project root:
#   python tests/demand_test.py

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
from faker import Faker
from soulcycle_network.config import RANDOM_SEED
from soulcycle_network.instructor_assignment import capacity_summary
from soulcycle_network.instructor_generator import generate_instructors
from soulcycle_network.studio_loader import load_studios

def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    fake = Faker("en_US")
    fake.seed_instance(RANDOM_SEED)

    studios = load_studios(DATA_DIR / "studios.csv")
    instructors = generate_instructors(DATA_DIR / "active_instructors_final.csv", DATA_DIR / "instructors_sample.csv", DATA_DIR / "studios.csv", rng, fake)
    summary = capacity_summary(instructors, studios)

    print(summary[summary["overallocated"]])
    print("Total available classes:", summary["available_classes"].sum())
    print("Total requested classes:", summary["requested_classes"].sum())
    print("Overallocated studios:", summary["overallocated"].sum())

if __name__ == "__main__":
    main()
