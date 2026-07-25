# Check baseline instructor-to-slot assignment.
# Run from the project root:
#   python tests/assignment_test.py

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
from faker import Faker
from soulcycle_network.baseline_instructor_schedule import assign_baseline_slots, day_load_totals, validate_baseline
from soulcycle_network.class_slot_builder import create_network_class_slots
from soulcycle_network.config import RANDOM_SEED
from soulcycle_network.instructor_generator import generate_instructors
from soulcycle_network.studio_loader import load_studios
from soulcycle_network.studio_schedule import create_all_weekly_schedules

def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    fake = Faker("en_US")
    fake.seed_instance(RANDOM_SEED)

    studios = load_studios(DATA_DIR / "studios.csv")
    create_all_weekly_schedules(studios, rng)
    class_slots = create_network_class_slots(studios)
    instructors = generate_instructors(DATA_DIR / "active_instructors_final.csv", DATA_DIR / "instructors_sample.csv", DATA_DIR / "studios.csv", rng, fake)

    assign_baseline_slots(instructors, class_slots, rng)
    validate_baseline(instructors, class_slots)

    print("Total slots:", len(class_slots))
    print("Total instructors:", len(instructors))
    print("Slots with usual instructor:", sum(1 for slot in class_slots if slot.usual_instructor is not None))
    print("Day load across network:", day_load_totals(instructors))
    print("Sample instructor I0001 slots:", instructors["I0001"].baseline_slot_ids)

if __name__ == "__main__":
    main()
