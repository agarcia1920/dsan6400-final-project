# Print week-1 realized schedule stats.
# Run from the project root:
#   python tests/weekly_test.py

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
from faker import Faker
from soulcycle_network.baseline_instructor_schedule import assign_baseline_slots
from soulcycle_network.class_slot_builder import create_network_class_slots
from soulcycle_network.config import PROB_OFF_WEEK, RANDOM_SEED
from soulcycle_network.instructor_generator import generate_instructors
from soulcycle_network.studio_loader import load_studios
from soulcycle_network.studio_schedule import create_all_weekly_schedules
from soulcycle_network.weekly_schedule import create_weekly_schedule, summarize_week

def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    fake = Faker("en_US")
    fake.seed_instance(RANDOM_SEED)

    studios = load_studios(DATA_DIR / "studios.csv")
    create_all_weekly_schedules(studios, rng)
    class_slots = create_network_class_slots(studios)
    instructors = generate_instructors(DATA_DIR / "active_instructors_final.csv", DATA_DIR / "instructors_sample.csv", DATA_DIR / "studios.csv", rng, fake)
    assign_baseline_slots(instructors, class_slots, rng)

    week_rng = np.random.default_rng(RANDOM_SEED + 1)
    result = create_weekly_schedule(1, instructors, class_slots, week_rng, PROB_OFF_WEEK)
    summary = summarize_week(result.sessions)

    print("Week 1 sessions:", summary["total_sessions"])
    print("Substitutions:", result.substitution_count)
    print("Off instructors:", len(result.off_instructor_ids))
    print("Uncovered sessions:", len(result.uncovered_session_ids))
    print("Assigned instructors:", summary["unique_assigned_instructors"])

    for s in [x for x in result.sessions if x.is_substitution][:5]:
        print(s.slot_id, "usual", s.usual_instructor_id, "assigned", s.assigned_instructor_id)

if __name__ == "__main__":
    main()
