import sys

#this file is meant to be run with pytest, not python directly
#from the project root: pytest tests/test_baseline_instructor_schedule.py
if __name__ == "__main__":
    print("Run this file with pytest from the project root:")
    print("  pytest tests/test_baseline_instructor_schedule.py")
    print("Or run the manual check script:")
    print("  python tests/assignment_test.py")
    sys.exit(1)

from soulcycle_network.baseline_instructor_schedule import assign_baseline_slots, validate_baseline
from soulcycle_network.class_slot_builder import create_network_class_slots
from soulcycle_network.config import DAYS_OF_WEEK
from soulcycle_network.instructor_generator import generate_instructors
from soulcycle_network.studio_loader import load_studios
from soulcycle_network.studio_schedule import create_all_weekly_schedules

def test_assign_baseline_slots_fills_every_slot(studios_csv_path, active_instructors_csv_path, instructor_sample_csv_path, rng, fake):
    studios = load_studios(studios_csv_path)
    create_all_weekly_schedules(studios, rng)
    class_slots = create_network_class_slots(studios)
    instructors = generate_instructors(active_instructors_csv_path, instructor_sample_csv_path, studios_csv_path, rng, fake)

    assign_baseline_slots(instructors, class_slots, rng)

    assert all(slot.usual_instructor is not None for slot in class_slots)
    assert len({slot.slot_id for slot in class_slots}) == len(class_slots)
    validate_baseline(instructors, class_slots)

def test_assign_baseline_slots_updates_instructor_fields(studios_csv_path, active_instructors_csv_path, instructor_sample_csv_path, rng, fake):
    studios = load_studios(studios_csv_path)
    create_all_weekly_schedules(studios, rng)
    class_slots = create_network_class_slots(studios)
    instructors = generate_instructors(active_instructors_csv_path, instructor_sample_csv_path, studios_csv_path, rng, fake)

    assign_baseline_slots(instructors, class_slots, rng)

    instructor = instructors["I0001"]
    assert len(instructor.baseline_slot_ids) == instructor.baseline_class_count
    assert sum(instructor.baseline_day_counts.values()) == instructor.baseline_class_count
    assert set(instructor.baseline_day_counts.keys()).issubset(set(DAYS_OF_WEEK))

def test_studio_slot_assignments_match_studio_allocations(studios_csv_path, active_instructors_csv_path, instructor_sample_csv_path, rng, fake):
    studios = load_studios(studios_csv_path)
    create_all_weekly_schedules(studios, rng)
    class_slots = create_network_class_slots(studios)
    instructors = generate_instructors(active_instructors_csv_path, instructor_sample_csv_path, studios_csv_path, rng, fake)
    lookup = {slot.slot_id: slot for slot in class_slots}

    assign_baseline_slots(instructors, class_slots, rng)

    for instructor in instructors.values():
        studio_counts: dict[str, int] = {}
        for slot_id in instructor.baseline_slot_ids:
            sid = lookup[slot_id].studio_id
            studio_counts[sid] = studio_counts.get(sid, 0) + 1
        assert studio_counts == instructor.baseline_studio_allocations
