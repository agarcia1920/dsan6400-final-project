import numpy as np
from soulcycle_network.baseline_instructor_schedule import assign_baseline_slots
from soulcycle_network.class_slot_builder import create_network_class_slots
from soulcycle_network.config import MAX_WEEKLY_DEVIATION, PROB_OFF_WEEK
from soulcycle_network.instructor_generator import generate_instructors
from soulcycle_network.studio_loader import load_studios
from soulcycle_network.studio_schedule import create_all_weekly_schedules
from soulcycle_network.weekly_class_session import WeeklyClassSession
from soulcycle_network.weekly_schedule import create_weekly_class_sessions, draw_off_instructors, pick_sub, summarize_week

def test_weekly_class_session_requires_substitution_flag():
    session = WeeklyClassSession(
        week_number=1,
        slot_id="GTWN_MON_A_01",
        studio_id="GTWN",
        day_of_week="Monday",
        daily_slot_index=1,
        room="A",
        capacity=59,
        usual_instructor_id="I0001",
        assigned_instructor_id="I0002",
        is_substitution=True,
    )

    assert session.is_substitution is True
    assert session.assigned_instructor_id == "I0002"

def test_create_weekly_class_sessions_matches_baseline_slot_count(studios_csv_path, active_instructors_csv_path, instructor_sample_csv_path, rng, fake):
    studios = load_studios(studios_csv_path)
    create_all_weekly_schedules(studios, rng)
    class_slots = create_network_class_slots(studios)
    instructors = generate_instructors(active_instructors_csv_path, instructor_sample_csv_path, studios_csv_path, rng, fake)
    assign_baseline_slots(instructors, class_slots, rng)

    sessions = create_weekly_class_sessions(1, instructors, class_slots, rng, prob_off=0.0)

    assert len(sessions) == len(class_slots)
    assert summarize_week(sessions)["substitutions"] == 0
    assert all(s.assigned_instructor_id == s.usual_instructor_id for s in sessions)

def test_pick_sub_prefers_same_market(studios_csv_path, active_instructors_csv_path, instructor_sample_csv_path, rng, fake):
    studios = load_studios(studios_csv_path)
    create_all_weekly_schedules(studios, rng)
    class_slots = create_network_class_slots(studios)
    instructors = generate_instructors(active_instructors_csv_path, instructor_sample_csv_path, studios_csv_path, rng, fake)
    assign_baseline_slots(instructors, class_slots, rng)

    slot = class_slots[0]
    lookup = {s.slot_id: s for s in class_slots}
    off_ids = {slot.usual_instructor}
    counts = {iid: len(instructor.baseline_slot_ids) for iid, instructor in instructors.items()}
    busy_times = {iid: set() for iid in instructors}
    for instructor in instructors.values():
        for slot_id in instructor.baseline_slot_ids:
            s = lookup[slot_id]
            busy_times[instructor.instructor_id].add((s.day_of_week, s.daily_slot_index))

    sub_id = pick_sub(slot.studio_id, instructors[slot.usual_instructor].network_market, slot.day_of_week, slot.daily_slot_index, instructors, off_ids, counts, busy_times, rng)

    assert sub_id != slot.usual_instructor
    assert instructors[sub_id].network_market == instructors[slot.usual_instructor].network_market

def test_draw_off_instructors_near_expected_rate(studios_csv_path, active_instructors_csv_path, instructor_sample_csv_path, rng, fake):
    instructors = generate_instructors(active_instructors_csv_path, instructor_sample_csv_path, studios_csv_path, rng, fake)
    trial_rng = np.random.default_rng(6400)
    rates = []

    for _ in range(100):
        off_ids = draw_off_instructors(instructors, trial_rng, PROB_OFF_WEEK)
        rates.append(len(off_ids) / len(instructors))

    assert 0.04 < float(np.mean(rates)) < 0.12

def test_substitute_load_stays_within_max_deviation(studios_csv_path, active_instructors_csv_path, instructor_sample_csv_path, rng, fake):
    studios = load_studios(studios_csv_path)
    create_all_weekly_schedules(studios, rng)
    class_slots = create_network_class_slots(studios)
    instructors = generate_instructors(active_instructors_csv_path, instructor_sample_csv_path, studios_csv_path, rng, fake)
    assign_baseline_slots(instructors, class_slots, rng)

    sessions = create_weekly_class_sessions(1, instructors, class_slots, np.random.default_rng(12345), PROB_OFF_WEEK)
    counts: dict[str, int] = {}

    for session in sessions:
        counts[session.assigned_instructor_id] = counts.get(session.assigned_instructor_id, 0) + 1

    for iid, n in counts.items():
        assert n <= instructors[iid].baseline_class_count + MAX_WEEKLY_DEVIATION
