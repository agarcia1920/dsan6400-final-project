# Turn baseline slots into one realized week of classes.
# Off instructors skip their classes; substitutes in the same market cover those slots.

import numpy as np
from soulcycle_network.baseline_class_slot import BaselineClassSlot
from soulcycle_network.config import MAX_WEEKLY_DEVIATION, PROB_OFF_WEEK
from soulcycle_network.instructor import Instructor
from soulcycle_network.weekly_class_session import WeeklyClassSession

MAX_OFF_TRIES = 100

def draw_off_instructors(instructors: dict[str, Instructor], rng: np.random.Generator, prob_off: float = PROB_OFF_WEEK) -> set[str]:
    if not isinstance(instructors, dict):
        raise TypeError("instructors must be a dictionary.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")
    if not isinstance(prob_off, float):
        raise TypeError("prob_off must be a float.")
    if prob_off < 0 or prob_off > 1:
        raise ValueError("prob_off must be between 0 and 1.")

    off_ids: set[str] = set()
    for iid in instructors:
        if rng.random() < prob_off:
            off_ids.add(iid)
    return off_ids

def sub_cap_left(instructor: Instructor, counts: dict[str, int]) -> int:
    if instructor.baseline_class_count is None:
        raise ValueError("Instructor " + instructor.instructor_id + " must have a baseline_class_count.")
    limit = instructor.baseline_class_count + MAX_WEEKLY_DEVIATION
    return limit - counts.get(instructor.instructor_id, 0)

def pick_sub(studio_id: str, market: str, instructors: dict[str, Instructor], off_ids: set[str], counts: dict[str, int], rng: np.random.Generator) -> str:
    if not isinstance(studio_id, str):
        raise TypeError("studio_id must be a string.")
    if not isinstance(market, str):
        raise TypeError("market must be a string.")
    if not isinstance(instructors, dict):
        raise TypeError("instructors must be a dictionary.")
    if not isinstance(off_ids, set):
        raise TypeError("off_ids must be a set.")
    if not isinstance(counts, dict):
        raise TypeError("counts must be a dictionary.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")

    studio_id = studio_id.strip()
    market = market.strip()

    preferred: list[Instructor] = []
    pool: list[Instructor] = []

    for iid, instructor in instructors.items():
        if iid in off_ids:
            continue
        if instructor.network_market != market:
            continue
        if sub_cap_left(instructor, counts) <= 0:
            continue
        pool.append(instructor)
        if studio_id in instructor.regular_studio_assignments:
            preferred.append(instructor)

    candidates = preferred if preferred else pool
    if not candidates:
        raise RuntimeError("No substitute available for " + studio_id + " in " + market + ".")

    weights = np.array([sub_cap_left(i, counts) for i in candidates], dtype=float)
    pick = int(rng.choice(len(candidates), p=weights / weights.sum()))
    return candidates[pick].instructor_id

def make_session(week: int, slot: BaselineClassSlot, assigned_id: str, is_sub: bool) -> WeeklyClassSession:
    return WeeklyClassSession(
        week_number=week,
        slot_id=slot.slot_id,
        studio_id=slot.studio_id,
        day_of_week=slot.day_of_week,
        room=slot.room,
        capacity=slot.capacity,
        usual_instructor_id=slot.usual_instructor,
        assigned_instructor_id=assigned_id,
        is_substitution=is_sub,
    )

def create_weekly_class_sessions(week: int, instructors: dict[str, Instructor], class_slots: list[BaselineClassSlot], rng: np.random.Generator, prob_off: float = PROB_OFF_WEEK) -> list[WeeklyClassSession]:
    if isinstance(week, bool) or not isinstance(week, int):
        raise TypeError("week must be an integer.")
    if week <= 0:
        raise ValueError("week must be positive.")
    if not isinstance(instructors, dict):
        raise TypeError("instructors must be a dictionary.")
    if not isinstance(class_slots, list):
        raise TypeError("class_slots must be a list.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")

    for attempt in range(MAX_OFF_TRIES):
        off_ids = draw_off_instructors(instructors, rng, prob_off)
        counts = {iid: 0 for iid in instructors}
        by_slot: dict[str, WeeklyClassSession] = {}

        for slot in class_slots:
            if not isinstance(slot, BaselineClassSlot):
                raise TypeError("class_slots must contain BaselineClassSlot objects.")
            if slot.usual_instructor in off_ids:
                continue
            counts[slot.usual_instructor] += 1
            by_slot[slot.slot_id] = make_session(week, slot, slot.usual_instructor, False)

        sub_slots = [slot for slot in class_slots if slot.usual_instructor in off_ids]
        rng.shuffle(sub_slots)

        try:
            for slot in sub_slots:
                market = instructors[slot.usual_instructor].network_market
                sub_id = pick_sub(slot.studio_id, market, instructors, off_ids, counts, rng)
                counts[sub_id] += 1
                by_slot[slot.slot_id] = make_session(week, slot, sub_id, True)
        except RuntimeError:
            if attempt == MAX_OFF_TRIES - 1:
                raise RuntimeError("Could not build a feasible weekly schedule after " + str(MAX_OFF_TRIES) + " tries.")
            continue

        sessions = [by_slot[slot.slot_id] for slot in class_slots]
        validate_week(sessions, class_slots, instructors, off_ids, counts)
        return sessions

    raise RuntimeError("Could not build a feasible weekly schedule.")

def validate_week(sessions: list[WeeklyClassSession], class_slots: list[BaselineClassSlot], instructors: dict[str, Instructor], off_ids: set[str], counts: dict[str, int]) -> None:
    if len(sessions) != len(class_slots):
        raise RuntimeError("Weekly session count does not match baseline slot count.")

    sub_count = 0
    seen: set[str] = set()

    for session in sessions:
        if session.slot_id in seen:
            raise ValueError("Duplicate weekly session for slot " + session.slot_id + ".")
        seen.add(session.slot_id)
        if session.is_substitution:
            sub_count += 1

    for iid, instructor in instructors.items():
        n = counts.get(iid, 0)

        if iid in off_ids:
            if n != 0:
                raise ValueError("Off instructor " + iid + " must teach 0 classes this week.")
            continue

        if n < instructor.baseline_class_count:
            raise ValueError("Active instructor " + iid + " must teach all baseline slots when not off.")
        if n > instructor.baseline_class_count + MAX_WEEKLY_DEVIATION:
            raise ValueError("Instructor " + iid + " exceeded maximum weekly teaching load.")

    expected_subs = sum(1 for slot in class_slots if slot.usual_instructor in off_ids)
    if sub_count != expected_subs:
        raise RuntimeError("Substitution count does not match number of off-instructor slots.")

def summarize_week(sessions: list[WeeklyClassSession]) -> dict[str, int]:
    if not isinstance(sessions, list):
        raise TypeError("sessions must be a list.")
    return {
        "total_sessions": len(sessions),
        "substitutions": sum(1 for s in sessions if s.is_substitution),
        "unique_assigned_instructors": len({s.assigned_instructor_id for s in sessions}),
    }
