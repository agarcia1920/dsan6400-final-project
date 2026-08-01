# Turn baseline slots into one realized week of classes.
# Off instructors skip their classes; substitutes in the same market cover those slots.
# Uses the same (day_of_week, daily_slot_index) time proxy as baseline assignment.

from dataclasses import dataclass, field

import numpy as np
from soulcycle_network.baseline_class_slot import BaselineClassSlot
from soulcycle_network.config import MAX_WEEKLY_DEVIATION, PROB_OFF_WEEK
from soulcycle_network.instructor import Instructor
from soulcycle_network.weekly_class_session import WeeklyClassSession

MAX_OFF_TRIES = 100

@dataclass
class WeeklyScheduleResult: #class to represent a weekly schedule result
    week_number: int
    sessions: list[WeeklyClassSession]
    off_instructor_ids: set[str] = field(default_factory=set)
    substitution_count: int = 0
    uncovered_session_ids: list[str] = field(default_factory=list)

def draw_off_instructors(instructors: dict[str, Instructor], rng: np.random.Generator, prob_off: float = PROB_OFF_WEEK) -> set[str]: #function to draw the off instructors
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

def pick_sub(studio_id: str, market: str, day: str, slot_index: int, instructors: dict[str, Instructor], off_ids: set[str], counts: dict[str, int], busy_times: dict[str, set[tuple[str, int]]], rng: np.random.Generator) -> str: #function to pick the substitute
    if not isinstance(studio_id, str):
        raise TypeError("studio_id must be a string.")
    if not isinstance(market, str):
        raise TypeError("market must be a string.")
    if not isinstance(day, str):
        raise TypeError("day must be a string.")
    if isinstance(slot_index, bool) or not isinstance(slot_index, int):
        raise TypeError("slot_index must be an integer.")
    if not isinstance(instructors, dict):
        raise TypeError("instructors must be a dictionary.")
    if not isinstance(off_ids, set):
        raise TypeError("off_ids must be a set.")
    if not isinstance(counts, dict):
        raise TypeError("counts must be a dictionary.")
    if not isinstance(busy_times, dict):
        raise TypeError("busy_times must be a dictionary.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")

    studio_id = studio_id.strip()
    market = market.strip()
    day = day.strip()
    time_key = (day, slot_index)

    preferred: list[Instructor] = []
    pool: list[Instructor] = []

    for iid, instructor in instructors.items():
        if iid in off_ids:
            continue
        if instructor.network_market != market:
            continue
        if time_key in busy_times.get(iid, set()):
            continue
        if sub_cap_left(instructor, counts) <= 0:
            continue
        pool.append(instructor)
        if studio_id in instructor.regular_studio_assignments:
            preferred.append(instructor)

    candidates = preferred if preferred else pool
    if not candidates:
        raise RuntimeError("No substitute available for " + studio_id + " in " + market + " at " + str(time_key) + ".")

    weights = np.array([sub_cap_left(i, counts) for i in candidates], dtype=float)
    pick = int(rng.choice(len(candidates), p=weights / weights.sum()))
    return candidates[pick].instructor_id

def make_session(week: int, slot: BaselineClassSlot, assigned_id: str, is_sub: bool) -> WeeklyClassSession: #function to make the session
    return WeeklyClassSession(
        week_number=week,
        slot_id=slot.slot_id,
        studio_id=slot.studio_id,
        day_of_week=slot.day_of_week,
        daily_slot_index=slot.daily_slot_index,
        room=slot.room,
        capacity=slot.capacity,
        usual_instructor_id=slot.usual_instructor,
        assigned_instructor_id=assigned_id,
        is_substitution=is_sub,
    )

def snapshot_baseline(class_slots: list[BaselineClassSlot]) -> dict[str, str | None]: #function to snapshot the baseline
    if not isinstance(class_slots, list):
        raise TypeError("class_slots must be a list.")
    return {slot.slot_id: slot.usual_instructor for slot in class_slots}

def create_weekly_schedule(week: int, instructors: dict[str, Instructor], baseline_slots: list[BaselineClassSlot], rng: np.random.Generator, prob_off: float = PROB_OFF_WEEK) -> WeeklyScheduleResult: #function to create the weekly schedule  
    if isinstance(week, bool) or not isinstance(week, int):
        raise TypeError("week must be an integer.")
    if week <= 0:
        raise ValueError("week must be positive.")
    if not isinstance(instructors, dict):
        raise TypeError("instructors must be a dictionary.")
    if not isinstance(baseline_slots, list):
        raise TypeError("baseline_slots must be a list.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")

    baseline_snapshot = snapshot_baseline(baseline_slots)

    for attempt in range(MAX_OFF_TRIES): #attempt to create the weekly schedule
        off_ids = draw_off_instructors(instructors, rng, prob_off)
        counts = {iid: 0 for iid in instructors}
        busy_times: dict[str, set[tuple[str, int]]] = {iid: set() for iid in instructors}
        by_slot: dict[str, WeeklyClassSession] = {}
        uncovered: list[str] = []

        for slot in baseline_slots: #assign the usual instructors to the baseline slots
            if not isinstance(slot, BaselineClassSlot):
                raise TypeError("baseline_slots must contain BaselineClassSlot objects.")
            if slot.usual_instructor in off_ids:
                continue
            counts[slot.usual_instructor] += 1
            busy_times[slot.usual_instructor].add((slot.day_of_week, slot.daily_slot_index))
            by_slot[slot.slot_id] = make_session(week, slot, slot.usual_instructor, False)

        sub_slots = [slot for slot in baseline_slots if slot.usual_instructor in off_ids]
        rng.shuffle(sub_slots)

        try: #try to assign the substitutes to the baseline slots
            for slot in sub_slots:
                market = instructors[slot.usual_instructor].network_market
                sub_id = pick_sub(slot.studio_id, market, slot.day_of_week, slot.daily_slot_index, instructors, off_ids, counts, busy_times, rng)
                counts[sub_id] += 1
                busy_times[sub_id].add((slot.day_of_week, slot.daily_slot_index))
                by_slot[slot.slot_id] = make_session(week, slot, sub_id, True)
        except RuntimeError:
            if attempt == MAX_OFF_TRIES - 1: #raise an error if the weekly schedule cannot be built
                raise RuntimeError("Could not build a feasible weekly schedule after " + str(MAX_OFF_TRIES) + " tries.")
            continue

        sessions = [by_slot[slot.slot_id] for slot in baseline_slots] #get the sessions
        sub_count = sum(1 for session in sessions if session.is_substitution)

        result = WeeklyScheduleResult( #create the weekly schedule result
            week_number=week,
            sessions=sessions,
            off_instructor_ids=off_ids,
            substitution_count=sub_count,
            uncovered_session_ids=uncovered,
        )

        validate_weekly_schedule(result.sessions, instructors, baseline_slots, off_ids, week_number=week, baseline_snapshot=baseline_snapshot)
        return result

    raise RuntimeError("Could not build a feasible weekly schedule.")

def create_weekly_class_sessions(week: int, instructors: dict[str, Instructor], class_slots: list[BaselineClassSlot], rng: np.random.Generator, prob_off: float = PROB_OFF_WEEK) -> list[WeeklyClassSession]:
    return create_weekly_schedule(week, instructors, class_slots, rng, prob_off).sessions #return the weekly class sessions

def validate_weekly_schedule(weekly_sessions: list[WeeklyClassSession], instructors: dict[str, Instructor], baseline_slots: list[BaselineClassSlot], off_instructor_ids: set[str], week_number: int | None = None, baseline_snapshot: dict[str, str | None] | None = None) -> None: #function to validate the weekly schedule
    if not isinstance(weekly_sessions, list):
        raise TypeError("weekly_sessions must be a list.")
    if not isinstance(instructors, dict):
        raise TypeError("instructors must be a dictionary.")
    if not isinstance(baseline_slots, list):
        raise TypeError("baseline_slots must be a list.")
    if not isinstance(off_instructor_ids, set):
        raise TypeError("off_instructor_ids must be a set.")

    if len(weekly_sessions) != len(baseline_slots): #raise an error if the weekly session count does not match the baseline slot count
        raise RuntimeError("Weekly session count does not match baseline slot count.")

    slot_ids = [session.slot_id for session in weekly_sessions] 
    if len(slot_ids) != len(set(slot_ids)):
        raise ValueError("Weekly schedule contains duplicate session IDs.") #raise an error if the weekly schedule contains duplicate session IDs

    baseline_lookup = {slot.slot_id: slot for slot in baseline_slots}
    expected_slot_ids = {slot.slot_id for slot in baseline_slots}
    if set(slot_ids) != expected_slot_ids:
        raise ValueError("Weekly sessions do not cover the same baseline slot IDs.")

    if baseline_snapshot is None: #if the baseline snapshot is not provided, create it
        baseline_snapshot = snapshot_baseline(baseline_slots)
    for slot in baseline_slots: #validate the baseline slots
        if slot.usual_instructor != baseline_snapshot.get(slot.slot_id):
            raise ValueError("Baseline slot " + slot.slot_id + " was mutated during weekly scheduling.")

    instructor_times: dict[str, set[tuple[str, int]]] = {}
    counts: dict[str, int] = {iid: 0 for iid in instructors}

    for session in weekly_sessions: #validate the weekly sessions
        if not isinstance(session, WeeklyClassSession):
            raise TypeError("weekly_sessions must contain WeeklyClassSession objects.")
        if week_number is not None and session.week_number != week_number:
            raise ValueError("Session " + session.slot_id + " has week_number " + str(session.week_number) + ", expected " + str(week_number) + ".")
        if not session.assigned_instructor_id:
            raise ValueError("Session " + session.slot_id + " has no assigned instructor.")

        baseline_slot = baseline_lookup[session.slot_id]
        if session.usual_instructor_id != baseline_slot.usual_instructor:
            raise ValueError("Session " + session.slot_id + " usual instructor does not match baseline.")
        if session.day_of_week != baseline_slot.day_of_week or session.daily_slot_index != baseline_slot.daily_slot_index:
            raise ValueError("Session " + session.slot_id + " time fields do not match baseline slot.")

        iid = session.assigned_instructor_id
        if iid in off_instructor_ids:
            raise ValueError("Off instructor " + iid + " was assigned a weekly session.")

        time_key = (session.day_of_week, session.daily_slot_index)
        busy = instructor_times.setdefault(iid, set())
        if time_key in busy:
            raise ValueError("Instructor " + iid + " has overlapping weekly sessions at " + str(time_key) + ".")
        busy.add(time_key)
        counts[iid] += 1

        if session.is_substitution:
            usual_market = instructors[session.usual_instructor_id].network_market
            sub_market = instructors[iid].network_market
            if sub_market != usual_market:
                raise ValueError("Substitute " + iid + " is not in the same market as usual instructor " + session.usual_instructor_id + ".")
        elif session.assigned_instructor_id != session.usual_instructor_id:
            raise ValueError("Non-substitution session " + session.slot_id + " must assign the usual instructor.")

    for iid, instructor in instructors.items():
        n = counts.get(iid, 0) #get the number of classes taught by the instructor
        if iid in off_instructor_ids:
            if n != 0:
                raise ValueError("Off instructor " + iid + " must teach 0 classes this week.")
            continue
        if n < instructor.baseline_class_count:
            raise ValueError("Active instructor " + iid + " must teach all baseline slots when not off.") #raise an error if the instructor has not taught all baseline slots
        if n > instructor.baseline_class_count + MAX_WEEKLY_DEVIATION:
            raise ValueError("Instructor " + iid + " exceeded maximum weekly teaching load.") #raise an error if the instructor has exceeded the maximum weekly teaching load

def summarize_week(sessions: list[WeeklyClassSession]) -> dict[str, int]: #function to summarize the week
    if not isinstance(sessions, list):
        raise TypeError("sessions must be a list.")
    return {
        "total_sessions": len(sessions),
        "substitutions": sum(1 for s in sessions if s.is_substitution),
        "unique_assigned_instructors": len({s.assigned_instructor_id for s in sessions}),
    }

def summarize_weekly_simulation(results: list[WeeklyScheduleResult], n_instructors: int) -> dict[str, float]: #function to summarize the weekly simulation
    if not isinstance(results, list):
        raise TypeError("results must be a list.")
    if isinstance(n_instructors, bool) or not isinstance(n_instructors, int):
        raise TypeError("n_instructors must be an integer.")

    n_weeks = len(results)
    total_off = sum(len(r.off_instructor_ids) for r in results)
    subs_per_week = [r.substitution_count for r in results]
    uncovered_per_week = [len(r.uncovered_session_ids) for r in results]

    return {
        "weeks": float(n_weeks),
        "observed_off_rate": total_off / (n_weeks * n_instructors) if n_weeks > 0 and n_instructors > 0 else 0.0,
        "avg_substitutions_per_week": float(np.mean(subs_per_week)) if subs_per_week else 0.0,
        "max_substitutions_in_week": float(max(subs_per_week)) if subs_per_week else 0.0,
        "total_uncovered_sessions": float(sum(uncovered_per_week)),
        "max_uncovered_in_week": float(max(uncovered_per_week)) if uncovered_per_week else 0.0,
    }
