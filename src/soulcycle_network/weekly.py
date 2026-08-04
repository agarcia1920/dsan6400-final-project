# Weekly schedules, rider booking into sessions, and enrollments passed to tie updates.

from __future__ import annotations

from soulcycle_network.coordination import CoordinationPair
from soulcycle_network.instructors import Instructor
from soulcycle_network.riders import Rider, coerce_float, draw_weekly_ride_count, simulated_session_capacity
from soulcycle_network.studios import BaselineClassSlot

# One rider attending one weekly class session.

from dataclasses import dataclass

@dataclass
class AttendanceRecord:
    week_number: int
    slot_id: str
    rider_id: str
    studio_id: str
    assigned_instructor_id: str
    day_of_week: str
    daily_slot_index: int
    is_coordinated: bool = False

    # validate input types and values
    def __post_init__(self) -> None:
        if isinstance(self.week_number, bool) or not isinstance(self.week_number, int):
            raise TypeError("week_number must be an integer.")
        if self.week_number <= 0:
            raise ValueError("week_number must be positive.")
        if not isinstance(self.slot_id, str):
            raise TypeError("slot_id must be a string.")
        if not isinstance(self.rider_id, str):
            raise TypeError("rider_id must be a string.")
        if not isinstance(self.studio_id, str):
            raise TypeError("studio_id must be a string.")
        if not isinstance(self.assigned_instructor_id, str):
            raise TypeError("assigned_instructor_id must be a string.")
        if not isinstance(self.day_of_week, str):
            raise TypeError("day_of_week must be a string.")
        if isinstance(self.daily_slot_index, bool) or not isinstance(self.daily_slot_index, int):
            raise TypeError("daily_slot_index must be an integer.")
        if not isinstance(self.is_coordinated, bool):
            raise TypeError("is_coordinated must be a boolean.")

        # strip whitespace
        self.slot_id = self.slot_id.strip()
        self.rider_id = self.rider_id.strip()
        self.studio_id = self.studio_id.strip()
        self.assigned_instructor_id = self.assigned_instructor_id.strip()
        self.day_of_week = self.day_of_week.strip()

        # validate input values
        if not self.slot_id:
            raise ValueError("slot_id cannot be empty.")
        if not self.rider_id:
            raise ValueError("rider_id cannot be empty.")
        if not self.studio_id:
            raise ValueError("studio_id cannot be empty.")
        if not self.assigned_instructor_id:
            raise ValueError("assigned_instructor_id cannot be empty.")
        if not self.day_of_week:
            raise ValueError("day_of_week cannot be empty.")
        if self.daily_slot_index <= 0:
            raise ValueError("daily_slot_index must be positive.")

# Generate a unique key for a session.
def session_key(week_number: int, slot_id: str) -> str:
    # validate input types
    if isinstance(week_number, bool) or not isinstance(week_number, int):
        raise TypeError("week_number must be an integer.")
    if not isinstance(slot_id, str):
        raise TypeError("slot_id must be a string.")
    if week_number <= 0:
        raise ValueError("week_number must be positive.")

    # strip whitespace
    slot_id = slot_id.strip()
    if not slot_id:
        raise ValueError("slot_id cannot be empty.")
    return "W" + str(week_number).zfill(2) + "_" + slot_id

# One realized class in a specific week.

from dataclasses import dataclass
from soulcycle_network.config import DAYS_OF_WEEK

@dataclass
class WeeklyClassSession: #class to represent a weekly class session
    week_number: int
    slot_id: str
    studio_id: str
    day_of_week: str
    daily_slot_index: int
    room: str
    capacity: int
    usual_instructor_id: str
    assigned_instructor_id: str
    is_substitution: bool

    def __post_init__(self):
        if isinstance(self.week_number, bool) or not isinstance(self.week_number, int):
            raise TypeError("week_number must be an integer.")
        if self.week_number <= 0:
            raise ValueError("week_number must be positive.")
        if not isinstance(self.slot_id, str):
            raise TypeError("slot_id must be a string.")
        if not isinstance(self.studio_id, str):
            raise TypeError("studio_id must be a string.")
        if not isinstance(self.day_of_week, str):
            raise TypeError("day_of_week must be a string.")
        if isinstance(self.daily_slot_index, bool) or not isinstance(self.daily_slot_index, int):
            raise TypeError("daily_slot_index for " + self.slot_id + " must be an integer.")
        if self.daily_slot_index <= 0:
            raise ValueError("daily_slot_index for " + self.slot_id + " must be positive.")
        if not isinstance(self.room, str):
            raise TypeError("room must be a string.")
        if not isinstance(self.usual_instructor_id, str):
            raise TypeError("usual_instructor_id must be a string.")
        if not isinstance(self.assigned_instructor_id, str):
            raise TypeError("assigned_instructor_id must be a string.")
        if not isinstance(self.is_substitution, bool):
            raise TypeError("is_substitution must be a boolean.")

        self.slot_id = self.slot_id.strip()
        self.studio_id = self.studio_id.strip()
        self.day_of_week = self.day_of_week.strip()
        self.room = self.room.strip()
        self.usual_instructor_id = self.usual_instructor_id.strip()
        self.assigned_instructor_id = self.assigned_instructor_id.strip()

        if not self.slot_id:
            raise ValueError("slot_id cannot be empty.")
        if not self.studio_id:
            raise ValueError("studio_id cannot be empty.")
        if not self.room:
            raise ValueError("room cannot be empty.")
        if self.day_of_week not in DAYS_OF_WEEK:
            raise ValueError("Invalid day of week '" + self.day_of_week + "' for slot " + self.slot_id + ".")
        if not self.usual_instructor_id:
            raise ValueError("usual_instructor_id cannot be empty.")
        if not self.assigned_instructor_id:
            raise ValueError("assigned_instructor_id cannot be empty.")

        if isinstance(self.capacity, bool) or not isinstance(self.capacity, int):
            raise TypeError("capacity for " + self.slot_id + " must be an integer.")
        if self.capacity <= 0:
            raise ValueError("capacity for " + self.slot_id + " must be positive.")

        if self.is_substitution and self.assigned_instructor_id == self.usual_instructor_id:
            raise ValueError("Substitution session " + self.slot_id + " cannot assign the usual instructor.")
        if not self.is_substitution and self.assigned_instructor_id != self.usual_instructor_id:
            raise ValueError("Non-substitution session " + self.slot_id + " must assign the usual instructor.")

# Turn baseline slots into one realized week of classes.
# Off instructors skip their classes; substitutes in the same market cover those slots.
# Uses the same (day_of_week, daily_slot_index) time proxy as baseline assignment.

from dataclasses import dataclass, field

import numpy as np
from soulcycle_network.studios import BaselineClassSlot
from soulcycle_network.config import MAX_WEEKLY_DEVIATION, PROB_OFF_WEEK
from soulcycle_network.instructors import Instructor

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
    prob_off = coerce_float(prob_off, "prob_off")
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

# Book riders into weekly class sessions with scaled capacity.

from dataclasses import dataclass, field
import numpy as np
from soulcycle_network.config import MARKET_WIDE_EXPLORATION_PROB, MAX_CLASSES_PER_DAY

@dataclass
class BookableSession: #class to store the bookable session
    week_number: int
    slot_id: str
    studio_id: str
    market: str
    day_of_week: str
    daily_slot_index: int
    assigned_instructor_id: str
    real_capacity: int
    sim_capacity: int
    enrolled: int = 0

    @property
    def seats_left(self) -> int:
        return self.sim_capacity - self.enrolled

@dataclass
class WeeklyBookingResult: #class to store the weekly booking result
    week_number: int
    records: list[AttendanceRecord] = field(default_factory=list)
    enrollments: dict[str, list[str]] = field(default_factory=dict)
    unmet_demand: int = 0
    coordinated_bookings: int = 0
    total_sim_seats: int = 0
    seats_filled: int = 0

def build_market_studios(studio_markets: dict[str, str]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for studio_id, market in studio_markets.items():
        out.setdefault(market, set()).add(studio_id)
    return out

def default_eligible_studio_ids(rider: Rider, cluster_studios: dict[str, set[str]]) -> set[str]:
    home = cluster_studios.get(rider.home_cluster, set())
    return home.union(rider.preferred_studio_ids)

def resolve_eligible_studio_ids(
    rider: Rider,
    cluster_studios: dict[str, set[str]],
    market_studios: dict[str, set[str]],
    explore_market: bool,
) -> set[str]:
    if explore_market:
        return set(market_studios.get(rider.home_market, set()))
    return default_eligible_studio_ids(rider, cluster_studios)

def build_bookable_sessions(schedule: WeeklyScheduleResult, scale: float, studio_markets: dict[str, str]) -> dict[str, BookableSession]:
    if not isinstance(schedule, WeeklyScheduleResult):
        raise TypeError("schedule must be a WeeklyScheduleResult.")
    scale = coerce_float(scale, "scale")
    if not isinstance(studio_markets, dict):
        raise TypeError("studio_markets must be a dictionary.")
    if scale <= 0:
        raise ValueError("scale must be positive.")

    out: dict[str, BookableSession] = {} #dictionary to store the bookable sessions
    for session in schedule.sessions:
        if not isinstance(session, WeeklyClassSession): 
            raise TypeError("schedule.sessions must contain WeeklyClassSession objects.")
        #get the market for the session
        market = studio_markets.get(session.studio_id)
        if market is None:
            raise ValueError("No market found for studio " + session.studio_id + ".")

        out[session.slot_id] = BookableSession( #add the bookable session to the dictionary
            week_number=session.week_number,
            slot_id=session.slot_id,
            studio_id=session.studio_id,
            market=market,
            day_of_week=session.day_of_week,
            daily_slot_index=session.daily_slot_index,
            assigned_instructor_id=session.assigned_instructor_id,
            real_capacity=session.capacity,
            sim_capacity=simulated_session_capacity(session.capacity, scale),
        )
    return out

def score_session(rider: Rider, session: BookableSession, home_cluster_studios: set[str]) -> float: #function to score the session
    if not isinstance(rider, Rider):
        raise TypeError("rider must be a Rider.")
    if not isinstance(session, BookableSession):
        raise TypeError("session must be a BookableSession.")
    if not isinstance(home_cluster_studios, set):
        raise TypeError("home_cluster_studios must be a set.")

    score = 1.0
    if session.studio_id in rider.preferred_studio_ids: #add 3 points if the session is in the rider's preferred studio
        score += 3.0
    if session.assigned_instructor_id in rider.preferred_instructor_ids: #add 2 points if the session is in the rider's preferred instructor
        score += 2.0
    if session.studio_id in home_cluster_studios: #add 1 point if the session is in the rider's home cluster
        score += 1.0
    return score

def pick_session(rider: Rider, candidates: list[BookableSession], home_cluster_studios: set[str], rng: np.random.Generator) -> BookableSession | None:
    if not isinstance(rider, Rider):
        raise TypeError("rider must be a Rider.")
    if not isinstance(candidates, list):
        raise TypeError("candidates must be a list.")
    if not isinstance(home_cluster_studios, set):
        raise TypeError("home_cluster_studios must be a set.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")

    open_sessions = [s for s in candidates if s.seats_left > 0] #get the open sessions
    if not open_sessions:
        return None #return None if there are no open sessions

    weights = np.array([score_session(rider, s, home_cluster_studios) for s in open_sessions], dtype=float) #calculate the weights for the sessions
    pick = int(rng.choice(len(open_sessions), p=weights / weights.sum())) #pick the session with the highest weight
    return open_sessions[pick] #return the session

def rider_candidates(
    rider: Rider,
    sessions: dict[str, BookableSession],
    booked_days: set[str],
    eligible_studio_ids: set[str],
) -> list[BookableSession]: #function to get the rider candidates
    if not isinstance(rider, Rider):
        raise TypeError("rider must be a Rider.")
    if not isinstance(sessions, dict):
        raise TypeError("sessions must be a dictionary.")
    if not isinstance(booked_days, set):
        raise TypeError("booked_days must be a set.")
    if not isinstance(eligible_studio_ids, set):
        raise TypeError("eligible_studio_ids must be a set.")

    out: list[BookableSession] = [] #list to store the rider candidates
    for session in sessions.values():
        if session.market != rider.home_market:
            continue #skip the session if it is not in the rider's home market
        if session.studio_id not in eligible_studio_ids:
            continue
        if session.day_of_week in booked_days:
            continue #skip the session if it is already booked
        if session.seats_left <= 0:
            continue #skip the session if it is full
        out.append(session)
    return out #return the rider candidates

def enroll_rider(rider: Rider, session: BookableSession, result: WeeklyBookingResult, is_coordinated: bool) -> AttendanceRecord:
    if not isinstance(rider, Rider):
        raise TypeError("rider must be a Rider.")
    if not isinstance(session, BookableSession):
        raise TypeError("session must be a BookableSession.")
    if not isinstance(result, WeeklyBookingResult):
        raise TypeError("result must be a WeeklyBookingResult.")
    if not isinstance(is_coordinated, bool):
        raise TypeError("is_coordinated must be a boolean.")
    if session.seats_left <= 0:
        raise ValueError("Session " + session.slot_id + " is full.")

    session.enrolled += 1 #increment the enrolled count
    record = AttendanceRecord( #create the attendance record
        week_number=session.week_number,
        slot_id=session.slot_id,
        rider_id=rider.rider_id,
        studio_id=session.studio_id,
        assigned_instructor_id=session.assigned_instructor_id,
        day_of_week=session.day_of_week,
        daily_slot_index=session.daily_slot_index,
        is_coordinated=is_coordinated,
    )
    result.records.append(record) #add the attendance record to the result
    result.enrollments.setdefault(session.slot_id, []).append(rider.rider_id)
    return record #return the attendance record

def apply_attendance(rider: Rider, record: AttendanceRecord) -> None: #function to apply the attendance
    if not isinstance(rider, Rider):
        raise TypeError("rider must be a Rider.")
    if not isinstance(record, AttendanceRecord):
        raise TypeError("record must be an AttendanceRecord.")

    key = "W" + str(record.week_number).zfill(2) + "_" + record.slot_id #create the key for the attendance record
    rider.attended_session_ids.append(key) #add the key to the attended session ids
    rider.attended_instructor_counts[record.assigned_instructor_id] = rider.attended_instructor_counts.get(record.assigned_instructor_id, 0) + 1 #increment the instructor count
    rider.attended_studio_counts[record.studio_id] = rider.attended_studio_counts.get(record.studio_id, 0) + 1 #increment the studio count

def book_coordination_pair(
    pair: CoordinationPair,
    riders: dict[str, Rider],
    sessions: dict[str, BookableSession],
    cluster_studios: dict[str, set[str]],
    rider_days: dict[str, set[str]],
    result: WeeklyBookingResult,
    rng: np.random.Generator,
) -> bool: #function to book the coordination pair
    if not isinstance(pair, CoordinationPair):
        raise TypeError("pair must be a CoordinationPair.")
    if not isinstance(riders, dict):
        raise TypeError("riders must be a dictionary.")
    if not isinstance(sessions, dict):
        raise TypeError("sessions must be a dictionary.")
    if not isinstance(cluster_studios, dict):
        raise TypeError("cluster_studios must be a dictionary.")
    if not isinstance(rider_days, dict):
        raise TypeError("rider_days must be a dictionary.")
    if not isinstance(result, WeeklyBookingResult):
        raise TypeError("result must be a WeeklyBookingResult.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")

    rider_a = riders[pair.rider_a]
    rider_b = riders[pair.rider_b]
    days_a = rider_days.get(rider_a.rider_id, set())
    days_b = rider_days.get(rider_b.rider_id, set())
    eligible_studios = default_eligible_studio_ids(rider_a, cluster_studios).union(
        default_eligible_studio_ids(rider_b, cluster_studios)
    )

    shared: list[BookableSession] = [] #list to store the shared sessions
    for session in sessions.values():
        if session.market != rider_a.home_market or session.market != rider_b.home_market:
            continue #skip the session if it is not in the rider's home market
        if session.studio_id not in eligible_studios:
            continue
        if session.day_of_week in days_a or session.day_of_week in days_b:
            continue #skip the session if it is already booked
        if session.seats_left < 2:
            continue #skip the session if it is not enough seats left
        shared.append(session)

    if not shared:
        return False #return False if there are no shared sessions

    home_a = cluster_studios.get(rider_a.home_cluster, set()) #get the home studios for rider a
    home_b = cluster_studios.get(rider_b.home_cluster, set()) #get the home studios for rider b
    home_studios = home_a.union(home_b) #get the home studios for both riders

    weights = np.array([ #calculate the weights for the shared sessions
        score_session(rider_a, s, home_studios) + score_session(rider_b, s, home_studios)
        for s in shared
    ], dtype=float) #calculate the weights for the shared sessions
    pick = int(rng.choice(len(shared), p=weights / weights.sum())) #pick the session with the highest weight
    session = shared[pick] #get the session with the highest weight

    enroll_rider(rider_a, session, result, True) #enroll the rider a
    enroll_rider(rider_b, session, result, True) #enroll the rider b
    rider_days.setdefault(rider_a.rider_id, set()).add(session.day_of_week)
    rider_days.setdefault(rider_b.rider_id, set()).add(session.day_of_week)
    result.coordinated_bookings += 1 #increment the coordinated bookings count
    return True

def draw_weekly_counts(riders: dict[str, Rider], rng: np.random.Generator) -> dict[str, int]: #function to draw the weekly counts
    if not isinstance(riders, dict):
        raise TypeError("riders must be a dictionary.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")

    weekly_counts: dict[str, int] = {} #dictionary to store the weekly counts
    for rider in riders.values():
        if not isinstance(rider, Rider):
            raise TypeError("riders must contain Rider objects.")
        weekly_counts[rider.rider_id] = draw_weekly_ride_count(rider, rng)
    return weekly_counts #return the weekly counts

def book_week(schedule: WeeklyScheduleResult, riders: dict[str, Rider], scale: float, studio_markets: dict[str, str], cluster_studios: dict[str, set[str]], weekly_counts: dict[str, int], coordination_pairs: list[CoordinationPair], rng: np.random.Generator) -> WeeklyBookingResult: #function to book the week
    if not isinstance(schedule, WeeklyScheduleResult):
        raise TypeError("schedule must be a WeeklyScheduleResult.")
    if not isinstance(riders, dict):
        raise TypeError("riders must be a dictionary.")
    scale = coerce_float(scale, "scale")
    if not isinstance(studio_markets, dict):
        raise TypeError("studio_markets must be a dictionary.")
    if not isinstance(cluster_studios, dict):
        raise TypeError("cluster_studios must be a dictionary.")
    if not isinstance(weekly_counts, dict):
        raise TypeError("weekly_counts must be a dictionary.")
    if not isinstance(coordination_pairs, list):
        raise TypeError("coordination_pairs must be a list.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")

    sessions = build_bookable_sessions(schedule, scale, studio_markets)
    market_studios = build_market_studios(studio_markets)
    result = WeeklyBookingResult(week_number=schedule.week_number) #create the booking result

    rider_days: dict[str, set[str]] = {rid: set() for rid in riders} #dictionary to store the rider days
    remaining = dict(weekly_counts) #dictionary to store the remaining counts

    for pair in coordination_pairs:
        if remaining.get(pair.rider_a, 0) <= 0 or remaining.get(pair.rider_b, 0) <= 0:
            continue #skip the pair if it is not enough riders
        if book_coordination_pair(pair, riders, sessions, cluster_studios, rider_days, result, rng):
            remaining[pair.rider_a] -= 1 #decrement the remaining count for riders
            remaining[pair.rider_b] -= 1 

    rider_ids = list(riders.keys())
    rng.shuffle(rider_ids)

    for rider_id in rider_ids:
        rider = riders[rider_id]
        target = remaining.get(rider_id, 0)
        if target <= 0:
            continue #skip the rider if it is not enough riders

        home_studios = cluster_studios.get(rider.home_cluster, set()) #get the home studios for the rider
        booked = rider_days.setdefault(rider_id, set()) #get the booked days for the rider
        explore_market = rng.random() < MARKET_WIDE_EXPLORATION_PROB
        eligible_studios = resolve_eligible_studio_ids(rider, cluster_studios, market_studios, explore_market)

        for _ in range(target):
            candidates = rider_candidates(rider, sessions, booked, eligible_studios) #get the candidates for the rider
            session = pick_session(rider, candidates, home_studios, rng)
            if session is None:
                result.unmet_demand += 1 #increment the unmet demand count
                continue

            enroll_rider(rider, session, result, False)
            booked.add(session.day_of_week)

    for record in result.records:
        apply_attendance(riders[record.rider_id], record)

    result.total_sim_seats = sum(session.sim_capacity for session in sessions.values())
    result.seats_filled = sum(session.enrolled for session in sessions.values())

    return result

def summarize_booking(result: WeeklyBookingResult) -> dict[str, float]: #function to summarize the booking
    if not isinstance(result, WeeklyBookingResult):
        raise TypeError("result must be a WeeklyBookingResult.")

    return {
        "attendance_count": float(len(result.records)),
        "unique_riders": float(len({r.rider_id for r in result.records})),
        "sessions_with_riders": float(len(result.enrollments)),
        "unmet_demand": float(result.unmet_demand),
        "coordinated_bookings": float(result.coordinated_bookings),
        "coordinated_records": float(sum(1 for r in result.records if r.is_coordinated)),
    }

