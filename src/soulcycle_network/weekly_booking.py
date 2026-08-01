# Book riders into weekly class sessions with scaled capacity.

from dataclasses import dataclass, field
import numpy as np
from soulcycle_network.attendance_record import AttendanceRecord
from soulcycle_network.config import MAX_CLASSES_PER_DAY
from soulcycle_network.rider import Rider
from soulcycle_network.rider_coordination import CoordinationPair
from soulcycle_network.rider_parameters import draw_weekly_ride_count, simulated_session_capacity
from soulcycle_network.weekly_class_session import WeeklyClassSession
from soulcycle_network.weekly_schedule import WeeklyScheduleResult

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

def build_bookable_sessions(schedule: WeeklyScheduleResult, scale: float, studio_markets: dict[str, str]) -> dict[str, BookableSession]:
    if not isinstance(schedule, WeeklyScheduleResult):
        raise TypeError("schedule must be a WeeklyScheduleResult.")
    if not isinstance(scale, float):
        raise TypeError("scale must be a float.")
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

def rider_candidates(rider: Rider, sessions: dict[str, BookableSession], booked_days: set[str]) -> list[BookableSession]: #function to get the rider candidates
    if not isinstance(rider, Rider):
        raise TypeError("rider must be a Rider.")
    if not isinstance(sessions, dict):
        raise TypeError("sessions must be a dictionary.")
    if not isinstance(booked_days, set):
        raise TypeError("booked_days must be a set.")

    out: list[BookableSession] = [] #list to store the rider candidates
    for session in sessions.values():
        if session.market != rider.home_market:
            continue #skip the session if it is not in the rider's home market
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

def book_coordination_pair(pair: CoordinationPair, riders: dict[str, Rider], sessions: dict[str, BookableSession], cluster_studios: dict[str, set[str]], rider_days: dict[str, set[str]], result: WeeklyBookingResult, rng: np.random.Generator) -> bool: #function to book the coordination pair
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

    shared: list[BookableSession] = [] #list to store the shared sessions
    for session in sessions.values():
        if session.market != rider_a.home_market or session.market != rider_b.home_market:
            continue #skip the session if it is not in the rider's home market  
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
    if not isinstance(scale, float):
        raise TypeError("scale must be a float.")
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

        for _ in range(target):
            candidates = rider_candidates(rider, sessions, booked) #get the candidates for the rider
            session = pick_session(rider, candidates, home_studios, rng)
            if session is None:
                result.unmet_demand += 1 #increment the unmet demand count
                continue

            enroll_rider(rider, session, result, False)
            booked.add(session.day_of_week)

    for record in result.records:
        apply_attendance(riders[record.rider_id], record)

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
