# Studios, recurring class slots, and scaled seat capacity for each session.

from __future__ import annotations
# Studio objects for the SoulCycle network simulation.

from dataclasses import dataclass, field

#these are the market tiers we use across the project
MARKET_TIERS = {"Mega", "Large", "Medium", "Concentrated"}

@dataclass
class Studio:
    studio_id: str
    studio_name: str
    official_region: str #the region SoulCycle officially assigns the studio to
    network_market: str #the broader market the studio belongs to, like DMV or Greater NYC
    market_tier: str
    local_ridership_cluster: str #where riders around this studio mostly come from
    weekly_class_count: int #total classes offered at this studio in a normal week
    room_class_counts: dict[str, int] = field(default_factory=dict) #classes per room, like {"A": 36, "B": 4}
    room_capacities: dict[str, int] = field(default_factory=dict) #bikes per room, like {"A": 59, "B": 46}
    daily_class_counts: dict[str, int] = field(default_factory=dict) #total classes per day across all rooms
    room_daily_class_counts: dict[str, dict[str, int]] = field(default_factory=dict) #classes per day within each room

    def __post_init__(self) -> None:
        #check string fields before stripping
        if not isinstance(self.studio_id, str):
            raise TypeError("studio_id must be a string.")
        if not isinstance(self.studio_name, str):
            raise TypeError("studio_name must be a string.")
        if not isinstance(self.official_region, str):
            raise TypeError("official_region must be a string.")
        if not isinstance(self.network_market, str):
            raise TypeError("network_market must be a string.")
        if not isinstance(self.market_tier, str):
            raise TypeError("market_tier must be a string.")
        if not isinstance(self.local_ridership_cluster, str):
            raise TypeError("local_ridership_cluster must be a string.")

        #remove whitespace from the studio's attributes
        self.studio_id = self.studio_id.strip()
        self.studio_name = self.studio_name.strip()
        self.official_region = self.official_region.strip()
        self.network_market = self.network_market.strip()
        self.market_tier = self.market_tier.strip()
        self.local_ridership_cluster = self.local_ridership_cluster.strip()

        #check that the studio's attributes are not empty
        if not self.studio_id:
            raise ValueError("studio_id cannot be empty.")
        if not self.studio_name:
            raise ValueError("Studio " + self.studio_id + " must have a studio_name.")
        if not self.official_region:
            raise ValueError("Studio " + self.studio_id + " must have an official_region.")
        if not self.network_market:
            raise ValueError("Studio " + self.studio_id + " must have a network_market.")
        if not self.local_ridership_cluster:
            raise ValueError("Studio " + self.studio_id + " must have a local_ridership_cluster.")

        #check that the market_tier is valid
        if self.market_tier not in MARKET_TIERS:
            raise ValueError("Invalid market_tier '" + self.market_tier + "' for studio " + self.studio_id + ". Expected one of " + str(sorted(MARKET_TIERS)) + ".")

        #check that weekly_class_count is an integer
        if isinstance(self.weekly_class_count, bool) or not isinstance(self.weekly_class_count, int):
            raise TypeError("weekly_class_count for " + self.studio_id + " must be an integer.")
        if self.weekly_class_count <= 0:
            raise ValueError("weekly_class_count for " + self.studio_id + " must be positive.")

        #check room-level class counts and capacities
        if not isinstance(self.room_class_counts, dict):
            raise TypeError("room_class_counts must be a dictionary.")
        if not isinstance(self.room_capacities, dict):
            raise TypeError("room_capacities must be a dictionary.")
        if set(self.room_class_counts.keys()) != set(self.room_capacities.keys()):
            raise ValueError("Studio " + self.studio_id + " room_class_counts and room_capacities must have the same room keys.")

        total_room_classes = 0
        for room, class_count in self.room_class_counts.items():
            if not isinstance(room, str):
                raise TypeError("room keys in room_class_counts must be strings.")
            if isinstance(class_count, bool) or not isinstance(class_count, int):
                raise TypeError("room_class_counts for " + self.studio_id + " must contain integers.")
            if class_count < 0:
                raise ValueError("room_class_counts for " + self.studio_id + " cannot be negative.")
            total_room_classes += class_count

            capacity = self.room_capacities[room]
            if isinstance(capacity, bool) or not isinstance(capacity, int):
                raise TypeError("room_capacities for " + self.studio_id + " must contain integers.")
            if capacity < 0:
                raise ValueError("room_capacities for " + self.studio_id + " cannot be negative.")
            if class_count > 0 and capacity <= 0:
                raise ValueError("Active rooms for " + self.studio_id + " must have positive room_capacities.")

        if total_room_classes <= 0:
            raise ValueError("Studio " + self.studio_id + " must have at least one room with classes.")

        if total_room_classes != self.weekly_class_count:
            raise ValueError("Studio " + self.studio_id + " weekly_class_count must equal the sum of room_class_counts.")

        if not isinstance(self.daily_class_counts, dict):
            raise TypeError("daily_class_counts must be a dictionary.")
        if not isinstance(self.room_daily_class_counts, dict):
            raise TypeError("room_daily_class_counts must be a dictionary.")

    @property
    def weekly_bike_supply(self) -> int:
        #total weekly bike supply across all active rooms
        total = 0
        for room, class_count in self.room_class_counts.items():
            if class_count > 0:
                total += class_count * self.room_capacities[room]
        return total

    @property
    def active_rooms(self) -> list[str]:
        #rooms that actually offer classes in a normal week
        return sorted([room for room, class_count in self.room_class_counts.items() if class_count > 0])

# Load studios from CSV.

from pathlib import Path
import pandas as pd

#required columns for the studio data
REQUIRED_COLUMNS = {"studio_id", "studio_name", "official_region", "network_market", "market_tier", "local_ridership_cluster", "rides_per_wk_a", "bikes_per_ride_a", "rides_per_wk_b", "bikes_per_ride_b"}

def load_studios(file_path: str | Path) -> dict[str, Studio]:
    #load studios from a CSV file and return them as a dictionary keyed by studio_id
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError("File not found: " + str(file_path))
    if file_path.suffix != ".csv":
        raise ValueError("File " + str(file_path) + " is not a CSV file.")

    studio_df = pd.read_csv(file_path)
    missing_cols = REQUIRED_COLUMNS - set(studio_df.columns)
    if missing_cols:
        raise ValueError("File " + str(file_path) + " is missing required columns: " + str(sorted(missing_cols)))

    studios: dict[str, Studio] = {}

    for idx, row in studio_df.iterrows():
        try:
            rides_a = int(row["rides_per_wk_a"])
            rides_b = int(row["rides_per_wk_b"])
            bikes_a = int(row["bikes_per_ride_a"])
            bikes_b = int(row["bikes_per_ride_b"])

            room_class_counts = {"A": rides_a, "B": rides_b}
            room_capacities = {"A": bikes_a, "B": bikes_b}
            weekly_class_count = rides_a + rides_b

            studio = Studio(
                studio_id=row["studio_id"],
                studio_name=row["studio_name"],
                official_region=row["official_region"],
                network_market=row["network_market"],
                market_tier=row["market_tier"],
                local_ridership_cluster=row["local_ridership_cluster"],
                weekly_class_count=weekly_class_count,
                room_class_counts=room_class_counts,
                room_capacities=room_capacities,
            )
        except (ValueError, TypeError, KeyError) as e:
            line_num = idx + 2
            raise ValueError("Error parsing studio data for row " + str(line_num) + ": " + str(e)) from e

        studios[row["studio_id"]] = studio

    return studios

# Divide each room's weekly class count across the seven days.

import numpy as np
from soulcycle_network.config import DAYS_OF_WEEK

def divide_classes(weekly_class_count: int, rng: np.random.Generator) -> dict[str, int]:
    #divide a weekly class count across the seven days as evenly as possible
    if isinstance(weekly_class_count, bool) or not isinstance(weekly_class_count, int):
        raise TypeError("weekly_class_count must be an integer.")
    if weekly_class_count <= 0:
        raise ValueError("weekly_class_count must be positive.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy random Generator.")

    num_days = len(DAYS_OF_WEEK)
    min_classes_per_day = weekly_class_count // num_days
    remaining_classes = weekly_class_count % num_days
    daily_schedule = {day: min_classes_per_day for day in DAYS_OF_WEEK}

    if remaining_classes > 0:
        for day in rng.choice(DAYS_OF_WEEK, size=remaining_classes, replace=False):
            daily_schedule[day] += 1

    if sum(daily_schedule.values()) != weekly_class_count:
        raise RuntimeError("Daily class counts do not sum to weekly_class_count.")

    return daily_schedule

def create_weekly_schedule(studio: Studio, rng: np.random.Generator) -> dict[str, int]:
    #build the daily schedule for one studio and store it on the studio object
    if not isinstance(studio, Studio):
        raise TypeError("studio must be a Studio object.")

    studio.room_daily_class_counts = {}
    studio.daily_class_counts = {day: 0 for day in DAYS_OF_WEEK}

    for room in studio.active_rooms:
        room_weekly_count = studio.room_class_counts[room]
        room_daily_schedule = divide_classes(room_weekly_count, rng)
        studio.room_daily_class_counts[room] = room_daily_schedule
        for day, class_count in room_daily_schedule.items():
            studio.daily_class_counts[day] += class_count

    if sum(studio.daily_class_counts.values()) != studio.weekly_class_count:
        raise RuntimeError("Studio " + studio.studio_id + " daily class counts do not sum to weekly_class_count.")

    return studio.daily_class_counts

def create_all_weekly_schedules(studios: dict[str, Studio], rng: np.random.Generator) -> dict[str, Studio]:
    #build daily schedules for every studio in the network
    if not isinstance(studios, dict):
        raise TypeError("studios must be a dictionary of Studio objects.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy random Generator.")

    for studio in studios.values():
        if not isinstance(studio, Studio):
            raise TypeError("studio must be a Studio object.")
        create_weekly_schedule(studio, rng)

    return studios

# Recurring class slots for each studio.

from dataclasses import dataclass
from soulcycle_network.config import DAYS_OF_WEEK

@dataclass
class BaselineClassSlot:
    studio_id: str
    slot_id: str #unique id for this recurring slot
    day_of_week: str
    daily_slot_index: int #time-position proxy, no explicit class times are modeled yet
    room: str #room label from the studio data
    capacity: int #bikes available in this room's class
    usual_instructor: str | None = None #filled in later when instructors are assigned to slots

    def __post_init__(self):
        # validate input types
        if not isinstance(self.studio_id, str):
            raise TypeError("studio_id must be a string.")
        if not isinstance(self.slot_id, str):
            raise TypeError("slot_id must be a string.")
        if not isinstance(self.day_of_week, str):
            raise TypeError("day_of_week must be a string.")
        if not isinstance(self.room, str):
            raise TypeError("room must be a string.")

        # strip whitespace
        self.slot_id = self.slot_id.strip()
        self.studio_id = self.studio_id.strip()
        self.day_of_week = self.day_of_week.strip()
        self.room = self.room.strip()

        # validate input values
        if not self.studio_id:
            raise ValueError("studio_id cannot be empty.")
        if not self.slot_id:
            raise ValueError("slot_id cannot be empty.")
        if not self.room:
            raise ValueError("room cannot be empty.")
        if self.day_of_week not in DAYS_OF_WEEK:
            raise ValueError("Invalid day of week '" + self.day_of_week + "' for class slot " + self.slot_id + ".")

        # validate daily_slot_index
        if isinstance(self.daily_slot_index, bool) or not isinstance(self.daily_slot_index, int):
            raise TypeError("daily_slot_index for " + self.slot_id + " must be an integer.")
        if self.daily_slot_index <= 0:
            raise ValueError("daily_slot_index for " + self.slot_id + " must be positive.")

        # validate capacity
        if isinstance(self.capacity, bool) or not isinstance(self.capacity, int):
            raise TypeError("capacity for " + self.slot_id + " must be an integer.")
        if self.capacity <= 0:
            raise ValueError("capacity for " + self.slot_id + " must be positive.")

        # validate usual_instructor
        if self.usual_instructor is not None:
            if not isinstance(self.usual_instructor, str):
                raise TypeError("usual_instructor for " + self.slot_id + " must be a string or None.")
            self.usual_instructor = self.usual_instructor.strip()
            if not self.usual_instructor:
                raise ValueError("usual_instructor for " + self.slot_id + " cannot be an empty string.")

# Build recurring class slots from studio daily schedules.
from soulcycle_network.config import DAYS_OF_WEEK

def create_studio_class_slots(studio: Studio) -> list[BaselineClassSlot]:
    #create all persistent recurring class slots for one studio
    if not isinstance(studio, Studio):
        raise TypeError("studio must be a Studio object")
    if not studio.room_daily_class_counts:
        raise ValueError("Studio " + studio.studio_id + " does not have a daily schedule.")

    class_slots: list[BaselineClassSlot] = []
    # iterate over all active rooms in the studio
    for room in studio.active_rooms:
        # get the daily class counts for the room
        room_daily_counts = studio.room_daily_class_counts.get(room)
        if room_daily_counts is None:
            raise ValueError("Studio " + studio.studio_id + " is missing a daily schedule for room " + room + ".")

        missing_days = set(DAYS_OF_WEEK) - set(room_daily_counts)
        if missing_days:
            raise ValueError("Studio " + studio.studio_id + " room " + room + " is missing daily class counts for: " + str(sorted(missing_days)))

        room_capacity = studio.room_capacities[room]

        for day in DAYS_OF_WEEK:
            num_classes = room_daily_counts[day]
            if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 0:
                raise ValueError("Studio " + studio.studio_id + " room " + room + " has invalid class count for " + day + ": " + str(num_classes))

            for i in range(1, num_classes + 1):
                day_code = day[:3].upper()
                slot_id = studio.studio_id + "_" + day_code + "_" + room + "_" + str(i).zfill(2)
                class_slots.append(BaselineClassSlot(studio_id=studio.studio_id, slot_id=slot_id, day_of_week=day, daily_slot_index=i, room=room, capacity=room_capacity, usual_instructor=None))

    if len(class_slots) != studio.weekly_class_count:
        raise RuntimeError("Expected " + str(studio.weekly_class_count) + " class slots, but created " + str(len(class_slots)))

    return class_slots

def create_network_class_slots(studios: dict[str, Studio]) -> list[BaselineClassSlot]:
    #create all persistent recurring class slots for the entire network
    if not isinstance(studios, dict):
        raise TypeError("studios must be a dictionary of Studio objects")

    all_class_slots: list[BaselineClassSlot] = []
    seen_slots: set[str] = set()

    # iterate over all studios in the network
    for studio_id, studio in studios.items():
        # validate input types
        if not isinstance(studio, Studio):
            raise TypeError("Studio " + studio_id + " is not a Studio object")

        studio_slots = create_studio_class_slots(studio)
        for slot in studio_slots:
            if slot.slot_id in seen_slots:
                raise ValueError("Duplicate slot ID found: " + slot.slot_id)
            seen_slots.add(slot.slot_id)
            all_class_slots.append(slot)

    expected_slots = sum(studio.weekly_class_count for studio in studios.values())
    if len(all_class_slots) != expected_slots:
        raise RuntimeError("Expected " + str(expected_slots) + " class slots, but created " + str(len(all_class_slots)))

    return all_class_slots

