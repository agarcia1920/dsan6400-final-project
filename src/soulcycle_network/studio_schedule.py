# Functions for creating persistent studio class schedules.
# Each room's weekly class count is divided across the seven days of the week.
# We are not assigning times yet, just how many classes happen on each day.

import numpy as np
from soulcycle_network.config import DAYS_OF_WEEK
from soulcycle_network.studio import Studio

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
