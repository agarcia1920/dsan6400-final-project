# Assign each instructor to their baseline recurring slots.

import numpy as np
from soulcycle_network.baseline_class_slot import BaselineClassSlot
from soulcycle_network.config import DAYS_OF_WEEK, DAY_INDEX
from soulcycle_network.instructor import Instructor

MAX_ASSIGN_TRIES = 50

def reset_baseline_assignment(instructors: dict[str, Instructor], class_slots: list[BaselineClassSlot]) -> None:
    for slot in class_slots:
        slot.usual_instructor = None
    for instructor in instructors.values():
        instructor.baseline_slot_ids = []
        instructor.baseline_day_counts = {}

def instructor_order(instructors: dict[str, Instructor], ids: list[str]) -> list[str]:
    ordered = list(ids)
    ordered.sort(key=lambda iid: (len(instructors[iid].regular_studio_assignments), instructors[iid].baseline_class_count), reverse=True)
    return ordered

def try_assign_baseline_slots(instructors: dict[str, Instructor], lookup: dict[str, BaselineClassSlot], open_by_studio: dict[str, list[BaselineClassSlot]], ids: list[str], rng: np.random.Generator) -> None:
    for iid in ids:
        instructor = instructors[iid]
        if not isinstance(instructor, Instructor):
            raise TypeError("Value stored under " + iid + " must be an Instructor object.")

        slot_ids: list[str] = []
        day_counts = {day: 0 for day in DAYS_OF_WEEK}
        busy_times: set[tuple[str, int]] = set()
        studio_allocs = list(instructor.baseline_studio_allocations.items())
        rng.shuffle(studio_allocs)

        for sid, n in studio_allocs:
            for _ in range(n):
                eligible = [slot for slot in open_by_studio.get(sid, []) if slot.usual_instructor is None and (slot.day_of_week, slot.daily_slot_index) not in busy_times]
                if not eligible:
                    raise RuntimeError("Studio " + sid + " does not have enough conflict-free slots for instructor " + iid + ".")

                pick = int(rng.choice(len(eligible)))
                slot = eligible[pick]
                slot.usual_instructor = instructor.instructor_id
                slot_ids.append(slot.slot_id)
                busy_times.add((slot.day_of_week, slot.daily_slot_index))
                day_counts[slot.day_of_week] += 1

        if len(slot_ids) != instructor.baseline_class_count:
            raise RuntimeError("Instructor " + iid + " was assigned " + str(len(slot_ids)) + " slots, but baseline_class_count is " + str(instructor.baseline_class_count) + ".")

        slot_ids.sort(key=lambda sid: (DAY_INDEX[lookup[sid].day_of_week], lookup[sid].daily_slot_index, sid))
        instructor.baseline_slot_ids = slot_ids
        instructor.baseline_day_counts = {day: n for day, n in day_counts.items() if n > 0}

def assign_baseline_slots(instructors: dict[str, Instructor], class_slots: list[BaselineClassSlot], rng: np.random.Generator) -> list[BaselineClassSlot]:
    if not isinstance(instructors, dict):
        raise TypeError("instructors must be a dictionary.")
    if not isinstance(class_slots, list):
        raise TypeError("class_slots must be a list.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")

    lookup = {slot.slot_id: slot for slot in class_slots}
    open_by_studio: dict[str, list[BaselineClassSlot]] = {}

    for slot in class_slots:
        if not isinstance(slot, BaselineClassSlot):
            raise TypeError("class_slots must contain BaselineClassSlot objects.")
        open_by_studio.setdefault(slot.studio_id, []).append(slot)

    for attempt in range(MAX_ASSIGN_TRIES):
        reset_baseline_assignment(instructors, class_slots)

        ids = list(instructors.keys())
        rng.shuffle(ids)
        ids = instructor_order(instructors, ids)

        try:
            try_assign_baseline_slots(instructors, lookup, open_by_studio, ids, rng)
            validate_baseline(instructors, class_slots)
            return class_slots
        except RuntimeError:
            if attempt == MAX_ASSIGN_TRIES - 1:
                raise RuntimeError("Could not assign conflict-free baseline slots after " + str(MAX_ASSIGN_TRIES) + " tries.")

    raise RuntimeError("Could not assign conflict-free baseline slots.")

def validate_baseline(instructors: dict[str, Instructor], class_slots: list[BaselineClassSlot]) -> None:
    if not isinstance(instructors, dict):
        raise TypeError("instructors must be a dictionary.")
    if not isinstance(class_slots, list):
        raise TypeError("class_slots must be a list.")

    lookup = {slot.slot_id: slot for slot in class_slots}
    assigned = 0
    seen: set[str] = set()

    for slot in class_slots:
        if not isinstance(slot, BaselineClassSlot):
            raise TypeError("class_slots must contain BaselineClassSlot objects.")
        if slot.usual_instructor is None:
            raise ValueError("Slot " + slot.slot_id + " does not have a usual instructor.")
        if slot.slot_id in seen:
            raise ValueError("Duplicate slot ID found during validation: " + slot.slot_id)
        seen.add(slot.slot_id)
        assigned += 1

    total = 0
    seen_slots: set[str] = set()

    for iid, instructor in instructors.items():
        if not isinstance(instructor, Instructor):
            raise TypeError("Value stored under " + iid + " must be an Instructor object.")
        if len(instructor.baseline_slot_ids) != instructor.baseline_class_count:
            raise ValueError("Instructor " + iid + " slot count does not match baseline_class_count.")
        if sum(instructor.baseline_day_counts.values()) != instructor.baseline_class_count:
            raise ValueError("Instructor " + iid + " day counts do not match baseline_class_count.")

        seen_times: set[tuple[str, int]] = set()
        for slot_id in instructor.baseline_slot_ids:
            if slot_id in seen_slots:
                raise ValueError("Slot " + slot_id + " was assigned to more than one instructor.")
            seen_slots.add(slot_id)

            slot = lookup[slot_id]
            time_key = (slot.day_of_week, slot.daily_slot_index)
            if time_key in seen_times:
                raise ValueError("Instructor " + iid + " has overlapping baseline slots at " + str(time_key) + ".")
            seen_times.add(time_key)

        total += instructor.baseline_class_count

    if assigned != len(class_slots):
        raise RuntimeError("Assigned slot count does not match class slot list length.")
    if total != assigned:
        raise RuntimeError("Total instructor baseline classes do not match assigned slot count.")

def day_load_totals(instructors: dict[str, Instructor]) -> dict[str, int]:
    if not isinstance(instructors, dict):
        raise TypeError("instructors must be a dictionary.")

    out = {day: 0 for day in DAYS_OF_WEEK}
    for instructor in instructors.values():
        for day, n in instructor.baseline_day_counts.items():
            out[day] += n
    return out
