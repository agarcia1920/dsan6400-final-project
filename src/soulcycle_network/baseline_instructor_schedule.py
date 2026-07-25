# Assign each instructor to their baseline recurring slots.

import numpy as np
from soulcycle_network.baseline_class_slot import BaselineClassSlot
from soulcycle_network.config import DAYS_OF_WEEK, DAY_INDEX
from soulcycle_network.instructor import Instructor

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
        if slot.usual_instructor is None:
            open_by_studio.setdefault(slot.studio_id, []).append(slot)

    ids = list(instructors.keys())
    rng.shuffle(ids)

    for iid in ids:
        instructor = instructors[iid]
        if not isinstance(instructor, Instructor):
            raise TypeError("Value stored under " + iid + " must be an Instructor object.")

        slot_ids: list[str] = []
        day_counts = {day: 0 for day in DAYS_OF_WEEK}

        for sid, n in instructor.baseline_studio_allocations.items():
            open_slots = [slot for slot in open_by_studio.get(sid, []) if slot.usual_instructor is None]
            if len(open_slots) < n:
                raise RuntimeError("Studio " + sid + " does not have enough open slots for instructor " + iid + ".")

            picks = rng.choice(len(open_slots), size=n, replace=False)
            for idx in picks:
                slot = open_slots[int(idx)]
                slot.usual_instructor = instructor.instructor_id
                slot_ids.append(slot.slot_id)
                day_counts[slot.day_of_week] += 1

        if len(slot_ids) != instructor.baseline_class_count:
            raise RuntimeError("Instructor " + iid + " was assigned " + str(len(slot_ids)) + " slots, but baseline_class_count is " + str(instructor.baseline_class_count) + ".")

        slot_ids.sort(key=lambda sid: (DAY_INDEX[lookup[sid].day_of_week], lookup[sid].daily_slot_index, sid))
        instructor.baseline_slot_ids = slot_ids
        instructor.baseline_day_counts = {day: n for day, n in day_counts.items() if n > 0}

    validate_baseline(instructors, class_slots)
    return class_slots

def validate_baseline(instructors: dict[str, Instructor], class_slots: list[BaselineClassSlot]) -> None:
    if not isinstance(instructors, dict):
        raise TypeError("instructors must be a dictionary.")
    if not isinstance(class_slots, list):
        raise TypeError("class_slots must be a list.")

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

        for sid in instructor.baseline_slot_ids:
            if sid in seen_slots:
                raise ValueError("Slot " + sid + " was assigned to more than one instructor.")
            seen_slots.add(sid)

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
