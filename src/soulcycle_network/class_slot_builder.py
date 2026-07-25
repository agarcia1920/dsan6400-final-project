# Functions for creating persistent baseline class slots.
# Each studio's daily class counts become individual recurring class-slot objects.

from soulcycle_network.class_session import ClassSession
from soulcycle_network.config import DAYS_OF_WEEK
from soulcycle_network.studio import Studio

def create_studio_class_slots(studio: Studio) -> list[ClassSession]:
    #create all persistent recurring class slots for one studio
    if not isinstance(studio, Studio):
        raise TypeError("studio must be a Studio object")

    #the studio needs a daily schedule before we can build slots
    if not studio.daily_class_counts:
        raise ValueError("Studio " + studio.studio_id + " does not have a daily schedule.")

    #make sure every day of the week is represented
    missing_days = set(DAYS_OF_WEEK) - set(studio.daily_class_counts)
    if missing_days:
        raise ValueError("Studio " + studio.studio_id + " is missing daily class counts for: " + str(sorted(missing_days)))

    class_slots: list[ClassSession] = []

    #loop through each day and create one slot per class on that day
    for day in DAYS_OF_WEEK:
        num_classes = studio.daily_class_counts[day]
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 0:
            raise ValueError("Studio " + studio.studio_id + " has invalid class count for " + day + ": " + str(num_classes))

        for i in range(1, num_classes + 1):
            day_code = day[:3].upper() #Monday becomes MON, Tuesday becomes TUE, etc.
            slot_id = studio.studio_id + "_" + day_code + "_" + str(i).zfill(2)
            class_slots.append(ClassSession(studio_id=studio.studio_id, slot_id=slot_id, day_of_week=day, daily_slot_index=i, capacity=studio.class_capacity, usual_instructor=None))

    if len(class_slots) != studio.weekly_class_count:
        raise RuntimeError("Expected " + str(studio.weekly_class_count) + " class slots, but created " + str(len(class_slots)))

    return class_slots

def create_network_class_slots(studios: dict[str, Studio]) -> list[ClassSession]:
    #create all persistent recurring class slots for the entire network
    if not isinstance(studios, dict):
        raise TypeError("studios must be a dictionary of Studio objects")

    all_class_slots: list[ClassSession] = []
    seen_slots: set[str] = set() #track slot ids so we catch duplicates across studios

    for studio_id, studio in studios.items():
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
