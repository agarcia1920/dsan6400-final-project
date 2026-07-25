# Baseline class-slot objects for the SoulCycle network simulation.
# These are persistent recurring slots for each studio, not realized weekly sessions.

from dataclasses import dataclass
from soulcycle_network.config import DAYS_OF_WEEK

@dataclass
class BaselineClassSlot:
    studio_id: str
    slot_id: str #unique id for this recurring slot, like GTWN_MON_A_01
    day_of_week: str
    daily_slot_index: int #which class on that day within the room, starting at 1
    room: str #room label from the studio data, usually A or B
    capacity: int #bikes available in this room's class
    usual_instructor: str | None = None #filled in later when we assign instructors to slots

    def __post_init__(self):
        if not isinstance(self.studio_id, str):
            raise TypeError("studio_id must be a string.")
        if not isinstance(self.slot_id, str):
            raise TypeError("slot_id must be a string.")
        if not isinstance(self.day_of_week, str):
            raise TypeError("day_of_week must be a string.")
        if not isinstance(self.room, str):
            raise TypeError("room must be a string.")

        self.slot_id = self.slot_id.strip()
        self.studio_id = self.studio_id.strip()
        self.day_of_week = self.day_of_week.strip()
        self.room = self.room.strip()

        if not self.studio_id:
            raise ValueError("studio_id cannot be empty.")
        if not self.slot_id:
            raise ValueError("slot_id cannot be empty.")
        if not self.room:
            raise ValueError("room cannot be empty.")
        if self.day_of_week not in DAYS_OF_WEEK:
            raise ValueError("Invalid day of week '" + self.day_of_week + "' for class slot " + self.slot_id + ".")

        if isinstance(self.daily_slot_index, bool) or not isinstance(self.daily_slot_index, int):
            raise TypeError("daily_slot_index for " + self.slot_id + " must be an integer.")
        if self.daily_slot_index <= 0:
            raise ValueError("daily_slot_index for " + self.slot_id + " must be positive.")

        if isinstance(self.capacity, bool) or not isinstance(self.capacity, int):
            raise TypeError("capacity for " + self.slot_id + " must be an integer.")
        if self.capacity <= 0:
            raise ValueError("capacity for " + self.slot_id + " must be positive.")

        if self.usual_instructor is not None:
            if not isinstance(self.usual_instructor, str):
                raise TypeError("usual_instructor for " + self.slot_id + " must be a string or None.")
            self.usual_instructor = self.usual_instructor.strip()
            if not self.usual_instructor:
                raise ValueError("usual_instructor for " + self.slot_id + " cannot be an empty string.")
