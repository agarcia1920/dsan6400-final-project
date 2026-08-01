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
