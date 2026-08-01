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

# generate a unique key for a session
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
