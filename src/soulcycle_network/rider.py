# Rider objects for the SoulCycle network simulation.

from dataclasses import dataclass, field

import numpy as np

@dataclass
class Rider:
    rider_id: str
    rider_name: str
    home_market: str
    home_cluster: str
    baseline_annual_ride_rate: float #persistent annual ride propensity; weekly attendance is drawn from this
    preferred_studio_ids: list[str] = field(default_factory=list)
    preferred_instructor_ids: list[str] = field(default_factory=list)
    attended_session_ids: list[str] = field(default_factory=list) #filled in as the simulation runs
    attended_instructor_counts: dict[str, int] = field(default_factory=dict)
    attended_studio_counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        #validate the input types and values
        if not isinstance(self.rider_id, str):
            raise TypeError("rider_id must be a string.")
        if not isinstance(self.rider_name, str):
            raise TypeError("rider_name must be a string.")
        if not isinstance(self.home_market, str):
            raise TypeError("home_market must be a string.")
        if not isinstance(self.home_cluster, str):
            raise TypeError("home_cluster must be a string.")

        self.rider_id = self.rider_id.strip()
        self.rider_name = self.rider_name.strip()
        self.home_market = self.home_market.strip()
        self.home_cluster = self.home_cluster.strip()

        if not self.rider_id:
            raise ValueError("rider_id cannot be empty.")
        if not self.rider_name:
            raise ValueError("Rider " + self.rider_id + " must have a rider_name.")
        if not self.home_market:
            raise ValueError("Rider " + self.rider_id + " must have a home_market.")
        if not self.home_cluster:
            raise ValueError("Rider " + self.rider_id + " must have a home_cluster.")

        if isinstance(self.baseline_annual_ride_rate, bool) or not isinstance(
            self.baseline_annual_ride_rate,
            (int, float, np.integer, np.floating),
        ):
            raise TypeError("baseline_annual_ride_rate for " + self.rider_id + " must be a number.")
        self.baseline_annual_ride_rate = float(self.baseline_annual_ride_rate)
        if self.baseline_annual_ride_rate <= 0:
            raise ValueError("baseline_annual_ride_rate for " + self.rider_id + " must be positive.")

        if not isinstance(self.preferred_studio_ids, list):
            raise TypeError("preferred_studio_ids for " + self.rider_id + " must be a list.")
        if not isinstance(self.preferred_instructor_ids, list):
            raise TypeError("preferred_instructor_ids for " + self.rider_id + " must be a list.")
        if not isinstance(self.attended_session_ids, list):
            raise TypeError("attended_session_ids for " + self.rider_id + " must be a list.")

        for sid in self.preferred_studio_ids:
            if not isinstance(sid, str) or not sid.strip():
                raise ValueError("preferred_studio_ids for " + self.rider_id + " must contain non-empty strings only.")
        for iid in self.preferred_instructor_ids:
            if not isinstance(iid, str) or not iid.strip():
                raise ValueError("preferred_instructor_ids for " + self.rider_id + " must contain non-empty strings only.")
        for session_id in self.attended_session_ids:
            if not isinstance(session_id, str) or not session_id.strip():
                raise ValueError("attended_session_ids for " + self.rider_id + " must contain non-empty strings only.")

        if not isinstance(self.attended_instructor_counts, dict):
            raise TypeError("attended_instructor_counts for " + self.rider_id + " must be a dictionary.")
        if not isinstance(self.attended_studio_counts, dict):
            raise TypeError("attended_studio_counts for " + self.rider_id + " must be a dictionary.")

        for key, count in self.attended_instructor_counts.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("attended_instructor_counts keys for " + self.rider_id + " must be non-empty strings.")
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError("attended_instructor_counts values for " + self.rider_id + " must be integers.")
            if count < 0:
                raise ValueError("attended_instructor_counts for " + self.rider_id + " cannot be negative.")

        for key, count in self.attended_studio_counts.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("attended_studio_counts keys for " + self.rider_id + " must be non-empty strings.")
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError("attended_studio_counts values for " + self.rider_id + " must be integers.")
            if count < 0:
                raise ValueError("attended_studio_counts for " + self.rider_id + " cannot be negative.")
