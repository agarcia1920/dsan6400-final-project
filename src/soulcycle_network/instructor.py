# Instructor objects for the SoulCycle network simulation.
# An instructor is a persistent node in the network with a home market and regular teaching pattern.
# Weekly availability, substitutions, and slot assignments are handled elsewhere.

from dataclasses import dataclass, field

MARKET_TIERS = {"Mega", "Large", "Medium", "Concentrated"}

@dataclass
class Instructor:
    instructor_id: str
    instructor_name: str
    official_region: str
    network_market: str #the market this instructor mainly works in
    market_tier: str
    home_cluster: str | None = None #where in the market the instructor is based
    baseline_class_count: int | None = None #their normal weekly teaching load
    regular_studio_assignments: list[str] = field(default_factory=list) #studios they regularly teach at
    baseline_studio_allocations: dict[str, int] = field(default_factory=dict) #how many classes at each regular studio
    baseline_day_counts: dict[str, int] = field(default_factory=dict) #filled in later when we assign days
    baseline_slot_ids: list[str] = field(default_factory=list) #filled in later when we assign specific slots

    def __post_init__(self) -> None:
        #check that the basic string fields are actually strings
        if not isinstance(self.instructor_id, str):
            raise TypeError("instructor_id must be a string.")
        if not isinstance(self.instructor_name, str):
            raise TypeError("instructor_name must be a string.")
        if not isinstance(self.official_region, str):
            raise TypeError("official_region must be a string.")
        if not isinstance(self.network_market, str):
            raise TypeError("network_market must be a string.")
        if not isinstance(self.market_tier, str):
            raise TypeError("market_tier must be a string.")

        #remove whitespace from the instructor's attributes
        self.instructor_id = self.instructor_id.strip()
        self.instructor_name = self.instructor_name.strip()
        self.official_region = self.official_region.strip()
        self.network_market = self.network_market.strip()
        self.market_tier = self.market_tier.strip()

        #check that the required fields are not empty
        if not self.instructor_id:
            raise ValueError("instructor_id cannot be empty.")
        if not self.instructor_name:
            raise ValueError("Instructor " + self.instructor_id + " must have an instructor_name.")
        if not self.official_region:
            raise ValueError("Instructor " + self.instructor_id + " must have an official_region.")
        if not self.network_market:
            raise ValueError("Instructor " + self.instructor_id + " must have a network_market.")
        if self.market_tier not in MARKET_TIERS:
            raise ValueError("Invalid market_tier '" + self.market_tier + "' for instructor " + self.instructor_id + ". Expected one of " + str(sorted(MARKET_TIERS)) + ".")

        #home_cluster is optional during initial creation
        if self.home_cluster is not None:
            if not isinstance(self.home_cluster, str):
                raise TypeError("home_cluster for " + self.instructor_id + " must be a string or None.")
            self.home_cluster = self.home_cluster.strip()
            if not self.home_cluster:
                raise ValueError("home_cluster for " + self.instructor_id + " cannot be an empty string.")

        #baseline_class_count is optional during initial creation
        if self.baseline_class_count is not None:
            if isinstance(self.baseline_class_count, bool) or not isinstance(self.baseline_class_count, int):
                raise TypeError("baseline_class_count for " + self.instructor_id + " must be an integer or None.")
            if self.baseline_class_count <= 0:
                raise ValueError("baseline_class_count for " + self.instructor_id + " must be positive.")

        #check that the schedule-related fields have the right types even if they are empty for now
        if not isinstance(self.regular_studio_assignments, list):
            raise TypeError("regular_studios for " + self.instructor_id + " must be a list.")
        if not isinstance(self.baseline_studio_allocations, dict):
            raise TypeError("baseline_studio_allocation for " + self.instructor_id + " must be a dictionary.")
        if not isinstance(self.baseline_day_counts, dict):
            raise TypeError("baseline_day_counts for " + self.instructor_id + " must be a dictionary.")
        if not isinstance(self.baseline_slot_ids, list):
            raise TypeError("baseline_slot_ids for " + self.instructor_id + " must be a list.")
