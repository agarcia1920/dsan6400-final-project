# Instructor objects for the SoulCycle network simulation.

from dataclasses import dataclass, field

MARKET_TIERS = {"Mega", "Large", "Medium", "Concentrated"}

@dataclass
class Instructor:
    instructor_id: str
    instructor_name: str
    network_market: str #the market this instructor mainly works in
    market_tier: str
    home_cluster: str | None = None #where in the market the instructor is based
    baseline_class_count: int | None = None #their normal weekly teaching load
    regular_studio_assignments: list[str] = field(default_factory=list) #studios they regularly teach at
    baseline_studio_allocations: dict[str, int] = field(default_factory=dict) #how many classes at each regular studio
    baseline_day_counts: dict[str, int] = field(default_factory=dict) #filled in later when we assign days
    baseline_slot_ids: list[str] = field(default_factory=list) #filled in later when we assign specific slots

    def __post_init__(self) -> None:
        #validate the input types and values
        if not isinstance(self.instructor_id, str):
            raise TypeError("instructor_id must be a string.")
        if not isinstance(self.instructor_name, str):
            raise TypeError("instructor_name must be a string.")
        if not isinstance(self.network_market, str):
            raise TypeError("network_market must be a string.")
        if not isinstance(self.market_tier, str):
            raise TypeError("market_tier must be a string.")
        
        #strip whitespace
        self.instructor_id = self.instructor_id.strip()
        self.instructor_name = self.instructor_name.strip()
        self.network_market = self.network_market.strip()
        self.market_tier = self.market_tier.strip()

        #validate the input values further
        if not self.instructor_id:
            raise ValueError("instructor_id cannot be empty.")
        if not self.instructor_name:
            raise ValueError("Instructor " + self.instructor_id + " must have an instructor_name.")
        if not self.network_market:
            raise ValueError("Instructor " + self.instructor_id + " must have a network_market.")
        if self.market_tier not in MARKET_TIERS:
            raise ValueError("Invalid market_tier '" + self.market_tier + "' for instructor " + self.instructor_id + ". Expected one of " + str(sorted(MARKET_TIERS)) + ".")

        #validate the home cluster
        if self.home_cluster is not None:
            if not isinstance(self.home_cluster, str):
                raise TypeError("home_cluster for " + self.instructor_id + " must be a string or None.")
            self.home_cluster = self.home_cluster.strip()
            if not self.home_cluster:
                raise ValueError("home_cluster for " + self.instructor_id + " cannot be an empty string.")

        #validate the baseline class count
        if self.baseline_class_count is not None:
            if isinstance(self.baseline_class_count, bool) or not isinstance(self.baseline_class_count, int):
                raise TypeError("baseline_class_count for " + self.instructor_id + " must be an integer or None.")
            if self.baseline_class_count <= 0:
                raise ValueError("baseline_class_count for " + self.instructor_id + " must be positive.")

        #validate the regular studio assignments
        if not isinstance(self.regular_studio_assignments, list):
            raise TypeError("regular_studios for " + self.instructor_id + " must be a list.")
        if not isinstance(self.baseline_studio_allocations, dict):
            raise TypeError("baseline_studio_allocation for " + self.instructor_id + " must be a dictionary.")
        if not isinstance(self.baseline_day_counts, dict):
            raise TypeError("baseline_day_counts for " + self.instructor_id + " must be a dictionary.")
        if not isinstance(self.baseline_slot_ids, list):
            raise TypeError("baseline_slot_ids for " + self.instructor_id + " must be a list.")
