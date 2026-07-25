# Studio objects for the SoulCycle network simulation.
# A studio is a persistent location in the network.
# Weekly class sessions and instructor assignments are handled elsewhere.

from dataclasses import dataclass, field

#these are the market tiers we use across the project
MARKET_TIERS = {"Mega", "Large", "Medium", "Concentrated"}

@dataclass
class Studio:
    studio_id: str
    studio_name: str
    official_region: str #the region SoulCycle officially assigns the studio to
    network_market: str #the broader market the studio belongs to, like DMV or Greater NYC
    market_tier: str
    local_ridership_cluster: str #where riders around this studio mostly come from
    weekly_class_count: int #total classes offered at this studio in a normal week
    class_capacity: int #bikes available in a class at this studio
    daily_class_counts: dict[str, int] = field(default_factory=dict) #filled in later by studio_schedule

    def __post_init__(self) -> None:
        #remove whitespace from the studio's attributes
        self.studio_id = self.studio_id.strip()
        self.studio_name = self.studio_name.strip()
        self.official_region = self.official_region.strip()
        self.network_market = self.network_market.strip()
        self.market_tier = self.market_tier.strip()
        self.local_ridership_cluster = self.local_ridership_cluster.strip()

        #check that the studio's attributes are not empty
        if not self.studio_id:
            raise ValueError("studio_id cannot be empty.")
        if not self.studio_name:
            raise ValueError("Studio " + self.studio_id + " must have a studio_name.")
        if not self.official_region:
            raise ValueError("Studio " + self.studio_id + " must have an official_region.")
        if not self.network_market:
            raise ValueError("Studio " + self.studio_id + " must have a network_market.")
        if not self.local_ridership_cluster:
            raise ValueError("Studio " + self.studio_id + " must have a local_ridership_cluster.")

        #check that the market_tier is valid
        if self.market_tier not in MARKET_TIERS:
            raise ValueError("Invalid market_tier '" + self.market_tier + "' for studio " + self.studio_id + ". Expected one of " + str(sorted(MARKET_TIERS)) + ".")

        #check that weekly_class_count is an integer
        #bool counts as a subclass of int in python so we exclude it here
        if isinstance(self.weekly_class_count, bool) or not isinstance(self.weekly_class_count, int):
            raise TypeError("weekly_class_count for " + self.studio_id + " must be an integer.")

        #check that weekly_class_count is positive
        if self.weekly_class_count <= 0:
            raise ValueError("weekly_class_count for " + self.studio_id + " must be positive.")

        #check that class_capacity is an integer
        if isinstance(self.class_capacity, bool) or not isinstance(self.class_capacity, int):
            raise TypeError("class_capacity for " + self.studio_id + " must be an integer.")

        #check that class_capacity is positive
        if self.class_capacity <= 0:
            raise ValueError("class_capacity for " + self.studio_id + " must be positive.")

        #check that daily_class_counts is a dictionary
        #this starts empty and gets populated when we build the weekly schedule
        if not isinstance(self.daily_class_counts, dict):
            raise TypeError("daily_class_counts must be a dictionary.")
