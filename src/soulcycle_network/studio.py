# Studio objects for the SoulCycle network simulation.

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
    room_class_counts: dict[str, int] = field(default_factory=dict) #classes per room, like {"A": 36, "B": 4}
    room_capacities: dict[str, int] = field(default_factory=dict) #bikes per room, like {"A": 59, "B": 46}
    daily_class_counts: dict[str, int] = field(default_factory=dict) #total classes per day across all rooms
    room_daily_class_counts: dict[str, dict[str, int]] = field(default_factory=dict) #classes per day within each room

    def __post_init__(self) -> None:
        #check string fields before stripping
        if not isinstance(self.studio_id, str):
            raise TypeError("studio_id must be a string.")
        if not isinstance(self.studio_name, str):
            raise TypeError("studio_name must be a string.")
        if not isinstance(self.official_region, str):
            raise TypeError("official_region must be a string.")
        if not isinstance(self.network_market, str):
            raise TypeError("network_market must be a string.")
        if not isinstance(self.market_tier, str):
            raise TypeError("market_tier must be a string.")
        if not isinstance(self.local_ridership_cluster, str):
            raise TypeError("local_ridership_cluster must be a string.")

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
        if isinstance(self.weekly_class_count, bool) or not isinstance(self.weekly_class_count, int):
            raise TypeError("weekly_class_count for " + self.studio_id + " must be an integer.")
        if self.weekly_class_count <= 0:
            raise ValueError("weekly_class_count for " + self.studio_id + " must be positive.")

        #check room-level class counts and capacities
        if not isinstance(self.room_class_counts, dict):
            raise TypeError("room_class_counts must be a dictionary.")
        if not isinstance(self.room_capacities, dict):
            raise TypeError("room_capacities must be a dictionary.")
        if set(self.room_class_counts.keys()) != set(self.room_capacities.keys()):
            raise ValueError("Studio " + self.studio_id + " room_class_counts and room_capacities must have the same room keys.")

        total_room_classes = 0
        for room, class_count in self.room_class_counts.items():
            if not isinstance(room, str):
                raise TypeError("room keys in room_class_counts must be strings.")
            if isinstance(class_count, bool) or not isinstance(class_count, int):
                raise TypeError("room_class_counts for " + self.studio_id + " must contain integers.")
            if class_count < 0:
                raise ValueError("room_class_counts for " + self.studio_id + " cannot be negative.")
            total_room_classes += class_count

            capacity = self.room_capacities[room]
            if isinstance(capacity, bool) or not isinstance(capacity, int):
                raise TypeError("room_capacities for " + self.studio_id + " must contain integers.")
            if capacity < 0:
                raise ValueError("room_capacities for " + self.studio_id + " cannot be negative.")
            if class_count > 0 and capacity <= 0:
                raise ValueError("Active rooms for " + self.studio_id + " must have positive room_capacities.")

        if total_room_classes <= 0:
            raise ValueError("Studio " + self.studio_id + " must have at least one room with classes.")

        if total_room_classes != self.weekly_class_count:
            raise ValueError("Studio " + self.studio_id + " weekly_class_count must equal the sum of room_class_counts.")

        if not isinstance(self.daily_class_counts, dict):
            raise TypeError("daily_class_counts must be a dictionary.")
        if not isinstance(self.room_daily_class_counts, dict):
            raise TypeError("room_daily_class_counts must be a dictionary.")

    @property
    def weekly_bike_supply(self) -> int:
        #total weekly bike supply across all active rooms
        total = 0
        for room, class_count in self.room_class_counts.items():
            if class_count > 0:
                total += class_count * self.room_capacities[room]
        return total

    @property
    def active_rooms(self) -> list[str]:
        #rooms that actually offer classes in a normal week
        return sorted([room for room, class_count in self.room_class_counts.items() if class_count > 0])
