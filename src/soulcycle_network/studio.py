"""
Studio objects for the SoulCycle network simulation.

This module defines the persistent characteristics of each studio.
Weekly class sessions and instructor assignments will be handled in
separate scheduling modules.
"""

from dataclasses import dataclass, field

MARKET_TIERS = {
    "Mega",
    "Large",
    "Medium",
    "Concentrated"
}

@dataclass
class Studio:
    """
    A SoulCycle studio.
    """
    studio_id: str
    studio_name: str
    official_region: str
    network_market: str
    market_tier: str
    local_ridership_cluster: str
    weekly_class_count: int
    class_capacity: int

    daily_class_counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Validate and initialize the studio.
        """
        #remove whitespace from the studio's attributes
        self.studio_id = self.studio_id.strip()
        self.studio_name = self.studio_name.strip()
        self.official_region = self.official_region.strip()
        self.network_market = self.network_market.strip()
        self.market_tier = self.market_tier.strip()
        self.local_ridership_cluster = self.local_ridership_cluster.strip()

        # check that the studio's attributes are not empty
        if not self.studio_id:
            raise ValueError("studio_id cannot be empty.")

        if not self.studio_name:
            raise ValueError(f"Studio {self.studio_id} must have a studio_name.")

        if not self.official_region:
            raise ValueError(f"Studio {self.studio_id} must have an official_region.")

        if not self.network_market:
            raise ValueError(f"Studio {self.studio_id} must have a network_market.")

        if not self.local_ridership_cluster:
            raise ValueError(f"Studio {self.studio_id} must have a local_ridership_cluster.")

        # check that the market_tier is valid
        if self.market_tier not in MARKET_TIERS:
            raise ValueError(
                f"Invalid market_tier '{self.market_tier}' for "
                f"studio {self.studio_id}. Expected one of "
                f"{sorted(MARKET_TIERS)}."
            )
        
        # check that weekly_class_count is an integer
        if (isinstance(self.weekly_class_count, bool) or not isinstance(self.weekly_class_count, int)):
            raise TypeError(f"weekly_class_count for {self.studio_id} must be an integer.")

        # check that weekly_class_count is positive
        if self.weekly_class_count <= 0:
            raise ValueError(f"weekly_class_count for {self.studio_id} must be positive.")

        # check that class_capacity is an integer
        if (isinstance(self.class_capacity, bool) or not isinstance(self.class_capacity, int)):
            raise TypeError(f"class_capacity for {self.studio_id} must be an integer.")

        # check that class_capacity is positive
        if self.class_capacity <= 0:
            raise ValueError(f"class_capacity for {self.studio_id} must be positive.")

        # check that daily_class_counts is a dictionary
        if not isinstance(self.daily_class_counts, dict):
            raise TypeError("daily_class_counts must be a dictionary.")