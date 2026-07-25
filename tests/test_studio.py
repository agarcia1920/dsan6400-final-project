import pytest
from soulcycle_network.studio import Studio

def test_studio_rejects_non_string_studio_id():
    with pytest.raises(TypeError):
        Studio(
            studio_id=123,
            studio_name="Georgetown",
            official_region="DC",
            network_market="DMV",
            market_tier="Large",
            local_ridership_cluster="DC-Arlington",
            weekly_class_count=36,
            room_class_counts={"A": 36},
            room_capacities={"A": 59},
        )

def test_studio_preserves_room_level_counts_and_capacities():
    studio = Studio(
        studio_id="E83",
        studio_name="East 83rd Street",
        official_region="NYC",
        network_market="Greater NYC",
        market_tier="Mega",
        local_ridership_cluster="Uptown Manhattan",
        weekly_class_count=77,
        room_class_counts={"A": 48, "B": 29},
        room_capacities={"A": 70, "B": 46},
    )

    assert studio.room_class_counts == {"A": 48, "B": 29}
    assert studio.room_capacities == {"A": 70, "B": 46}
    assert studio.weekly_bike_supply == (48 * 70) + (29 * 46)

def test_studio_weekly_class_count_must_match_room_totals():
    with pytest.raises(ValueError):
        Studio(
            studio_id="GTWN",
            studio_name="Georgetown",
            official_region="DC",
            network_market="DMV",
            market_tier="Large",
            local_ridership_cluster="DC-Arlington",
            weekly_class_count=40,
            room_class_counts={"A": 36},
            room_capacities={"A": 59},
        )
