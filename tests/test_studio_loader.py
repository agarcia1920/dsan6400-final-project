from soulcycle_network.studio_loader import load_studios

def test_load_studios_returns_expected_count(studios_csv_path):
    studios = load_studios(studios_csv_path)

    assert len(studios) > 0
    assert "GTWN" in studios
    assert studios["GTWN"].network_market == "DMV"

def test_load_studios_preserves_room_level_data(studios_csv_path):
    studios = load_studios(studios_csv_path)

    georgetown = studios["GTWN"]
    assert georgetown.room_class_counts == {"A": 36, "B": 0}
    assert georgetown.room_capacities == {"A": 59, "B": 0}
    assert georgetown.weekly_class_count == 36

    east_83 = studios["E83"]
    assert east_83.room_class_counts["A"] == 48
    assert east_83.room_class_counts["B"] == 29
    assert east_83.room_capacities["A"] == 70
    assert east_83.room_capacities["B"] == 46
