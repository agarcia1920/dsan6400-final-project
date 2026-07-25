from soulcycle_network.class_slot_builder import create_network_class_slots, create_studio_class_slots
from soulcycle_network.studio_loader import load_studios
from soulcycle_network.studio_schedule import create_all_weekly_schedules

def test_create_studio_class_slots_uses_room_capacities(studios_csv_path, rng):
    studios = load_studios(studios_csv_path)
    east_83 = studios["E83"]
    create_all_weekly_schedules({east_83.studio_id: east_83}, rng)

    slots = create_studio_class_slots(east_83)

    assert len(slots) == 77
    room_a_slots = [slot for slot in slots if slot.room == "A"]
    room_b_slots = [slot for slot in slots if slot.room == "B"]
    assert len(room_a_slots) == 48
    assert len(room_b_slots) == 29
    assert all(slot.capacity == 70 for slot in room_a_slots)
    assert all(slot.capacity == 46 for slot in room_b_slots)

def test_create_network_class_slots_returns_full_network(studios_csv_path, rng):
    studios = load_studios(studios_csv_path)
    create_all_weekly_schedules(studios, rng)
    slots = create_network_class_slots(studios)

    assert len(slots) == sum(studio.weekly_class_count for studio in studios.values())
    assert len({slot.slot_id for slot in slots}) == len(slots)
