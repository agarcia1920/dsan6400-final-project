from soulcycle_network.config import DAYS_OF_WEEK
from soulcycle_network.studio_loader import load_studios
from soulcycle_network.studio_schedule import create_all_weekly_schedules, create_weekly_schedule, divide_classes

def test_divide_classes_preserves_weekly_total(rng):
    daily_schedule = divide_classes(36, rng)

    assert set(daily_schedule.keys()) == set(DAYS_OF_WEEK)
    assert sum(daily_schedule.values()) == 36

def test_create_weekly_schedule_builds_room_level_counts(studios_csv_path, rng):
    studios = load_studios(studios_csv_path)
    east_83 = studios["E83"]

    create_weekly_schedule(east_83, rng)

    assert sum(east_83.daily_class_counts.values()) == east_83.weekly_class_count
    assert sum(east_83.room_daily_class_counts["A"].values()) == 48
    assert sum(east_83.room_daily_class_counts["B"].values()) == 29

def test_create_all_weekly_schedules_updates_every_studio(studios_csv_path, rng):
    studios = load_studios(studios_csv_path)
    create_all_weekly_schedules(studios, rng)

    for studio in studios.values():
        assert studio.daily_class_counts
        assert sum(studio.daily_class_counts.values()) == studio.weekly_class_count
