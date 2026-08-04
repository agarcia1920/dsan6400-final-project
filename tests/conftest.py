# Shared pytest data paths, fixed RNG, Faker, and a small studio/instructor setup for tests.

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

@pytest.fixture
def studios_csv_path() -> Path:
    return DATA_DIR / "studios.csv"

@pytest.fixture
def active_instructors_csv_path() -> Path:
    return DATA_DIR / "active_instructors_final.csv"

@pytest.fixture
def instructor_sample_csv_path() -> Path:
    return DATA_DIR / "instructors_sample.csv"

@pytest.fixture
def rng():
    import numpy as np
    return np.random.default_rng(6400)

@pytest.fixture
def fake():
    from faker import Faker
    fake = Faker()
    Faker.seed(6400)
    return fake

@pytest.fixture
def initialized_environment(studios_csv_path, active_instructors_csv_path, instructor_sample_csv_path, rng, fake):
    from soulcycle_network.instructors import assign_baseline_slots
    from soulcycle_network.studios import create_network_class_slots
    from soulcycle_network.instructors import generate_instructors
    from soulcycle_network.studios import load_studios
    from soulcycle_network.studios import create_all_weekly_schedules
    from soulcycle_network.weekly import snapshot_baseline

    studios = load_studios(studios_csv_path)
    create_all_weekly_schedules(studios, rng)
    baseline_slots = create_network_class_slots(studios)
    instructors = generate_instructors(active_instructors_csv_path, instructor_sample_csv_path, studios_csv_path, rng, fake)
    assign_baseline_slots(instructors, baseline_slots, rng)
    baseline_snapshot = snapshot_baseline(baseline_slots)
    return instructors, baseline_slots, baseline_snapshot

@pytest.fixture
def simulation_context(studios_csv_path, active_instructors_csv_path, instructor_sample_csv_path, rng, fake):
    from soulcycle_network.simulation import init_simulation

    return init_simulation(studios_csv_path, active_instructors_csv_path, instructor_sample_csv_path, rng, fake, n_riders=300)
