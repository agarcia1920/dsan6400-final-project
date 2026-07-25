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
