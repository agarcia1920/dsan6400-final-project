# Covers studios, booking limits, tie thresholds, and basic graph construction rules.

import networkx as nx

from soulcycle_network.config import (
    MAX_CLASSES_PER_DAY,
    MIN_CLASSES_FOR_FAMILIARITY,
    MIN_CLASSES_FOR_SOCIAL_TIE,
    TIE_DECAY_RATE,
)
from soulcycle_network.analysis.metrics import build_rider_graph, isolate_gcc
from soulcycle_network.analysis.models import nonlinear_preferential_attachment
from soulcycle_network.instructors import generate_instructors
from soulcycle_network.network_formation import (
    NetworkState,
    decay_ties,
    familiarity_pairs,
    pair_key,
    social_tie_pairs,
    update_from_enrollments,
)
from soulcycle_network.riders import generate_riders, load_studio_data, simulation_scale
from soulcycle_network.simulation import build_cluster_studios
from soulcycle_network.studios import load_studios
from soulcycle_network.weekly import book_week, build_bookable_sessions, create_weekly_schedule, draw_weekly_counts


def test_load_studios(studios_csv_path):
    studios = load_studios(studios_csv_path)
    assert len(studios) > 0
    assert studios["GTWN"].network_market == "DMV"


def test_initialized_environment_has_slots(initialized_environment):
    instructors, baseline_slots, snapshot = initialized_environment
    assert instructors and baseline_slots and snapshot


def test_build_bookable_sessions_applies_scale(initialized_environment, studios_csv_path, rng, fake):
    instructors, baseline_slots, _ = initialized_environment
    schedule = create_weekly_schedule(1, instructors, baseline_slots, rng)
    studio_data = load_studio_data(studios_csv_path)
    studio_markets = studio_data.set_index("studio_id")["network_market"].astype(str).to_dict()
    bookable = build_bookable_sessions(schedule, 0.05, studio_markets)
    session = next(iter(bookable.values()))
    assert session.sim_capacity == max(1, round(session.real_capacity * 0.05))


def test_book_week_one_class_per_day(initialized_environment, studios_csv_path, rng, fake):
    instructors, baseline_slots, _ = initialized_environment
    riders = generate_riders(studios_csv_path, instructors, rng, fake, n_riders=300)
    schedule = create_weekly_schedule(1, instructors, baseline_slots, rng)
    studio_data = load_studio_data(studios_csv_path)
    studio_markets = studio_data.set_index("studio_id")["network_market"].astype(str).to_dict()
    cluster_studios = build_cluster_studios(studio_data)
    scale = simulation_scale(200_000, 300)
    weekly_counts = draw_weekly_counts(riders, rng)
    booking = book_week(schedule, riders, scale, studio_markets, cluster_studios, weekly_counts, [], rng)
    by_rider_day: dict[tuple[str, str], int] = {}
    for record in booking.records:
        key = (record.rider_id, record.day_of_week)
        by_rider_day[key] = by_rider_day.get(key, 0) + 1
    assert max(by_rider_day.values()) <= MAX_CLASSES_PER_DAY


def test_pair_key_sorts_ids():
    assert pair_key("R000002", "R000001") == ("R000001", "R000002")


def test_familiarity_and_social_thresholds():
    state = NetworkState()
    key = ("R000001", "R000002")
    state.co_counts[key] = MIN_CLASSES_FOR_FAMILIARITY
    assert key in familiarity_pairs(state)
    assert key not in social_tie_pairs(state)
    state.co_counts[key] = MIN_CLASSES_FOR_SOCIAL_TIE
    state.tie_strength[key] = 6.0
    assert key in social_tie_pairs(state)
    state.tie_strength[key] = 0.5
    assert key not in social_tie_pairs(state)


def test_decay_ties():
    state = NetworkState()
    key = ("R000001", "R000002")
    state.tie_strength[key] = 10.0
    decay_ties(state, TIE_DECAY_RATE)
    assert abs(state.tie_strength[key] - 10.0 * TIE_DECAY_RATE) < 1e-9


def test_update_from_enrollments_counts_pairs():
    state = NetworkState()
    update_from_enrollments(state, {"slot_a": ["R000001", "R000002", "R000003"]})
    assert state.co_counts[("R000001", "R000002")] == 1


def test_isolate_gcc():
    graph = nx.Graph()
    graph.add_edges_from([("a", "b"), ("b", "c"), ("x", "y")])
    assert set(isolate_gcc(graph).nodes()) == {"a", "b", "c"}


def test_active_social_graph_requires_strength():
    records = [
        {"rider_1": "a", "rider_2": "b", "coattendance_count": 8, "tie_strength": 0.1},
        {"rider_1": "a", "rider_2": "c", "coattendance_count": 8, "tie_strength": 4.0},
    ]
    graph = build_rider_graph(["a", "b", "c"], records, graph_type="active_social")
    assert not graph.has_edge("a", "b")
    assert graph.has_edge("a", "c")


def test_preferential_attachment_reproducible():
    a = nonlinear_preferential_attachment(n=100, m_links=2, alpha=1.0, seed=6400)
    b = nonlinear_preferential_attachment(n=100, m_links=2, alpha=1.0, seed=6400)
    assert set(a.edges()) == set(b.edges())
