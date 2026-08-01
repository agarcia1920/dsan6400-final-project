from soulcycle_network.config import MAX_CLASSES_PER_DAY
from soulcycle_network.rider_generator import generate_riders, load_studio_data
from soulcycle_network.instructor_generator import generate_instructors
from soulcycle_network.simulation import build_cluster_studios, init_simulation
from soulcycle_network.weekly_booking import book_week, build_bookable_sessions, draw_weekly_counts
from soulcycle_network.weekly_schedule import create_weekly_schedule
from soulcycle_network.rider_parameters import simulation_scale
from soulcycle_network.rider_coordination import plan_coordination
from soulcycle_network.network_formation import empty_network

def test_build_bookable_sessions_applies_scale(initialized_environment, studios_csv_path, active_instructors_csv_path, instructor_sample_csv_path, rng, fake):
    instructors, baseline_slots, _ = initialized_environment
    schedule = create_weekly_schedule(1, instructors, baseline_slots, rng)
    studio_data = load_studio_data(studios_csv_path)
    studio_markets = studio_data.set_index("studio_id")["network_market"].astype(str).to_dict()
    scale = 0.05

    bookable = build_bookable_sessions(schedule, scale, studio_markets)
    session = next(iter(bookable.values()))
    assert session.sim_capacity == max(1, round(session.real_capacity * scale))

def test_book_week_respects_one_class_per_day(initialized_environment, studios_csv_path, active_instructors_csv_path, instructor_sample_csv_path, rng, fake):
    instructors, baseline_slots, _ = initialized_environment
    riders = generate_riders(studios_csv_path, instructors, rng, fake, n_riders=300)
    schedule = create_weekly_schedule(1, instructors, baseline_slots, rng)

    studio_data = load_studio_data(studios_csv_path)
    studio_markets = studio_data.set_index("studio_id")["network_market"].astype(str).to_dict()
    cluster_studios = build_cluster_studios(studio_data)
    scale = simulation_scale(200000, 300)

    weekly_counts = draw_weekly_counts(riders, rng)
    booking = book_week(schedule, riders, scale, studio_markets, cluster_studios, weekly_counts, [], rng)

    by_rider_day: dict[tuple[str, str], int] = {}
    for record in booking.records:
        key = (record.rider_id, record.day_of_week)
        by_rider_day[key] = by_rider_day.get(key, 0) + 1

    assert max(by_rider_day.values()) <= MAX_CLASSES_PER_DAY

def test_book_week_never_exceeds_scaled_capacity(initialized_environment, studios_csv_path, active_instructors_csv_path, instructor_sample_csv_path, rng, fake):
    instructors, baseline_slots, _ = initialized_environment
    riders = generate_riders(studios_csv_path, instructors, rng, fake, n_riders=500)
    schedule = create_weekly_schedule(1, instructors, baseline_slots, rng)

    studio_data = load_studio_data(studios_csv_path)
    studio_markets = studio_data.set_index("studio_id")["network_market"].astype(str).to_dict()
    cluster_studios = build_cluster_studios(studio_data)
    scale = simulation_scale(200000, 500)

    weekly_counts = draw_weekly_counts(riders, rng)
    booking = book_week(schedule, riders, scale, studio_markets, cluster_studios, weekly_counts, [], rng)
    bookable = build_bookable_sessions(schedule, scale, studio_markets)

    for slot_id, rider_ids in booking.enrollments.items():
        assert len(rider_ids) <= bookable[slot_id].sim_capacity

def test_plan_coordination_respects_partner_cap(initialized_environment, studios_csv_path, active_instructors_csv_path, instructor_sample_csv_path, rng, fake):
    instructors, baseline_slots, _ = initialized_environment
    riders = generate_riders(studios_csv_path, instructors, rng, fake, n_riders=50)
    state = empty_network()

    active_ids = sorted(riders.keys())[:6]
    for i in range(len(active_ids)):
        for j in range(i + 1, len(active_ids)):
            key = (active_ids[i], active_ids[j]) if active_ids[i] < active_ids[j] else (active_ids[j], active_ids[i])
            state.co_counts[key] = 6
            state.tie_strength[key] = 10.0

    weekly_counts = {rid: 2 for rid in active_ids}
    for rid in riders:
        if rid not in weekly_counts:
            weekly_counts[rid] = 0

    pairs = plan_coordination(riders, state, weekly_counts, rng)
    partner_use: dict[str, int] = {}
    for pair in pairs:
        partner_use[pair.rider_a] = partner_use.get(pair.rider_a, 0) + 1
        partner_use[pair.rider_b] = partner_use.get(pair.rider_b, 0) + 1

    assert max(partner_use.values()) <= 2

def test_init_simulation_builds_context(studios_csv_path, active_instructors_csv_path, instructor_sample_csv_path, rng, fake):
    ctx = init_simulation(studios_csv_path, active_instructors_csv_path, instructor_sample_csv_path, rng, fake, n_riders=100)
    assert len(ctx.riders) == 100
    assert ctx.scale > 0
    assert ctx.implied_population > 0
    assert ctx.studio_markets
    assert ctx.cluster_studios
