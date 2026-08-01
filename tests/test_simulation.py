from soulcycle_network.simulation import init_simulation, run_simulation, summarize_simulation

def test_run_simulation_short_horizon(studios_csv_path, active_instructors_csv_path, instructor_sample_csv_path, rng, fake):
    ctx = init_simulation(studios_csv_path, active_instructors_csv_path, instructor_sample_csv_path, rng, fake, n_riders=200)
    result = run_simulation(ctx, rng, n_weeks=4)

    assert len(result.week_results) == 4
    assert result.scale == ctx.scale
    summary = summarize_simulation(result, len(ctx.instructors))
    assert summary["total_attendance"] > 0

def test_simulation_builds_co_attendance_at_full_scale(studios_csv_path, active_instructors_csv_path, instructor_sample_csv_path, rng, fake):
    ctx = init_simulation(studios_csv_path, active_instructors_csv_path, instructor_sample_csv_path, rng, fake, n_riders=10000)
    result = run_simulation(ctx, rng, n_weeks=6)

    assert result.network_state.co_counts
    summary = summarize_simulation(result, len(ctx.instructors))
    assert summary["familiarity_pair_count"] >= 0.0

def test_full_year_simulation_smoke(studios_csv_path, active_instructors_csv_path, instructor_sample_csv_path, rng, fake):
    ctx = init_simulation(studios_csv_path, active_instructors_csv_path, instructor_sample_csv_path, rng, fake, n_riders=500)
    result = run_simulation(ctx, rng, n_weeks=52)
    summary = summarize_simulation(result, len(ctx.instructors))

    assert len(result.week_results) == 52
    assert summary["total_attendance"] > 0
    assert summary["observed_off_rate"] > 0
