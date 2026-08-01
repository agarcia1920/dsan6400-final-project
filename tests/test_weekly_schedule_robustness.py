import numpy as np
from soulcycle_network.config import PROB_OFF_WEEK
from soulcycle_network.weekly_schedule import create_weekly_schedule, summarize_weekly_simulation, validate_weekly_schedule

def test_weekly_schedule_across_many_weeks(initialized_environment):
    instructors, baseline_slots, baseline_snapshot = initialized_environment

    for seed in range(6400, 6410):
        rng = np.random.default_rng(seed)
        week_results = []

        for week_number in range(1, 53):
            result = create_weekly_schedule(week_number, instructors, baseline_slots, rng)

            assert len(result.sessions) == len(baseline_slots)
            assert len(result.uncovered_session_ids) == 0
            assert all(session.week_number == week_number for session in result.sessions)

            validate_weekly_schedule(
                result.sessions,
                instructors,
                baseline_slots,
                result.off_instructor_ids,
                week_number=week_number,
                baseline_snapshot=baseline_snapshot,
            )

            week_results.append(result)

        summary = summarize_weekly_simulation(week_results, len(instructors))
        assert summary["total_uncovered_sessions"] == 0.0
        assert summary["max_uncovered_in_week"] == 0.0
        assert 0.05 < summary["observed_off_rate"] < 0.11

def test_baseline_slots_unchanged_after_52_weeks(initialized_environment):
    instructors, baseline_slots, baseline_snapshot = initialized_environment
    rng = np.random.default_rng(6400)

    for week_number in range(1, 53):
        create_weekly_schedule(week_number, instructors, baseline_slots, rng)

    for slot in baseline_slots:
        assert slot.usual_instructor == baseline_snapshot[slot.slot_id]

def test_weekly_schedule_aggregate_substitutions(initialized_environment):
    instructors, baseline_slots, _baseline_snapshot = initialized_environment
    rng = np.random.default_rng(6400)
    week_results = []

    for week_number in range(1, 53):
        week_results.append(create_weekly_schedule(week_number, instructors, baseline_slots, rng))

    summary = summarize_weekly_simulation(week_results, len(instructors))

    assert summary["avg_substitutions_per_week"] > 0.0
    assert summary["max_substitutions_in_week"] >= summary["avg_substitutions_per_week"]
    assert abs(summary["observed_off_rate"] - PROB_OFF_WEEK) < 0.03
