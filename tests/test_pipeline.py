# Fixed seeds should repeat, and a full experiment seed should write the expected CSV files.

from soulcycle_network.experiment_runner import SCENARIOS, run_experiment_seed
from soulcycle_network.simulation import init_simulation, run_simulation, summarize_simulation


def test_short_simulation(studios_csv_path, active_instructors_csv_path, instructor_sample_csv_path, rng, fake):
    ctx = init_simulation(
        studios_csv_path,
        active_instructors_csv_path,
        instructor_sample_csv_path,
        rng,
        fake,
        n_riders=200,
    )
    result = run_simulation(ctx, rng, n_weeks=4)
    assert len(result.week_results) == 4
    assert summarize_simulation(result, len(ctx.instructors))["total_attendance"] > 0


def test_fixed_seed_reproducible(studios_csv_path, active_instructors_csv_path, instructor_sample_csv_path, fake):
    import numpy as np

    def run_once(seed: int):
        rng = np.random.default_rng(seed)
        ctx = init_simulation(
            studios_csv_path,
            active_instructors_csv_path,
            instructor_sample_csv_path,
            rng,
            fake,
            n_riders=120,
        )
        result = run_simulation(ctx, rng, n_weeks=3)
        return len(result.network_state.co_counts)

    assert run_once(6400) == run_once(6400)


def test_activity_frequency_tertiles():
    import pandas as pd

    from soulcycle_network.analysis.metrics import assign_activity_frequency_tier

    nodes = pd.DataFrame(
        {"rider_id": [f"R{i:04d}" for i in range(9)], "baseline_annual_ride_rate": list(range(1, 10))}
    )
    tiered = assign_activity_frequency_tier(nodes)
    assert tiered["activity_frequency_tier"].nunique() == 3
    assert "activity_tertile_q33" in tiered.columns


def test_experiment_seed_exports(tmp_path, studios_csv_path):
    data_dir = studios_csv_path.parent
    longitudinal = run_experiment_seed(
        data_dir,
        SCENARIOS["no_coordination"],
        seed=6400,
        output_dir=tmp_path,
        n_weeks=2,
        n_riders=80,
    )
    out = tmp_path / "no_coordination" / "seed_6400"
    for name in (
        "attendance.csv",
        "node_attributes.csv",
        "pair_history.csv",
        "longitudinal_metrics.csv",
    ):
        assert (out / name).exists()
    assert not longitudinal.empty
