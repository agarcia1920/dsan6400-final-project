# Exports attendance, pair history, node attributes, edge lists, and longitudinal tables

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from soulcycle_network.config import (
    MIN_ACTIVE_TIE_STRENGTH_FOR_SOCIAL_TIE,
    MIN_CLASSES_FOR_FAMILIARITY,
    MIN_CLASSES_FOR_SOCIAL_TIE,
)
from soulcycle_network.network_formation import (
    NetworkState,
    familiarity_pairs,
    social_tie_pairs,
)
from soulcycle_network.simulation import (
    SimulationContext,
    SimulationResult,
)


ATTENDANCE_COLUMNS = [
    "week_number",
    "session_id",
    "rider_id",
    "studio_id",
    "market",
    "home_market",
    "home_cluster",
    "actual_instructor_id",
    "day_of_week",
    "coordinated_booking",
]

WEEKLY_SUMMARY_COLUMNS = [
    "week_number",
    "attendance_count",
    "unique_riders",
    "unmet_demand",
    "coordinated_bookings",
    "seat_occupancy_rate",
    "substitutions",
    "uncovered_sessions",
]

PAIR_HISTORY_COLUMNS = [
    "rider_1",
    "rider_2",
    "coattendance_count",
    "tie_strength",
    "first_shared_week",
    "last_shared_week",
    "became_familiar_week",
    "became_social_week",
    "active_social_tie",
]

NODE_ATTRIBUTE_COLUMNS = [
    "rider_id",
    "home_market",
    "home_cluster",
    "annual_ride_rate",
    "total_attendance",
    "unique_studios",
    "unique_instructors",
    "familiarity_degree",
    "social_degree",
    "coordination_count",
]

EDGE_COLUMNS = [
    "rider_1",
    "rider_2",
    "coattendance_count",
    "tie_strength",
]


@dataclass
class PairWeekTracker:
    first_shared_week: dict[tuple[str, str], int] = field(
        default_factory=dict
    )
    last_shared_week: dict[tuple[str, str], int] = field(
        default_factory=dict
    )
    became_familiar_week: dict[tuple[str, str], int] = field(
        default_factory=dict
    )
    became_social_week: dict[tuple[str, str], int] = field(
        default_factory=dict
    )
    previous_co_counts: dict[tuple[str, str], int] = field(
        default_factory=dict
    )

    def update(
        self,
        state: NetworkState,
        week_number: int,
    ) -> None:
        for key, count in state.co_counts.items():
            previous_count = self.previous_co_counts.get(key, 0)

            if count > previous_count:
                if key not in self.first_shared_week:
                    self.first_shared_week[key] = week_number

                self.last_shared_week[key] = week_number
                self.previous_co_counts[key] = count

            if (
                count >= MIN_CLASSES_FOR_FAMILIARITY
                and key not in self.became_familiar_week
            ):
                self.became_familiar_week[key] = week_number

            strength = state.tie_strength.get(key, 0.0)
            if (
                count >= MIN_CLASSES_FOR_SOCIAL_TIE
                and strength
                >= MIN_ACTIVE_TIE_STRENGTH_FOR_SOCIAL_TIE
                and key not in self.became_social_week
            ):
                self.became_social_week[key] = week_number


def rider_lookup(ctx: SimulationContext) -> tuple[dict[str, str], dict[str, str]]:
    cluster = {rider.rider_id: rider.home_cluster for rider in ctx.riders.values()}
    market = {rider.rider_id: rider.home_market for rider in ctx.riders.values()}
    return cluster, market


def build_attendance_frame(
    ctx: SimulationContext,
    result: SimulationResult,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for week in result.week_results:
        for record in week.booking.records:
            rider = ctx.riders[record.rider_id]

            rows.append(
                {
                    "week_number": record.week_number,
                    "session_id": record.slot_id,
                    "rider_id": record.rider_id,
                    "studio_id": record.studio_id,
                    "market": ctx.studio_markets.get(
                        record.studio_id,
                        rider.home_market,
                    ),
                    "home_market": rider.home_market,
                    "home_cluster": rider.home_cluster,
                    "actual_instructor_id": (
                        record.assigned_instructor_id
                    ),
                    "day_of_week": record.day_of_week,
                    "coordinated_booking": record.is_coordinated,
                }
            )

    return pd.DataFrame(
        rows,
        columns=ATTENDANCE_COLUMNS,
    )


def build_weekly_summary_frame(
    result: SimulationResult,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for week in result.week_results:
        booking = week.booking
        occupancy = booking.seats_filled / booking.total_sim_seats if booking.total_sim_seats > 0 else 0.0

        rows.append(
            {
                "week_number": week.week_number,
                "attendance_count": len(booking.records),
                "unique_riders": len(
                    {
                        record.rider_id
                        for record in booking.records
                    }
                ),
                "unmet_demand": booking.unmet_demand,
                "coordinated_bookings": booking.coordinated_bookings,
                "seat_occupancy_rate": occupancy,
                "substitutions": week.schedule.substitution_count,
                "uncovered_sessions": len(
                    week.schedule.uncovered_session_ids
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=WEEKLY_SUMMARY_COLUMNS,
    )


def build_pair_history_frame(
    state: NetworkState,
    tracker: PairWeekTracker,
) -> pd.DataFrame:
    active_social = social_tie_pairs(state)
    rows: list[dict[str, object]] = []

    for key, count in state.co_counts.items():
        rider_1, rider_2 = key

        rows.append(
            {
                "rider_1": rider_1,
                "rider_2": rider_2,
                "coattendance_count": count,
                "tie_strength": state.tie_strength.get(
                    key,
                    0.0,
                ),
                "first_shared_week": (
                    tracker.first_shared_week.get(key)
                ),
                "last_shared_week": (
                    tracker.last_shared_week.get(key)
                ),
                "became_familiar_week": (
                    tracker.became_familiar_week.get(key)
                ),
                "became_social_week": (
                    tracker.became_social_week.get(key)
                ),
                "active_social_tie": key in active_social,
            }
        )

    return pd.DataFrame(
        rows,
        columns=PAIR_HISTORY_COLUMNS,
    )


def _degree_map(pairs: set[tuple[str, str]]) -> dict[str, int]:
    degree: dict[str, int] = {}

    for rider_1, rider_2 in pairs:
        degree[rider_1] = degree.get(rider_1, 0) + 1
        degree[rider_2] = degree.get(rider_2, 0) + 1

    return degree


def build_node_attributes_frame(
    ctx: SimulationContext,
    result: SimulationResult,
    coordination_counts: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    familiarity = familiarity_pairs(result.network_state)
    social = social_tie_pairs(result.network_state)

    familiarity_degree = _degree_map(familiarity)
    social_degree = _degree_map(social)
    coordination = coordination_counts or {}

    rows: list[dict[str, object]] = []

    for rider in ctx.riders.values():
        rows.append(
            {
                "rider_id": rider.rider_id,
                "home_market": rider.home_market,
                "home_cluster": rider.home_cluster,
                "annual_ride_rate": (
                    rider.baseline_annual_ride_rate
                ),
                "total_attendance": len(
                    rider.attended_session_ids
                ),
                "unique_studios": len(
                    rider.attended_studio_counts
                ),
                "unique_instructors": len(
                    rider.attended_instructor_counts
                ),
                "familiarity_degree": (
                    familiarity_degree.get(
                        rider.rider_id,
                        0,
                    )
                ),
                "social_degree": (
                    social_degree.get(
                        rider.rider_id,
                        0,
                    )
                ),
                "coordination_count": coordination.get(
                    rider.rider_id,
                    0,
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=NODE_ATTRIBUTE_COLUMNS,
    )


def build_edge_list_frame(
    state: NetworkState,
    layer: str,
) -> pd.DataFrame:
    if layer == "familiarity":
        pairs = familiarity_pairs(state)
    elif layer == "social":
        pairs = social_tie_pairs(state)
    else:
        raise ValueError(
            "layer must be 'familiarity' or 'social'."
        )

    rows: list[dict[str, object]] = []

    for rider_1, rider_2 in pairs:
        key = tuple(
            sorted(
                (rider_1, rider_2)
            )
        )

        rows.append(
            {
                "rider_1": key[0],
                "rider_2": key[1],
                "coattendance_count": state.co_counts[key],
                "tie_strength": state.tie_strength.get(
                    key,
                    0.0,
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=EDGE_COLUMNS,
    )


def coordination_counts_from_result(
    result: SimulationResult,
) -> dict[str, int]:
    counts: dict[str, int] = {}

    for week in result.week_results:
        for record in week.booking.records:
            if not record.is_coordinated:
                continue

            counts[record.rider_id] = (
                counts.get(record.rider_id, 0) + 1
            )

    return counts


def export_seed_outputs(
    output_dir: str | Path,
    scenario: str,
    seed: int,
    ctx: SimulationContext,
    result: SimulationResult,
    tracker: PairWeekTracker,
    summary: Mapping[str, object],
    longitudinal: pd.DataFrame,
) -> Path:
    root = Path(output_dir) / scenario / ("seed_" + str(seed))
    root.mkdir(parents=True, exist_ok=True)

    summary_row = dict(summary)
    summary_row["scenario"] = scenario
    summary_row["seed"] = seed

    pd.DataFrame(
        [summary_row]
    ).to_csv(
        root / "simulation_summary.csv",
        index=False,
    )

    build_weekly_summary_frame(
        result
    ).to_csv(
        root / "weekly_summary.csv",
        index=False,
    )

    build_attendance_frame(
        ctx,
        result,
    ).to_csv(
        root / "attendance.csv",
        index=False,
    )

    build_node_attributes_frame(
        ctx,
        result,
        coordination_counts_from_result(result),
    ).to_csv(
        root / "node_attributes.csv",
        index=False,
    )

    build_pair_history_frame(
        result.network_state,
        tracker,
    ).to_csv(
        root / "pair_history.csv",
        index=False,
    )

    build_edge_list_frame(
        result.network_state,
        "familiarity",
    ).to_csv(
        root / "familiarity_edges.csv",
        index=False,
    )

    build_edge_list_frame(
        result.network_state,
        "social",
    ).to_csv(
        root / "social_edges.csv",
        index=False,
    )

    longitudinal.to_csv(
        root / "longitudinal_metrics.csv",
        index=False,
    )

    return root