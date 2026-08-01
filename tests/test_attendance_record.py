import pytest
from soulcycle_network.attendance_record import AttendanceRecord, session_key
from soulcycle_network.network_formation import NetworkState, decay_ties, familiarity_pairs, pair_key, social_tie_pairs, update_from_enrollments
from soulcycle_network.config import MIN_CLASSES_FOR_FAMILIARITY, MIN_CLASSES_FOR_SOCIAL_TIE, TIE_DECAY_RATE

def test_session_key_format():
    assert session_key(1, "GTWN_MON_A_01") == "W01_GTWN_MON_A_01"

def test_attendance_record_validation():
    record = AttendanceRecord(
        week_number=1,
        slot_id="GTWN_MON_A_01",
        rider_id="R000001",
        studio_id="GTWN",
        assigned_instructor_id="I0001",
        day_of_week="Monday",
        daily_slot_index=1,
    )
    assert record.is_coordinated is False

def test_pair_key_sorts_ids():
    assert pair_key("R000002", "R000001") == ("R000001", "R000002")

def test_update_from_enrollments_counts_pairs():
    state = NetworkState()
    update_from_enrollments(state, {"slot_a": ["R000001", "R000002", "R000003"]})
    assert state.co_counts[("R000001", "R000002")] == 1
    assert state.co_counts[("R000001", "R000003")] == 1
    assert state.co_counts[("R000002", "R000003")] == 1

def test_familiarity_and_social_thresholds():
    state = NetworkState()
    key = ("R000001", "R000002")
    state.co_counts[key] = MIN_CLASSES_FOR_FAMILIARITY
    assert key in familiarity_pairs(state)
    assert key not in social_tie_pairs(state)

    state.co_counts[key] = MIN_CLASSES_FOR_SOCIAL_TIE
    assert key in social_tie_pairs(state)

def test_decay_ties():
    state = NetworkState()
    key = ("R000001", "R000002")
    state.tie_strength[key] = 10.0
    decay_ties(state, TIE_DECAY_RATE)
    assert abs(state.tie_strength[key] - 10.0 * TIE_DECAY_RATE) < 1e-9

def test_pair_key_rejects_same_rider():
    with pytest.raises(ValueError):
        pair_key("R000001", "R000001")
