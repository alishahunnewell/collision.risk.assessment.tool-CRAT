"""Tests for maneuver recommendation."""

import numpy as np
import pytest

from conjunction_risk.constants import MU_EARTH_KM3_S2
from conjunction_risk.maneuver import (
    along_track_unit_vector,
    apply_delta_v,
    evaluate_conjunction,
    recommend_along_track_delta_v,
)
from conjunction_risk.state import StateVector

R0_KM = 7000.0
V_CIRC_KM_S = np.sqrt(MU_EARTH_KM3_S2 / R0_KM)
PERIOD_S = 2.0 * np.pi * np.sqrt(R0_KM**3 / MU_EARTH_KM3_S2)


def circular_state(theta_deg, incl_deg):
    """Circular orbit state at radius R0_KM, true anomaly theta_deg, inclined
    by incl_deg about the x-axis."""
    theta_rad = np.deg2rad(theta_deg)
    incl_rad = np.deg2rad(incl_deg)
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    cos_i, sin_i = np.cos(incl_rad), np.sin(incl_rad)
    r_km = R0_KM * np.array([cos_t, sin_t * cos_i, sin_t * sin_i])
    v_km_s = V_CIRC_KM_S * np.array([-sin_t, cos_t * cos_i, cos_t * sin_i])
    return StateVector(r_km=r_km, v_km_s=v_km_s)


def breaching_scenario():
    # A 2 degree relative inclination gives a realistic, non-degenerate
    # relative velocity at the node (a few hundred m/s), unlike a tiny
    # fraction-of-a-degree offset, which makes the encounter close to
    # resonant/degenerate and along-track delta-v poorly behaved.
    primary = circular_state(-0.1, 0.0)
    secondary = circular_state(-0.1 + 0.0002, 2.0)
    cov_km2 = np.eye(3) * 0.1**2
    return primary, secondary, cov_km2


def test_apply_delta_v_only_changes_velocity():
    state = StateVector(r_km=[7000.0, 0.0, 0.0], v_km_s=[0.0, 7.5, 0.0], epoch_s=12.0)
    delta_v_km_s = np.array([0.0, 0.001, 0.0])

    maneuvered = apply_delta_v(state, delta_v_km_s)

    assert np.allclose(maneuvered.r_km, state.r_km)
    assert np.allclose(maneuvered.v_km_s, state.v_km_s + delta_v_km_s)
    assert maneuvered.epoch_s == state.epoch_s


def test_along_track_unit_vector_is_unit_length_and_parallel_to_velocity():
    v_km_s = np.array([1.0, 2.0, 2.0])
    unit = along_track_unit_vector(v_km_s)

    assert np.linalg.norm(unit) == pytest.approx(1.0)
    assert np.allclose(unit, v_km_s / np.linalg.norm(v_km_s))


def test_evaluate_conjunction_confirms_pc_breach():
    primary, secondary, cov_km2 = breaching_scenario()

    result = evaluate_conjunction(
        primary, secondary, t_start_s=0.0, t_end_s=PERIOD_S, coarse_step_s=1.0,
        cov_a_km2=cov_km2, cov_b_km2=cov_km2, hbr_km=0.01,
    )

    assert result["pc"] > 1.0e-4


def test_recommend_along_track_delta_v_clears_threshold_and_increases_miss_distance():
    primary, secondary, cov_km2 = breaching_scenario()
    target_pc = 1.0e-4

    result = recommend_along_track_delta_v(
        primary, secondary, target_pc,
        t_start_s=0.0, t_end_s=PERIOD_S, coarse_step_s=1.0,
        cov_a_km2=cov_km2, cov_b_km2=cov_km2, hbr_km=0.01,
        max_delta_v_km_s=0.001,
    )

    # The maneuver is found as the root of Pc(delta_v) - target_pc, so the
    # result lands at (not necessarily strictly below) target_pc, subject to
    # root-finder tolerance.
    assert result["after"]["pc"] == pytest.approx(target_pc, rel=1.0e-2)
    assert result["after"]["miss_distance_km"] > result["before"]["miss_distance_km"]
    assert result["before"]["pc"] > target_pc

    delta_v_mag_km_s = np.linalg.norm(result["delta_v_km_s"])
    assert 0.0 < delta_v_mag_km_s <= 0.001


def test_recommend_along_track_delta_v_raises_if_already_below_target():
    primary, secondary, cov_km2 = breaching_scenario()

    with pytest.raises(ValueError):
        recommend_along_track_delta_v(
            primary, secondary, target_pc=1.0e-2,
            t_start_s=0.0, t_end_s=PERIOD_S, coarse_step_s=1.0,
            cov_a_km2=cov_km2, cov_b_km2=cov_km2, hbr_km=0.01,
            max_delta_v_km_s=0.001,
        )


def test_recommend_along_track_delta_v_raises_if_max_delta_v_too_small():
    primary, secondary, cov_km2 = breaching_scenario()

    with pytest.raises(ValueError):
        recommend_along_track_delta_v(
            primary, secondary, target_pc=1.0e-4,
            t_start_s=0.0, t_end_s=PERIOD_S, coarse_step_s=1.0,
            cov_a_km2=cov_km2, cov_b_km2=cov_km2, hbr_km=0.01,
            max_delta_v_km_s=1.0e-6,
        )
