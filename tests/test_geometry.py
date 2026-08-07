"""Tests for TCA / miss-distance finding, against analytically known cases."""

import numpy as np
import pytest

from conjunction_risk.constants import MU_EARTH_KM3_S2
from conjunction_risk.geometry import find_closest_approach, relative_range_km, state_fn_from_state
from conjunction_risk.state import StateVector


def linear_state_fn(r0_vec, v_vec):
    """A state_fn for straight-line constant-velocity motion (no gravity).

    Used to test the TCA-finding logic itself against a case with a closed-form
    answer, independent of the two-body propagator.
    """
    r0_vec = np.asarray(r0_vec, dtype=float)
    v_vec = np.asarray(v_vec, dtype=float)

    def state_at(t_s):
        return r0_vec + v_vec * t_s, v_vec

    return state_at


def test_linear_crossing_has_known_tca_and_miss_distance():
    # Object A: starts at origin, moving along +x at 1 km/s.
    state_fn_a = linear_state_fn([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    # Object B: same x-velocity (cancels in the relative frame), starts offset
    # 100 km in y and 5 km in z, closing in z at 10 km/s.
    # Relative position is then (0, 100, 5 - 10*t): closest approach at t=0.5s,
    # with y fixed at 100 km, so miss distance is exactly 100 km.
    state_fn_b = linear_state_fn([0.0, 100.0, 5.0], [1.0, 0.0, -10.0])

    result = find_closest_approach(state_fn_a, state_fn_b, t_start_s=0.0, t_end_s=5.0, coarse_step_s=0.1)

    assert result["tca_s"] == pytest.approx(0.5, abs=1e-3)
    assert result["miss_distance_km"] == pytest.approx(100.0, abs=1e-3)
    assert result["relative_speed_km_s"] == pytest.approx(10.0, abs=1e-6)


def test_relative_range_km_matches_direct_norm():
    state_fn_a = linear_state_fn([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    state_fn_b = linear_state_fn([3.0, 4.0, 0.0], [0.0, 0.0, 0.0])

    assert relative_range_km(state_fn_a, state_fn_b, t_s=0.0) == pytest.approx(5.0)


def test_find_closest_approach_with_two_body_states():
    # A and B share the same circular orbit (radius, speed, plane, direction),
    # with B leading A by a 90 degree phase angle. That's a rigid relative
    # geometry — the separation is the constant chord length between two
    # points 90 degrees apart on the circle, 2*r0*sin(90deg/2), for all time.
    # This is a smoke test that state_fn_from_state interoperates correctly
    # with find_closest_approach, checked against a closed-form distance.
    r0_km = 7000.0
    v_circ_km_s = np.sqrt(MU_EARTH_KM3_S2 / r0_km)

    state_a = StateVector(r_km=[r0_km, 0.0, 0.0], v_km_s=[0.0, v_circ_km_s, 0.0])
    state_b = StateVector(r_km=[0.0, r0_km, 0.0], v_km_s=[-v_circ_km_s, 0.0, 0.0])

    state_fn_a = state_fn_from_state(state_a)
    state_fn_b = state_fn_from_state(state_b)

    period_s = 2.0 * np.pi * np.sqrt(r0_km**3 / MU_EARTH_KM3_S2)
    result = find_closest_approach(state_fn_a, state_fn_b, t_start_s=0.0, t_end_s=period_s, coarse_step_s=1.0)

    expected_chord_km = 2.0 * r0_km * np.sin(np.deg2rad(90.0) / 2.0)
    assert result["miss_distance_km"] == pytest.approx(expected_chord_km, abs=1.0)
