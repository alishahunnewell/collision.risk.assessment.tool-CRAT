"""Tests for the two-body universal-variable propagator against known analytic cases."""

import numpy as np
import pytest

from conjunction_risk.constants import MU_EARTH_KM3_S2
from conjunction_risk.propagation import propagate_two_body


def circular_orbit_state(r0_km):
    """A circular equatorial orbit in the xy-plane at radius r0_km."""
    v_circ_km_s = np.sqrt(MU_EARTH_KM3_S2 / r0_km)
    r0_vec = np.array([r0_km, 0.0, 0.0])
    v0_vec = np.array([0.0, v_circ_km_s, 0.0])
    return r0_vec, v0_vec


def orbital_period_s(r0_km):
    return 2.0 * np.pi * np.sqrt(r0_km**3 / MU_EARTH_KM3_S2)


def test_zero_dt_returns_initial_state():
    r0_vec, v0_vec = circular_orbit_state(7000.0)
    r_vec, v_vec = propagate_two_body(r0_vec, v0_vec, 0.0)
    assert np.allclose(r_vec, r0_vec)
    assert np.allclose(v_vec, v0_vec)


def test_circular_orbit_returns_to_start_after_one_period():
    r0_vec, v0_vec = circular_orbit_state(7000.0)
    period_s = orbital_period_s(7000.0)

    r_vec, v_vec = propagate_two_body(r0_vec, v0_vec, period_s)

    assert np.allclose(r_vec, r0_vec, atol=1e-4)
    assert np.allclose(v_vec, v0_vec, atol=1e-8)


def test_circular_orbit_quarter_period_rotates_90_degrees():
    r0_km = 7000.0
    r0_vec, v0_vec = circular_orbit_state(r0_km)
    period_s = orbital_period_s(r0_km)

    r_vec, v_vec = propagate_two_body(r0_vec, v0_vec, period_s / 4.0)

    expected_r_vec = np.array([0.0, r0_km, 0.0])
    assert np.allclose(r_vec, expected_r_vec, atol=1e-4)
    assert np.isclose(np.linalg.norm(r_vec), r0_km, atol=1e-6)


def test_circular_orbit_half_period_is_opposite_side():
    r0_km = 7000.0
    r0_vec, v0_vec = circular_orbit_state(r0_km)
    period_s = orbital_period_s(r0_km)

    r_vec, v_vec = propagate_two_body(r0_vec, v0_vec, period_s / 2.0)

    assert np.allclose(r_vec, -r0_vec, atol=1e-4)
    assert np.allclose(v_vec, -v0_vec, atol=1e-8)


def test_propagation_conserves_specific_orbital_energy():
    r0_vec, v0_vec = circular_orbit_state(7000.0)
    r0 = np.linalg.norm(r0_vec)
    v0 = np.linalg.norm(v0_vec)
    energy0 = v0**2 / 2.0 - MU_EARTH_KM3_S2 / r0

    r_vec, v_vec = propagate_two_body(r0_vec, v0_vec, 1234.5)
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    energy = v**2 / 2.0 - MU_EARTH_KM3_S2 / r

    assert energy == pytest.approx(energy0, abs=1e-9)


def test_forward_then_backward_returns_to_start():
    r0_vec, v0_vec = circular_orbit_state(7000.0)
    dt_s = 500.0

    r_mid_vec, v_mid_vec = propagate_two_body(r0_vec, v0_vec, dt_s)
    r_back_vec, v_back_vec = propagate_two_body(r_mid_vec, v_mid_vec, -dt_s)

    assert np.allclose(r_back_vec, r0_vec, atol=1e-6)
    assert np.allclose(v_back_vec, v0_vec, atol=1e-9)
