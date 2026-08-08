"""Tests for multi-object conjunction screening, against analytically known chord distances."""

import numpy as np
import pytest

from conjunction_risk.constants import MU_EARTH_KM3_S2
from conjunction_risk.screening import filter_by_threshold, screen_conjunctions
from conjunction_risk.state import StateVector

R0_KM = 7000.0
V_CIRC_KM_S = np.sqrt(MU_EARTH_KM3_S2 / R0_KM)
PERIOD_S = 2.0 * np.pi * np.sqrt(R0_KM**3 / MU_EARTH_KM3_S2)


def circular_state(theta_deg):
    """Equatorial circular orbit state at true anomaly theta_deg, radius R0_KM.

    Two objects on this same circle differ only by a fixed phase angle and so
    keep a constant separation for all time, the chord between them, which
    gives a closed-form answer to check screening results against.
    """
    theta_rad = np.deg2rad(theta_deg)
    r_km = R0_KM * np.array([np.cos(theta_rad), np.sin(theta_rad), 0.0])
    v_km_s = V_CIRC_KM_S * np.array([-np.sin(theta_rad), np.cos(theta_rad), 0.0])
    return StateVector(r_km=r_km, v_km_s=v_km_s)


def chord_km(dtheta_deg):
    """Separation between two points dtheta_deg apart on the same circle of radius R0_KM."""
    return 2.0 * R0_KM * np.sin(np.deg2rad(dtheta_deg) / 2.0)


def test_screen_conjunctions_sorts_by_miss_distance():
    primary = circular_state(0.0)
    secondaries = [
        ("far", circular_state(90.0)),
        ("near", circular_state(1.0)),
        ("mid", circular_state(10.0)),
    ]

    results = screen_conjunctions(primary, secondaries, t_start_s=0.0, t_end_s=PERIOD_S, coarse_step_s=5.0)

    assert [result.name for result in results] == ["near", "mid", "far"]
    assert results[0].miss_distance_km == pytest.approx(chord_km(1.0), abs=1.0)
    assert results[1].miss_distance_km == pytest.approx(chord_km(10.0), abs=1.0)
    assert results[2].miss_distance_km == pytest.approx(chord_km(90.0), abs=1.0)


def test_screen_conjunctions_reports_relative_speed_and_tca():
    primary = circular_state(0.0)
    secondaries = [("companion", circular_state(30.0))]

    results = screen_conjunctions(primary, secondaries, t_start_s=0.0, t_end_s=PERIOD_S, coarse_step_s=5.0)

    assert len(results) == 1
    assert results[0].relative_speed_km_s > 0.0
    assert 0.0 <= results[0].tca_s <= PERIOD_S


def test_filter_by_threshold_keeps_only_close_approaches():
    primary = circular_state(0.0)
    secondaries = [
        ("far", circular_state(90.0)),
        ("near", circular_state(1.0)),
    ]

    results = screen_conjunctions(primary, secondaries, t_start_s=0.0, t_end_s=PERIOD_S, coarse_step_s=5.0)
    flagged = filter_by_threshold(results, threshold_km=200.0)

    assert [result.name for result in flagged] == ["near"]
