"""Tests for probability of collision (Pc), against analytically known cases."""

import numpy as np
import pytest

from conjunction_risk.constants import MU_EARTH_KM3_S2
from conjunction_risk.geometry import find_closest_approach
from conjunction_risk.probability import (
    encounter_plane_basis,
    probability_of_collision,
    probability_of_collision_at_tca,
    project_covariance_to_encounter_plane,
)
from conjunction_risk.state import StateVector


def test_encounter_plane_basis_is_orthonormal_and_right_handed():
    r_rel_km = np.array([3.0, 4.0, 0.0])
    v_rel_km_s = np.array([0.0, 0.0, 2.0])

    x_hat, y_hat, z_hat = encounter_plane_basis(r_rel_km, v_rel_km_s)

    assert np.linalg.norm(x_hat) == pytest.approx(1.0)
    assert np.linalg.norm(y_hat) == pytest.approx(1.0)
    assert np.linalg.norm(z_hat) == pytest.approx(1.0)
    assert np.dot(x_hat, z_hat) == pytest.approx(0.0, abs=1e-12)
    assert np.dot(x_hat, y_hat) == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(np.cross(x_hat, y_hat), z_hat)


def test_isotropic_covariance_centered_at_zero_matches_rayleigh_cdf():
    # For an isotropic (circularly symmetric) 2D Gaussian, the radial distance
    # from the mean follows a Rayleigh distribution with scale sigma, so the
    # probability mass within radius hbr_km of the mean has a closed form:
    # P(R <= hbr_km) = 1 - exp(-hbr_km**2 / (2 * sigma**2)). An isotropic 3D
    # covariance projects to an isotropic 2D covariance in any orthonormal
    # plane, so this closed form applies regardless of encounter geometry.
    sigma_a_km = 0.5
    sigma_b_km = 0.3
    sigma_total_km = np.sqrt(sigma_a_km**2 + sigma_b_km**2)
    cov_a_km2 = np.eye(3) * sigma_a_km**2
    cov_b_km2 = np.eye(3) * sigma_b_km**2

    # A tiny but nonzero miss vector, needed only to define a direction for
    # the encounter-plane basis; negligible compared to sigma_total_km.
    r_rel_km = np.array([1e-6, 0.0, 0.0])
    v_rel_km_s = np.array([0.0, 1.0, 0.0])
    hbr_km = 1.0

    pc = probability_of_collision(r_rel_km, v_rel_km_s, cov_a_km2, cov_b_km2, hbr_km)
    expected_pc = 1.0 - np.exp(-(hbr_km**2) / (2.0 * sigma_total_km**2))

    assert pc == pytest.approx(expected_pc, abs=1e-4)


def test_pc_approaches_one_for_hbr_much_larger_than_covariance():
    # hbr_km = 50 * sigma comfortably captures effectively all of the
    # Gaussian's mass without pushing the numerical integrator into the
    # extreme-domain-vs-needle-thin-peak regime that a much larger ratio
    # (e.g. 1000 * sigma) would.
    sigma_km = 0.1
    cov_a_km2 = np.eye(3) * sigma_km**2
    cov_b_km2 = np.eye(3) * sigma_km**2
    r_rel_km = np.array([0.05, 0.0, 0.0])
    v_rel_km_s = np.array([0.0, 1.0, 0.0])

    pc = probability_of_collision(r_rel_km, v_rel_km_s, cov_a_km2, cov_b_km2, hbr_km=50.0 * sigma_km)

    assert pc == pytest.approx(1.0, abs=1e-6)


def test_pc_approaches_zero_for_distant_miss_and_small_hbr():
    cov_a_km2 = np.eye(3) * 0.05**2
    cov_b_km2 = np.eye(3) * 0.05**2
    r_rel_km = np.array([50.0, 0.0, 0.0])
    v_rel_km_s = np.array([0.0, 1.0, 0.0])

    pc = probability_of_collision(r_rel_km, v_rel_km_s, cov_a_km2, cov_b_km2, hbr_km=0.01)

    assert pc == pytest.approx(0.0, abs=1e-9)


def test_probability_of_collision_at_tca_integrates_with_find_closest_approach():
    # Same near-node-crossing circular-orbit setup used in test_geometry.py /
    # examples/basic_conjunction.py: a genuine close approach with a known
    # small but nonzero miss distance.
    r0_km = 7000.0
    v_circ_km_s = np.sqrt(MU_EARTH_KM3_S2 / r0_km)
    period_s = 2.0 * np.pi * np.sqrt(r0_km**3 / MU_EARTH_KM3_S2)

    def circular_state(theta_rad, incl_rad):
        cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
        cos_i, sin_i = np.cos(incl_rad), np.sin(incl_rad)
        r_km = r0_km * np.array([cos_t, sin_t * cos_i, sin_t * sin_i])
        v_km_s = v_circ_km_s * np.array([-sin_t, cos_t * cos_i, cos_t * sin_i])
        return StateVector(r_km=r_km, v_km_s=v_km_s)

    from conjunction_risk.geometry import state_fn_from_state

    primary = circular_state(np.deg2rad(-0.5), incl_rad=0.0)
    secondary = circular_state(np.deg2rad(-0.49), incl_rad=np.deg2rad(0.05))

    closest_approach = find_closest_approach(
        state_fn_from_state(primary), state_fn_from_state(secondary), t_start_s=0.0, t_end_s=period_s
    )

    cov_km2 = np.eye(3) * 0.5**2
    pc = probability_of_collision_at_tca(closest_approach, cov_km2, cov_km2, hbr_km=0.01)

    assert 0.0 <= pc <= 1.0


def test_project_covariance_to_encounter_plane_matches_manual_projection():
    cov_km2 = np.diag([4.0, 9.0, 16.0])
    x_hat = np.array([1.0, 0.0, 0.0])
    y_hat = np.array([0.0, 1.0, 0.0])

    cov_2d = project_covariance_to_encounter_plane(cov_km2, x_hat, y_hat)

    assert np.allclose(cov_2d, np.diag([4.0, 9.0]))
