"""Probability of collision (Pc) via the standard 2D encounter-plane method.

At the time of closest approach (TCA), the relative position and velocity
vectors are perpendicular (that's the defining condition of a closest
approach: d|r_rel|/dt = 0 there). So the relative position vector already
lies in the plane perpendicular to the relative velocity, called the
encounter plane or B-plane. Projecting each object's position covariance
onto this plane and summing them gives a 2D Gaussian describing the combined
position uncertainty at TCA; Pc is the probability mass of that Gaussian
that falls within a disk of radius equal to the combined hard-body radius
(HBR, the sum of the two objects' physical radii), centered on the
(projected) miss vector.

This follows the standard formulation used in satellite conjunction
assessment (Foster 1992; Akella & Alfriend 1998) and, via the shared
B-plane/Opik heritage noted in CLAUDE.md, the same B-plane targeting logic
used for close-approach analysis of near-Earth objects (NEOs).

Covariances are 3x3 position covariance matrices, km^2, expressed in the
same frame as the corresponding position vector (ECI by default). This
toolkit does not propagate covariance; see the note on StateVector.cov_km2
in conjunction_risk/state.py.
"""

import numpy as np
from scipy.integrate import dblquad


def encounter_plane_basis(r_rel_km, v_rel_km_s):
    """Right-handed basis (x_hat, y_hat, z_hat) for the encounter plane at TCA.

    x_hat points along the relative position vector (the miss vector).
    z_hat points along the relative velocity vector (normal to the encounter
    plane). y_hat completes the right-handed set, in-plane.
    """
    r_rel_km = np.asarray(r_rel_km, dtype=float)
    v_rel_km_s = np.asarray(v_rel_km_s, dtype=float)

    x_hat = r_rel_km / np.linalg.norm(r_rel_km)
    z_hat = v_rel_km_s / np.linalg.norm(v_rel_km_s)
    y_hat = np.cross(z_hat, x_hat)

    return x_hat, y_hat, z_hat


def project_covariance_to_encounter_plane(cov_km2, x_hat, y_hat):
    """Project a 3x3 position covariance (km^2) onto the 2D encounter plane.

    Returns the 2x2 covariance matrix in (x_hat, y_hat) encounter-plane
    coordinates.
    """
    cov_km2 = np.asarray(cov_km2, dtype=float)
    projection = np.vstack([x_hat, y_hat])
    return projection @ cov_km2 @ projection.T


def probability_of_collision(r_rel_km, v_rel_km_s, cov_a_km2, cov_b_km2, hbr_km):
    """Probability of collision (Pc) via the 2D encounter-plane method.

    r_rel_km, v_rel_km_s: relative position (km) and velocity (km/s) at TCA,
    object A minus object B, in the same frame as cov_a_km2/cov_b_km2.
    cov_a_km2, cov_b_km2: 3x3 position covariance matrices (km^2) of the
    primary and secondary, treated as valid at TCA (see module docstring).
    hbr_km: combined hard-body radius, the sum of the two objects' physical
    radii, km.

    Returns Pc: the probability that the combined relative position at TCA,
    modeled as a 2D Gaussian in the encounter plane, falls within a disk of
    radius hbr_km centered on the miss vector.
    """
    x_hat, y_hat, _ = encounter_plane_basis(r_rel_km, v_rel_km_s)

    cov_2d = project_covariance_to_encounter_plane(
        cov_a_km2, x_hat, y_hat
    ) + project_covariance_to_encounter_plane(cov_b_km2, x_hat, y_hat)

    # The miss vector lies along x_hat by construction, so in encounter-plane
    # coordinates it is (miss_distance_km, 0).
    miss_distance_km = np.linalg.norm(r_rel_km)
    mean = np.array([miss_distance_km, 0.0])

    inv_cov = np.linalg.inv(cov_2d)
    det_cov = np.linalg.det(cov_2d)
    norm_const = 1.0 / (2.0 * np.pi * np.sqrt(det_cov))

    def gaussian_pdf(y, x):
        offset = np.array([x, y]) - mean
        exponent = -0.5 * offset @ inv_cov @ offset
        return norm_const * np.exp(exponent)

    def y_lower(x):
        return -np.sqrt(hbr_km**2 - x**2)

    def y_upper(x):
        return np.sqrt(hbr_km**2 - x**2)

    pc, _ = dblquad(gaussian_pdf, -hbr_km, hbr_km, y_lower, y_upper)
    return float(pc)


def probability_of_collision_at_tca(closest_approach, cov_a_km2, cov_b_km2, hbr_km):
    """Convenience wrapper: compute Pc from a find_closest_approach() result.

    closest_approach: the dict returned by geometry.find_closest_approach,
    which supplies r_rel_km and v_rel_km_s at TCA.
    cov_a_km2, cov_b_km2, hbr_km: see probability_of_collision.
    """
    return probability_of_collision(
        closest_approach["r_rel_km"],
        closest_approach["v_rel_km_s"],
        cov_a_km2,
        cov_b_km2,
        hbr_km,
    )
