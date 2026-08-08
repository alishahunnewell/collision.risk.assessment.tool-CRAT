"""Two-body (Keplerian) orbit propagation via the universal variable formulation.

This solves Kepler's problem: given a state vector (r0_vec, v0_vec) at some
epoch, find the state vector after a time step dt, for any conic (circular,
elliptical, parabolic, or hyperbolic) using a single universal anomaly chi,
following the classic Newton iteration on Stumpff functions C(z) and S(z)
(see Curtis, "Orbital Mechanics for Engineering Students", Algorithms 3.3/3.4;
Vallado, "Fundamentals of Astrodynamics and Applications", Algorithm 8).

All vectors are ECI (Earth-Centered Inertial) unless otherwise noted. Units
are km, km/s, s throughout.
"""

import numpy as np

from .constants import MU_EARTH_KM3_S2


def stumpff_c(z):
    """Stumpff function C(z), used in the universal Kepler's equation."""
    if z > 1e-6:
        sqrt_z = np.sqrt(z)
        return (1.0 - np.cos(sqrt_z)) / z
    elif z < -1e-6:
        sqrt_neg_z = np.sqrt(-z)
        return (np.cosh(sqrt_neg_z) - 1.0) / (-z)
    else:
        return 0.5 - z / 24.0 + z**2 / 720.0


def stumpff_s(z):
    """Stumpff function S(z), used in the universal Kepler's equation."""
    if z > 1e-6:
        sqrt_z = np.sqrt(z)
        return (sqrt_z - np.sin(sqrt_z)) / sqrt_z**3
    elif z < -1e-6:
        sqrt_neg_z = np.sqrt(-z)
        return (np.sinh(sqrt_neg_z) - sqrt_neg_z) / sqrt_neg_z**3
    else:
        return 1.0 / 6.0 - z / 120.0 + z**2 / 5040.0


def solve_universal_anomaly(r0_vec, v0_vec, dt_s, mu_km3_s2=MU_EARTH_KM3_S2, tol=1e-8, max_iter=100):
    """Solve the universal Kepler's equation for the universal anomaly chi.

    chi parameterizes how far the object has moved along its orbit over dt_s,
    in a form that works uniformly across circular, elliptical, parabolic, and
    hyperbolic orbits (unlike eccentric or hyperbolic anomaly individually).
    """
    r0_vec = np.asarray(r0_vec, dtype=float)
    v0_vec = np.asarray(v0_vec, dtype=float)

    r0 = np.linalg.norm(r0_vec)
    v0 = np.linalg.norm(v0_vec)
    vr0 = np.dot(r0_vec, v0_vec) / r0

    alpha = 2.0 / r0 - v0**2 / mu_km3_s2
    sqrt_mu = np.sqrt(mu_km3_s2)

    chi = sqrt_mu * abs(alpha) * dt_s

    for _ in range(max_iter):
        z = alpha * chi**2
        c = stumpff_c(z)
        s = stumpff_s(z)

        f_chi = (
            (r0 * vr0 / sqrt_mu) * chi**2 * c
            + (1.0 - alpha * r0) * chi**3 * s
            + r0 * chi
            - sqrt_mu * dt_s
        )
        f_prime_chi = (
            (r0 * vr0 / sqrt_mu) * chi * (1.0 - alpha * chi**2 * s)
            + (1.0 - alpha * r0) * chi**2 * c
            + r0
        )

        d_chi = f_chi / f_prime_chi
        chi -= d_chi

        if abs(d_chi) < tol:
            return chi

    raise RuntimeError(f"universal anomaly did not converge after {max_iter} iterations")


def propagate_two_body(r0_vec, v0_vec, dt_s, mu_km3_s2=MU_EARTH_KM3_S2):
    """Propagate a two-body state vector forward (or backward) by dt_s.

    r0_vec, v0_vec: initial position (km) and velocity (km/s) in ECI.
    dt_s: time step, seconds. May be negative.

    Returns (r_vec, v_vec) at epoch + dt_s, same frame and units as input.
    """
    r0_vec = np.asarray(r0_vec, dtype=float)
    v0_vec = np.asarray(v0_vec, dtype=float)

    if dt_s == 0.0:
        return r0_vec.copy(), v0_vec.copy()

    r0 = np.linalg.norm(r0_vec)
    v0 = np.linalg.norm(v0_vec)
    vr0 = np.dot(r0_vec, v0_vec) / r0
    alpha = 2.0 / r0 - v0**2 / mu_km3_s2
    sqrt_mu = np.sqrt(mu_km3_s2)

    chi = solve_universal_anomaly(r0_vec, v0_vec, dt_s, mu_km3_s2)

    z = alpha * chi**2
    c = stumpff_c(z)
    s = stumpff_s(z)

    f = 1.0 - (chi**2 / r0) * c
    g = dt_s - (chi**3 / sqrt_mu) * s

    r_vec = f * r0_vec + g * v0_vec
    r = np.linalg.norm(r_vec)

    f_dot = (sqrt_mu / (r * r0)) * (alpha * chi**3 * s - chi)
    g_dot = 1.0 - (chi**2 / r) * c

    v_vec = f_dot * r0_vec + g_dot * v0_vec

    return r_vec, v_vec
