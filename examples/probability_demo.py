"""Phase 3 demo: probability of collision (Pc) for a genuine close approach.

Reuses the near-node-crossing satellite scenario from basic_conjunction.py
(a real close approach, not a distant flyby) and attaches illustrative
position covariances and a hard-body radius (HBR) to compute Pc at TCA,
then compares it against a commonly cited operational maneuver threshold.

The covariances here are simplified: real Conjunction Data Messages (CDMs)
give anisotropic covariance in the RTN (radial/transverse/normal) frame,
with much larger along-track (transverse) uncertainty than radial or
cross-track, reflecting how along-track position error grows fastest over a
propagation window. This demo uses isotropic covariance directly in ECI for
simplicity; converting a realistic RTN covariance to ECI would need an
RTN-to-ECI rotation this toolkit does not implement yet.

Run with: python examples/probability_demo.py
"""

import numpy as np

from conjunction_risk import (
    MU_EARTH_KM3_S2,
    StateVector,
    find_closest_approach,
    probability_of_collision_at_tca,
    state_fn_from_state,
)

R0_KM = 7000.0
V_CIRC_KM_S = np.sqrt(MU_EARTH_KM3_S2 / R0_KM)
PERIOD_S = 2.0 * np.pi * np.sqrt(R0_KM**3 / MU_EARTH_KM3_S2)

# A commonly cited (not universal) operational Pc threshold for considering
# a collision-avoidance maneuver in LEO.
MANEUVER_THRESHOLD_PC = 1.0e-4

# Typical small-satellite physical radius, for a combined hard-body radius.
SAT_RADIUS_KM = 0.005


def circular_state(theta_deg, incl_deg, sigma_km):
    """Circular orbit state at radius R0_KM, with an isotropic position
    covariance of standard deviation sigma_km in each axis (see module
    docstring for the simplification this represents)."""
    theta_rad = np.deg2rad(theta_deg)
    incl_rad = np.deg2rad(incl_deg)
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    cos_i, sin_i = np.cos(incl_rad), np.sin(incl_rad)

    r_km = R0_KM * np.array([cos_t, sin_t * cos_i, sin_t * sin_i])
    v_km_s = V_CIRC_KM_S * np.array([-sin_t, cos_t * cos_i, cos_t * sin_i])
    cov_km2 = np.eye(3) * sigma_km**2
    return StateVector(r_km=r_km, v_km_s=v_km_s, cov_km2=cov_km2)


def main():
    primary = circular_state(theta_deg=-0.5, incl_deg=0.0, sigma_km=0.1)
    secondary = circular_state(theta_deg=-0.49, incl_deg=0.05, sigma_km=0.3)

    closest_approach = find_closest_approach(
        state_fn_from_state(primary), state_fn_from_state(secondary), t_start_s=0.0, t_end_s=PERIOD_S
    )

    hbr_km = 2.0 * SAT_RADIUS_KM
    pc = probability_of_collision_at_tca(closest_approach, primary.cov_km2, secondary.cov_km2, hbr_km)

    print(f"TCA:              {closest_approach['tca_s']:.2f} s after epoch")
    print(f"Miss distance:    {closest_approach['miss_distance_km']:.4f} km")
    print(f"Hard-body radius: {hbr_km * 1000.0:.1f} m")
    print(f"Probability of collision (Pc): {pc:.3e}")

    if pc >= MANEUVER_THRESHOLD_PC:
        print(f"Pc exceeds the {MANEUVER_THRESHOLD_PC:.0e} maneuver threshold: this would warrant a closer look.")
    else:
        print(f"Pc is below the {MANEUVER_THRESHOLD_PC:.0e} maneuver threshold.")


if __name__ == "__main__":
    main()
