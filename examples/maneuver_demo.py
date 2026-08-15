"""Phase 4 demo: recommend a small delta-v to clear a Pc threshold breach.

Sets up a primary/secondary pair with a genuine close approach (a real Pc
breach, not a distant flyby), then finds the smallest along-track delta-v
that brings Pc back under a threshold, and reports before/after risk.

The secondary here has a real (2 degree) relative inclination rather than a
tiny fraction-of-a-degree offset like the Phase 1-3 demos: with two objects
sharing almost the same orbit, an along-track burn barely changes their
alignment at the shared node (the relative velocity there is nearly all
out-of-plane, not along-track), so it does not reliably clear a real Pc
breach. A larger, more realistic relative inclination gives a genuine
along-track component to close on, matching how along-track avoidance
maneuvers behave for real, non-resonant conjunctions.

Run with: python examples/maneuver_demo.py
"""

import numpy as np

from conjunction_risk import MU_EARTH_KM3_S2, StateVector, recommend_along_track_delta_v

R0_KM = 7000.0
V_CIRC_KM_S = np.sqrt(MU_EARTH_KM3_S2 / R0_KM)
PERIOD_S = 2.0 * np.pi * np.sqrt(R0_KM**3 / MU_EARTH_KM3_S2)

TARGET_PC = 1.0e-4
MAX_DELTA_V_KM_S = 0.001  # 1 m/s, a generous upper bound for a "small" CA burn
SAT_RADIUS_KM = 0.005


def circular_state(theta_deg, incl_deg, sigma_km):
    theta_rad = np.deg2rad(theta_deg)
    incl_rad = np.deg2rad(incl_deg)
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    cos_i, sin_i = np.cos(incl_rad), np.sin(incl_rad)

    r_km = R0_KM * np.array([cos_t, sin_t * cos_i, sin_t * sin_i])
    v_km_s = V_CIRC_KM_S * np.array([-sin_t, cos_t * cos_i, cos_t * sin_i])
    cov_km2 = np.eye(3) * sigma_km**2
    return StateVector(r_km=r_km, v_km_s=v_km_s, cov_km2=cov_km2)


def main():
    primary = circular_state(theta_deg=-0.1, incl_deg=0.0, sigma_km=0.1)
    secondary = circular_state(theta_deg=-0.0998, incl_deg=2.0, sigma_km=0.1)

    hbr_km = 2.0 * SAT_RADIUS_KM
    result = recommend_along_track_delta_v(
        primary,
        secondary,
        target_pc=TARGET_PC,
        t_start_s=0.0,
        t_end_s=PERIOD_S,
        coarse_step_s=1.0,
        cov_a_km2=primary.cov_km2,
        cov_b_km2=secondary.cov_km2,
        hbr_km=hbr_km,
        max_delta_v_km_s=MAX_DELTA_V_KM_S,
    )

    before, after = result["before"], result["after"]
    delta_v_mag_mm_s = np.linalg.norm(result["delta_v_km_s"]) * 1.0e6

    print(f"Before maneuver (TCA = {before['tca_s']:.2f} s):")
    print(f"  Miss distance: {before['miss_distance_km'] * 1000.0:.2f} m")
    print(f"  Pc:            {before['pc']:.3e}")
    print()
    print(f"Recommended along-track delta-v: {delta_v_mag_mm_s:.2f} mm/s")
    print()
    print("After maneuver (evaluated at the original TCA):")
    print(f"  Miss distance: {after['miss_distance_km'] * 1000.0:.2f} m")
    print(f"  Pc:            {after['pc']:.3e}")
    print()
    print(f"Target Pc was {TARGET_PC:.0e}.")


if __name__ == "__main__":
    main()
