"""Phase 2 demo: screen a small catalog of secondary objects against a primary.

Sets up a primary in a circular orbit and a handful of secondaries at varying
relative inclination and phase offset, some passing close to the primary at
their shared node, others clearly safe. Runs screen_conjunctions to rank the
whole catalog by miss distance, then filter_by_threshold to flag which ones
are worth a closer look.

Run with: python examples/screening_demo.py
"""

import numpy as np

from conjunction_risk import MU_EARTH_KM3_S2, StateVector, filter_by_threshold, screen_conjunctions

R0_KM = 7000.0
V_CIRC_KM_S = np.sqrt(MU_EARTH_KM3_S2 / R0_KM)
PERIOD_S = 2.0 * np.pi * np.sqrt(R0_KM**3 / MU_EARTH_KM3_S2)

FLAG_THRESHOLD_KM = 10.0


def circular_state(theta_deg, incl_deg=0.0):
    """Circular orbit state at radius R0_KM, true anomaly theta_deg, inclined
    by incl_deg about the x-axis relative to the primary's equatorial plane."""
    theta_rad = np.deg2rad(theta_deg)
    incl_rad = np.deg2rad(incl_deg)
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    cos_i, sin_i = np.cos(incl_rad), np.sin(incl_rad)

    r_km = R0_KM * np.array([cos_t, sin_t * cos_i, sin_t * sin_i])
    v_km_s = V_CIRC_KM_S * np.array([-sin_t, cos_t * cos_i, cos_t * sin_i])
    return StateVector(r_km=r_km, v_km_s=v_km_s)


def main():
    primary = circular_state(theta_deg=-0.5)

    catalog = [
        ("DEBRIS-1", circular_state(theta_deg=-0.49, incl_deg=0.05)),
        ("DEBRIS-2", circular_state(theta_deg=5.0, incl_deg=0.5)),
        ("SAT-ALPHA", circular_state(theta_deg=90.0, incl_deg=0.0)),
        ("SAT-BETA", circular_state(theta_deg=-30.0, incl_deg=2.0)),
    ]

    results = screen_conjunctions(primary, catalog, t_start_s=0.0, t_end_s=PERIOD_S, coarse_step_s=1.0)
    flagged = filter_by_threshold(results, threshold_km=FLAG_THRESHOLD_KM)
    flagged_names = {result.name for result in flagged}

    print(f"{'name':<12}{'TCA (min)':>12}{'miss dist (km)':>18}{'rel speed (km/s)':>20}{'flagged':>10}")
    for result in results:
        is_flagged = result.name in flagged_names
        print(
            f"{result.name:<12}"
            f"{result.tca_s / 60.0:>12.2f}"
            f"{result.miss_distance_km:>18.4f}"
            f"{result.relative_speed_km_s:>20.4f}"
            f"{'yes' if is_flagged else '':>10}"
        )

    print(f"\n{len(flagged)} of {len(results)} objects pass within {FLAG_THRESHOLD_KM} km.")


if __name__ == "__main__":
    main()
