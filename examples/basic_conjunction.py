"""Phase 1 demo: miss distance vs. time and time of closest approach (TCA).

Sets up two objects in circular orbits of the same radius and period but with
a small relative inclination and phase offset — the kind of near-coplanar,
near-resonant geometry that produces a genuine close approach at the node
crossing rather than a distant flyby. Propagates both with the two-body
propagator, samples relative range over one orbital period, and reports/plots
the TCA and miss distance.

Run with: python examples/basic_conjunction.py
"""

import matplotlib.pyplot as plt
import numpy as np

from conjunction_risk import (
    MU_EARTH_KM3_S2,
    StateVector,
    find_closest_approach,
    state_fn_from_state,
)

R0_KM = 7000.0
V_CIRC_KM_S = np.sqrt(MU_EARTH_KM3_S2 / R0_KM)
PERIOD_S = 2.0 * np.pi * np.sqrt(R0_KM**3 / MU_EARTH_KM3_S2)

# Relative geometry: both objects start shortly before the point (R0_KM, 0, 0),
# which is a node of both orbital planes (rotating about the x-axis leaves the
# x-axis fixed, so this point lies on both circles regardless of phase/incl.).
# A nonzero DELTA_I and PHASE_OFFSET separate the two objects at that node
# instead of letting them collide exactly.
DELTA_I_RAD = np.deg2rad(0.05)
PHASE_OFFSET_RAD = np.deg2rad(0.01)
THETA_P0_RAD = np.deg2rad(-0.5)
THETA_S0_RAD = THETA_P0_RAD + PHASE_OFFSET_RAD


def circular_state(theta0_rad, incl_rad):
    """State vector for a circular orbit of radius R0_KM, inclined by incl_rad
    about the x-axis, at true anomaly theta0_rad."""
    cos_t, sin_t = np.cos(theta0_rad), np.sin(theta0_rad)
    cos_i, sin_i = np.cos(incl_rad), np.sin(incl_rad)

    r_km = R0_KM * np.array([cos_t, sin_t * cos_i, sin_t * sin_i])
    v_km_s = V_CIRC_KM_S * np.array([-sin_t, cos_t * cos_i, cos_t * sin_i])
    return StateVector(r_km=r_km, v_km_s=v_km_s, epoch_s=0.0)


def main():
    primary = circular_state(THETA_P0_RAD, incl_rad=0.0)
    secondary = circular_state(THETA_S0_RAD, incl_rad=DELTA_I_RAD)

    state_fn_primary = state_fn_from_state(primary)
    state_fn_secondary = state_fn_from_state(secondary)

    result = find_closest_approach(
        state_fn_primary, state_fn_secondary, t_start_s=0.0, t_end_s=PERIOD_S, coarse_step_s=1.0
    )

    print(f"TCA:              {result['tca_s']:.2f} s after epoch")
    print(f"Miss distance:    {result['miss_distance_km']:.4f} km")
    print(f"Relative speed:   {result['relative_speed_km_s']:.4f} km/s")

    times_min = result["coarse_times_s"] / 60.0
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(times_min, result["coarse_ranges_km"], label="Relative range")
    ax.axvline(result["tca_s"] / 60.0, color="tab:red", linestyle="--", label="TCA")
    ax.scatter([result["tca_s"] / 60.0], [result["miss_distance_km"]], color="tab:red", zorder=5)
    ax.set_xlabel("Time since epoch (min)")
    ax.set_ylabel("Miss distance (km)")
    ax.set_title("Miss distance vs. time")
    ax.legend()
    fig.tight_layout()

    out_path = "examples/miss_distance_vs_time.png"
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
