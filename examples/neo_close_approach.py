"""Applying the same toolkit to a near-Earth object (NEO) close approach.

Demonstrates that the propagator and closest-approach code built for
satellite conjunction assessment work unchanged for a heliocentric
Earth/asteroid encounter: swap Earth's GM for the Sun's GM and the same
state_fn_from_state / find_closest_approach pipeline reports how close a
near-Earth object passes to Earth, and when.

Simplification: for clarity, the NEO here shares Earth's semi-major axis and
is displaced only by a small relative inclination and phase offset, the same
same-orbit node-crossing setup used for the satellite pair in
basic_conjunction.py, just scaled up to heliocentric distance. A real
cataloged NEO usually has a distinct semi-major axis and larger eccentricity
than Earth, which is what gives real encounters their higher relative
speeds; modeling that would need converting orbital elements (semi-major
axis, eccentricity, inclination, etc.) to a state vector, which this
toolkit does not implement yet.

Run with: python examples/neo_close_approach.py
"""

import matplotlib.pyplot as plt
import numpy as np

from conjunction_risk import AU_KM, MU_SUN_KM3_S2, StateVector, find_closest_approach, state_fn_from_state

V_EARTH_KM_S = np.sqrt(MU_SUN_KM3_S2 / AU_KM)
PERIOD_S = 2.0 * np.pi * np.sqrt(AU_KM**3 / MU_SUN_KM3_S2)

DELTA_I_RAD = np.deg2rad(0.05)
PHASE_OFFSET_RAD = np.deg2rad(0.01)
THETA_EARTH0_RAD = np.deg2rad(-3.0)
THETA_NEO0_RAD = THETA_EARTH0_RAD + PHASE_OFFSET_RAD

SCREEN_WINDOW_S = PERIOD_S / 20.0


def heliocentric_circular_state(theta0_rad, incl_rad):
    """State vector for a circular heliocentric orbit at 1 AU, inclined by
    incl_rad about the x-axis, at true anomaly theta0_rad."""
    cos_t, sin_t = np.cos(theta0_rad), np.sin(theta0_rad)
    cos_i, sin_i = np.cos(incl_rad), np.sin(incl_rad)

    r_km = AU_KM * np.array([cos_t, sin_t * cos_i, sin_t * sin_i])
    v_km_s = V_EARTH_KM_S * np.array([-sin_t, cos_t * cos_i, cos_t * sin_i])
    return StateVector(r_km=r_km, v_km_s=v_km_s, epoch_s=0.0)


def main():
    earth = heliocentric_circular_state(THETA_EARTH0_RAD, incl_rad=0.0)
    neo = heliocentric_circular_state(THETA_NEO0_RAD, incl_rad=DELTA_I_RAD)

    state_fn_earth = state_fn_from_state(earth, mu_km3_s2=MU_SUN_KM3_S2)
    state_fn_neo = state_fn_from_state(neo, mu_km3_s2=MU_SUN_KM3_S2)

    result = find_closest_approach(
        state_fn_earth, state_fn_neo, t_start_s=0.0, t_end_s=SCREEN_WINDOW_S, coarse_step_s=60.0
    )

    print(f"TCA:              {result['tca_s'] / 3600.0:.2f} hr after epoch")
    print(
        f"Miss distance:    {result['miss_distance_km']:,.0f} km "
        f"({result['miss_distance_km'] / AU_KM:.6f} AU)"
    )
    print(f"Relative speed:   {result['relative_speed_km_s']:.4f} km/s")

    times_hr = result["coarse_times_s"] / 3600.0
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(times_hr, result["coarse_ranges_km"], label="Earth-NEO range")
    ax.axvline(result["tca_s"] / 3600.0, color="tab:red", linestyle="--", label="TCA")
    ax.scatter([result["tca_s"] / 3600.0], [result["miss_distance_km"]], color="tab:red", zorder=5)
    ax.set_xlabel("Time since epoch (hr)")
    ax.set_ylabel("Miss distance (km)")
    ax.set_title("NEO close approach: miss distance vs. time")
    ax.legend()
    fig.tight_layout()

    out_path = "examples/neo_close_approach.png"
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
