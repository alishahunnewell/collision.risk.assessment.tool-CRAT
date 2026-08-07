"""Close-approach geometry: miss distance vs. time and time of closest approach (TCA).

This module works against a `state_fn(t_s) -> (r_vec, v_vec)` interface rather
than against StateVector/propagate_two_body directly, so the same TCA-finding
logic can later be reused with SGP4-propagated states (Phase 2) without
changes here.
"""

import numpy as np
from scipy.optimize import minimize_scalar

from .constants import MU_EARTH_KM3_S2
from .propagation import propagate_two_body
from .state import StateVector


def state_fn_from_state(state0: StateVector, mu_km3_s2=MU_EARTH_KM3_S2):
    """Build a state_fn(t_s) -> (r_vec, v_vec) closure around a two-body StateVector."""

    def state_at(t_s):
        dt_s = t_s - state0.epoch_s
        return propagate_two_body(state0.r_km, state0.v_km_s, dt_s, mu_km3_s2)

    return state_at


def relative_range_km(state_fn_a, state_fn_b, t_s):
    """Distance between two objects at time t_s, km."""
    r_a, _ = state_fn_a(t_s)
    r_b, _ = state_fn_b(t_s)
    return float(np.linalg.norm(r_a - r_b))


def sample_range_history(state_fn_a, state_fn_b, t_start_s, t_end_s, step_s):
    """Sample relative range over [t_start_s, t_end_s] at a fixed step.

    Returns (times_s, ranges_km), both numpy arrays. Useful for plotting
    miss distance vs. time and for coarse-locating the TCA before refinement.
    """
    times_s = np.arange(t_start_s, t_end_s + step_s, step_s)
    ranges_km = np.array([relative_range_km(state_fn_a, state_fn_b, t_s) for t_s in times_s])
    return times_s, ranges_km


def find_closest_approach(state_fn_a, state_fn_b, t_start_s, t_end_s, coarse_step_s=10.0):
    """Find the time of closest approach between two objects over a window.

    Coarse-samples the range over [t_start_s, t_end_s] to bracket the minimum,
    then refines with a bounded 1D minimizer. Relative velocity at TCA is read
    directly off the propagated states, not finite-differenced.

    Returns a dict with tca_s, miss_distance_km, relative_speed_km_s, and the
    coarse (times_s, ranges_km) sample used for bracketing/plotting.
    """
    coarse_times_s, coarse_ranges_km = sample_range_history(
        state_fn_a, state_fn_b, t_start_s, t_end_s, coarse_step_s
    )
    min_idx = int(np.argmin(coarse_ranges_km))

    bracket_lo_s = coarse_times_s[max(min_idx - 1, 0)]
    bracket_hi_s = coarse_times_s[min(min_idx + 1, len(coarse_times_s) - 1)]

    result = minimize_scalar(
        lambda t_s: relative_range_km(state_fn_a, state_fn_b, t_s),
        bounds=(bracket_lo_s, bracket_hi_s),
        method="bounded",
    )

    tca_s = float(result.x)
    miss_distance_km = float(result.fun)

    r_a, v_a = state_fn_a(tca_s)
    r_b, v_b = state_fn_b(tca_s)
    relative_speed_km_s = float(np.linalg.norm(v_a - v_b))

    return {
        "tca_s": tca_s,
        "miss_distance_km": miss_distance_km,
        "relative_speed_km_s": relative_speed_km_s,
        "coarse_times_s": coarse_times_s,
        "coarse_ranges_km": coarse_ranges_km,
    }
