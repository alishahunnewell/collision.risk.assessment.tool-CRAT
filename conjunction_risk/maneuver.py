"""Maneuver recommendation: propose a small delta-v to clear a Pc threshold breach.

A small along-track delta-v (a burn along the velocity direction, i.e. the
"in-track" direction in the radial/transverse/normal (RTN) sense) applied to
the primary, well before the time of closest approach (TCA), shifts when the
primary arrives at the conjunction point. For two objects on genuinely
different orbits, that time shift generally moves them apart at the moment
the conjunction was predicted to occur, which is the standard, fuel-efficient
collision-avoidance maneuver strategy.

Evaluating a candidate maneuver is done at the ORIGINAL, unmaneuvered TCA
(a fixed time), not by re-running find_closest_approach on the maneuvered
trajectory to search for a new closest approach. That distinction matters:
a full re-search can wander to a different, coincidentally close pass
elsewhere in the orbit (particularly for near-resonant geometries, where the
same two objects nearly repeat their alignment every period), which does not
answer the operational question of interest, "how far apart are they at the
moment the predicted conjunction was going to happen." This module reuses
probability_of_collision (not probability_of_collision_at_tca) for exactly
this reason, passing it relative state at the fixed reference time.
"""

import numpy as np
from scipy.optimize import brentq

from .constants import MU_EARTH_KM3_S2
from .geometry import find_closest_approach, state_fn_from_state
from .probability import encounter_plane_basis, probability_of_collision, probability_of_collision_at_tca
from .state import StateVector


def along_track_unit_vector(v_km_s):
    """Unit vector along the velocity direction (along-track/in-track)."""
    v_km_s = np.asarray(v_km_s, dtype=float)
    return v_km_s / np.linalg.norm(v_km_s)


def apply_delta_v(state: StateVector, delta_v_km_s) -> StateVector:
    """Return a new StateVector with delta_v_km_s (a 3-vector, km/s) added to
    velocity, at the same epoch, position, frame, and covariance."""
    delta_v_km_s = np.asarray(delta_v_km_s, dtype=float)
    return StateVector(
        r_km=state.r_km.copy(),
        v_km_s=state.v_km_s + delta_v_km_s,
        epoch_s=state.epoch_s,
        frame=state.frame,
        cov_km2=state.cov_km2,
    )


def evaluate_conjunction(
    primary,
    secondary,
    t_start_s,
    t_end_s,
    coarse_step_s,
    cov_a_km2,
    cov_b_km2,
    hbr_km,
    mu_km3_s2=MU_EARTH_KM3_S2,
):
    """Baseline screening: full closest-approach search plus Pc.

    Used to establish the reference TCA and confirm a Pc threshold breach,
    before evaluating candidate maneuvers against that fixed reference time
    (see _evaluate_at_fixed_time).

    Returns a dict with tca_s, miss_distance_km, relative_speed_km_s, and pc.
    """
    state_fn_primary = state_fn_from_state(primary, mu_km3_s2)
    state_fn_secondary = state_fn_from_state(secondary, mu_km3_s2)

    closest_approach = find_closest_approach(
        state_fn_primary, state_fn_secondary, t_start_s, t_end_s, coarse_step_s
    )
    pc = probability_of_collision_at_tca(closest_approach, cov_a_km2, cov_b_km2, hbr_km)

    return {
        "tca_s": closest_approach["tca_s"],
        "miss_distance_km": closest_approach["miss_distance_km"],
        "relative_speed_km_s": closest_approach["relative_speed_km_s"],
        "pc": pc,
    }


def _evaluate_at_fixed_time(primary, secondary, t_s, cov_a_km2, cov_b_km2, hbr_km, mu_km3_s2):
    """Evaluate relative geometry and Pc at a fixed time t_s (no TCA search).

    See module docstring for why a fixed reference time, rather than a fresh
    closest-approach search, is the right way to assess a candidate maneuver.
    """
    state_fn_primary = state_fn_from_state(primary, mu_km3_s2)
    state_fn_secondary = state_fn_from_state(secondary, mu_km3_s2)

    r_a, v_a = state_fn_primary(t_s)
    r_b, v_b = state_fn_secondary(t_s)
    r_rel_km = r_a - r_b
    v_rel_km_s = v_a - v_b

    x_hat, _, _ = encounter_plane_basis(r_rel_km, v_rel_km_s)
    miss_distance_km = float(np.dot(r_rel_km, x_hat))
    pc = probability_of_collision(r_rel_km, v_rel_km_s, cov_a_km2, cov_b_km2, hbr_km)

    return {
        "miss_distance_km": miss_distance_km,
        "relative_speed_km_s": float(np.linalg.norm(v_rel_km_s)),
        "pc": pc,
    }


def recommend_along_track_delta_v(
    primary,
    secondary,
    target_pc,
    t_start_s,
    t_end_s,
    coarse_step_s,
    cov_a_km2,
    cov_b_km2,
    hbr_km,
    max_delta_v_km_s=0.001,
    mu_km3_s2=MU_EARTH_KM3_S2,
):
    """Find the smallest along-track delta-v (km/s) that brings Pc at or below target_pc.

    Probes a small delta-v magnitude in both the prograde and retrograde
    along-track directions to see which one lowers Pc (evaluated at the
    original TCA, see _evaluate_at_fixed_time), then bisects that direction,
    assuming Pc decreases monotonically with delta-v magnitude there (true
    for a small delta-v applied to a genuine close approach), for the
    minimum magnitude, up to max_delta_v_km_s, that clears target_pc.

    Returns a dict with delta_v_km_s (signed 3-vector, along the primary's
    velocity direction), before (evaluate_conjunction result, pre-maneuver),
    and after (the post-maneuver miss distance/relative speed/Pc at the
    original TCA).
    """
    before = evaluate_conjunction(
        primary, secondary, t_start_s, t_end_s, coarse_step_s, cov_a_km2, cov_b_km2, hbr_km, mu_km3_s2
    )
    if before["pc"] <= target_pc:
        raise ValueError("Pc is already at or below target_pc; no maneuver needed.")

    tca_s = before["tca_s"]
    along_track = along_track_unit_vector(primary.v_km_s)

    def pc_at(delta_v_mag):
        maneuvered = apply_delta_v(primary, along_track * delta_v_mag)
        return _evaluate_at_fixed_time(maneuvered, secondary, tca_s, cov_a_km2, cov_b_km2, hbr_km, mu_km3_s2)["pc"]

    probe_km_s = max_delta_v_km_s * 0.01
    sign = 1.0 if pc_at(probe_km_s) <= pc_at(-probe_km_s) else -1.0

    def pc_minus_target(delta_v_mag):
        return pc_at(sign * delta_v_mag) - target_pc

    if pc_minus_target(max_delta_v_km_s) > 0.0:
        raise ValueError(
            f"No delta-v up to {max_delta_v_km_s} km/s (in the better along-track "
            f"direction) brings Pc down to target_pc={target_pc}."
        )

    delta_v_mag_km_s = brentq(pc_minus_target, 0.0, max_delta_v_km_s, xtol=1e-9)
    delta_v_km_s = sign * delta_v_mag_km_s * along_track

    maneuvered_primary = apply_delta_v(primary, delta_v_km_s)
    after = _evaluate_at_fixed_time(maneuvered_primary, secondary, tca_s, cov_a_km2, cov_b_km2, hbr_km, mu_km3_s2)

    return {
        "delta_v_km_s": delta_v_km_s,
        "before": before,
        "after": after,
    }
