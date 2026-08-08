"""Multi-object screening: rank a catalog of secondary objects by closest approach to a primary.

Builds on find_closest_approach (geometry.py), running it once per secondary
and sorting the results so the closest, most concerning approaches surface
first. This is where a genuine conjunction gets separated from a routine
distant flyby.
"""

from dataclasses import dataclass

from .geometry import find_closest_approach, state_fn_from_state
from .state import StateVector


@dataclass
class ScreeningResult:
    """Closest-approach result for one named secondary object against the primary."""

    name: str
    tca_s: float
    miss_distance_km: float
    relative_speed_km_s: float


def screen_conjunctions(primary: StateVector, secondaries, t_start_s, t_end_s, coarse_step_s=10.0):
    """Screen a catalog of secondary objects against a primary object.

    primary: StateVector for the object being protected.
    secondaries: iterable of (name, StateVector) pairs.
    t_start_s, t_end_s, coarse_step_s: screening window, same meaning as in
    find_closest_approach.

    Returns a list of ScreeningResult, sorted by miss_distance_km ascending
    (closest, highest-priority approach first).
    """
    state_fn_primary = state_fn_from_state(primary)

    results = []
    for name, secondary_state in secondaries:
        state_fn_secondary = state_fn_from_state(secondary_state)
        approach = find_closest_approach(
            state_fn_primary, state_fn_secondary, t_start_s, t_end_s, coarse_step_s
        )
        results.append(
            ScreeningResult(
                name=name,
                tca_s=approach["tca_s"],
                miss_distance_km=approach["miss_distance_km"],
                relative_speed_km_s=approach["relative_speed_km_s"],
            )
        )

    results.sort(key=lambda result: result.miss_distance_km)
    return results


def filter_by_threshold(results, threshold_km):
    """Keep only screening results with miss_distance_km below threshold_km.

    Use after screen_conjunctions to separate genuine conjunction events
    (worth a closer look) from routine distant flybys.
    """
    return [result for result in results if result.miss_distance_km < threshold_km]
