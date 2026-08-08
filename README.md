# Collision Risk Assessment Tool

A close-approach / collision-risk assessment toolkit for orbital objects. Given the state vectors of two bodies (two satellites, a satellite and a piece of debris, or a satellite/Earth and a near-Earth asteroid), the tool propagates their trajectories over a time window, computes time of closest approach (TCA), miss distance, and (eventually) probability of collision (Pc), and flags close approaches worth a closer look.

The same core propagation and closest-approach geometry applies across two related domains: satellite conjunction assessment for mission operations, and near-Earth object close-approach screening for planetary defense.

## Status

**Phase 1 (geometry) is done.** Two-body propagation, time-of-closest-approach finding, and miss-distance-vs-time analysis are implemented and tested. Phases 2-4 (multi-object screening, probability of collision, maneuver recommendations) are not yet built, see [Roadmap](#roadmap).

## Installation

Requires Python 3.10+.

```bash
pip install -e ".[dev]"
```

## Usage

Run the worked example, which sets up two objects on a near-node-crossing conjunction and reports TCA, miss distance, and relative speed, then plots miss distance vs. time:

```bash
python examples/basic_conjunction.py
```

Run the test suite:

```bash
pytest
```

## How it works

- [`conjunction_risk/state.py`](conjunction_risk/state.py): `StateVector`, the position/velocity of an object at an epoch, in an inertial (ECI) frame.
- [`conjunction_risk/propagation.py`](conjunction_risk/propagation.py): two-body (Keplerian) propagation via the universal-variable formulation, valid for circular, elliptical, parabolic, and hyperbolic orbits.
- [`conjunction_risk/geometry.py`](conjunction_risk/geometry.py): samples relative range over a time window and refines to find time of closest approach (TCA), miss distance, and relative speed at TCA.

The geometry module works against a `state_fn(t) -> (r_vec, v_vec)` interface rather than directly against the two-body propagator, so the same TCA-finding logic can be reused later with other propagation sources (e.g. SGP4 for TLE-based objects) without changes.

## Roadmap

1. **Phase 1, Geometry only** *(done)*: propagate two objects, compute miss distance vs. time, find TCA.
2. **Phase 2, Screening**: given a primary object and a list of secondary objects (debris/TLE catalog, or a set of NEO orbital elements), rank by miss distance to find genuine close-approach events.
3. **Phase 3, Probability of collision**: add covariance handling and compute Pc via a standard 2D encounter-plane method.
4. **Phase 4 (stretch)**: given a Pc threshold breach, propose a small delta-v maneuver and show before/after risk (satellite-specific).
