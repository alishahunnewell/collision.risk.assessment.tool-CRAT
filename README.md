# Collision Risk Assessment Tool

A close-approach / collision-risk assessment toolkit for orbital objects. Given the state vectors of two bodies (two satellites, a satellite and a piece of debris, or a satellite/Earth and a near-Earth asteroid), the tool propagates their trajectories over a time window, computes time of closest approach (TCA), miss distance, and probability of collision (Pc), and flags close approaches worth a closer look.

The same core propagation and closest-approach geometry applies across two related domains: satellite conjunction assessment for mission operations, and near-Earth object close-approach screening for planetary defense.

## Status

**All four phases are done.** Two-body propagation, time-of-closest-approach finding, miss-distance-vs-time analysis, multi-object screening, probability of collision (via the 2D encounter-plane method), and along-track maneuver recommendation are implemented and tested. See [Roadmap](#roadmap).

## Installation

Requires Python 3.10+.

```bash
pip install -e ".[dev]"
```

## Usage

Run the Phase 1 example, which sets up two objects on a near-node-crossing conjunction and reports TCA, miss distance, and relative speed, then plots miss distance vs. time:

```bash
python examples/basic_conjunction.py
```

Run the Phase 2 example, which screens a small catalog of secondary objects against a primary and ranks them by miss distance:

```bash
python examples/screening_demo.py
```

Run the near-Earth object (NEO) example, which applies the same propagation and closest-approach code to a heliocentric Earth/asteroid encounter instead of a satellite pair:

```bash
python examples/neo_close_approach.py
```

Run the Phase 3 example, which attaches position covariances and a hard-body radius to a close approach and computes probability of collision (Pc):

```bash
python examples/probability_demo.py
```

Run the Phase 4 example, which finds the smallest along-track delta-v that clears a Pc threshold breach and reports before/after risk:

```bash
python examples/maneuver_demo.py
```

Run the test suite:

```bash
pytest
```

## How it works

- [`conjunction_risk/state.py`](conjunction_risk/state.py): `StateVector`, the position/velocity of an object at an epoch, in an inertial (ECI) frame.
- [`conjunction_risk/propagation.py`](conjunction_risk/propagation.py): two-body (Keplerian) propagation via the universal-variable formulation, valid for circular, elliptical, parabolic, and hyperbolic orbits.
- [`conjunction_risk/geometry.py`](conjunction_risk/geometry.py): samples relative range over a time window and refines to find time of closest approach (TCA), miss distance, and relative speed at TCA.
- [`conjunction_risk/screening.py`](conjunction_risk/screening.py): runs closest-approach finding across a catalog of secondary objects, ranks by miss distance, and flags the ones inside a threshold.
- [`conjunction_risk/probability.py`](conjunction_risk/probability.py): projects each object's position covariance onto the encounter plane and integrates the combined 2D Gaussian over a disk of radius equal to the combined hard-body radius to get Pc.
- [`conjunction_risk/maneuver.py`](conjunction_risk/maneuver.py): finds the smallest along-track delta-v that brings Pc at or below a target, evaluated at the original TCA rather than by re-searching for a new closest approach (see the module docstring for why that distinction matters).

The geometry module works against a `state_fn(t) -> (r_vec, v_vec)` interface rather than directly against the two-body propagator, so the same TCA-finding logic can be reused later with other propagation sources (e.g. SGP4 for TLE-based objects) without changes.

## Roadmap

1. **Phase 1, Geometry only** *(done)*: propagate two objects, compute miss distance vs. time, find TCA.
2. **Phase 2, Screening** *(done)*: given a primary object and a list of secondary objects (debris/TLE catalog, or a set of NEO orbital elements), rank by miss distance to find genuine close-approach events.
3. **Phase 3, Probability of collision** *(done)*: covariance handling and Pc via a standard 2D encounter-plane method.
4. **Phase 4, Maneuver recommendation** *(done)*: given a Pc threshold breach, propose a small along-track delta-v and show before/after risk.
