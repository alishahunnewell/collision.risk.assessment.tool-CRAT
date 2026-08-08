"""Physical constants used throughout the conjunction risk toolkit."""

MU_EARTH_KM3_S2 = 398600.4418
"""Earth gravitational parameter (GM), km^3/s^2."""

R_EARTH_KM = 6378.137
"""Earth mean equatorial radius, km."""

MU_SUN_KM3_S2 = 1.32712440018e11
"""Sun gravitational parameter (GM), km^3/s^2. Used for heliocentric propagation,
e.g. near-Earth object (NEO) orbits, as opposed to the geocentric MU_EARTH_KM3_S2
used for satellite/debris orbits."""

AU_KM = 149597870.7
"""Astronomical unit (mean Earth-Sun distance), km."""
