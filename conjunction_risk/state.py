"""State vector representation for a single space object."""

from dataclasses import dataclass

import numpy as np


@dataclass
class StateVector:
    """Position and velocity of an object at a given epoch, in an inertial frame.

    r_km and v_km_s are expressed in the frame named by `frame` (ECI by default,
    meaning Earth-Centered Inertial). epoch_s is seconds since an arbitrary but
    shared reference epoch — callers are responsible for keeping epochs of
    different StateVectors consistent with each other.
    """

    r_km: np.ndarray
    v_km_s: np.ndarray
    epoch_s: float = 0.0
    frame: str = "ECI"

    def __post_init__(self):
        self.r_km = np.asarray(self.r_km, dtype=float)
        self.v_km_s = np.asarray(self.v_km_s, dtype=float)

        if self.r_km.shape != (3,):
            raise ValueError(f"r_km must have shape (3,), got {self.r_km.shape}")
        if self.v_km_s.shape != (3,):
            raise ValueError(f"v_km_s must have shape (3,), got {self.v_km_s.shape}")
