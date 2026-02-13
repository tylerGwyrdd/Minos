"""Core abstractions for aerodynamic coefficient identification.

These types are intentionally small and strict. Identification loops evaluate
thousands of candidates, so we validate once at construction boundaries and keep
runtime objects predictable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from minos.physics.types import AeroCoefficients


@dataclass(frozen=True)
class FlightDataset:
    """Measured flight data and control inputs used for identification.

    Why this exists
    ----------------
    Legacy scripts passed several loosely-related arrays around. Consolidating
    them into one validated object reduces shape/order bugs and makes evaluator
    code independent from experiment scripts.
    """

    time_s: np.ndarray
    flap_left: np.ndarray
    flap_right: np.ndarray
    wind_inertial: np.ndarray
    measured_positions: np.ndarray

    def __post_init__(self) -> None:
        # Fail fast on malformed datasets. It is cheaper to reject invalid data
        # once than to discover hard-to-debug numerical issues inside GA loops.
        n = int(np.asarray(self.time_s).size)
        if n < 2:
            raise ValueError("time_s must contain at least 2 samples.")
        if np.asarray(self.flap_left).size != n:
            raise ValueError("flap_left length must match time_s.")
        if np.asarray(self.flap_right).size != n:
            raise ValueError("flap_right length must match time_s.")
        if np.asarray(self.wind_inertial).shape != (n, 3):
            raise ValueError("wind_inertial must have shape (N, 3).")
        if np.asarray(self.measured_positions).shape != (n, 3):
            raise ValueError("measured_positions must have shape (N, 3).")

    def sim_inputs(self) -> list[object]:
        """Return input format expected by simulation runners.

        Why list[object]
        ----------------
        Simulation wrappers currently accept a historical input layout:
        ``[left_series, right_series, wind_list]``. We keep this adapter to
        preserve compatibility while the rest of identification stays typed.
        """
        wind_list = [np.asarray(w, dtype=float) for w in np.asarray(self.wind_inertial)]
        return [np.asarray(self.flap_left), np.asarray(self.flap_right), wind_list]


@dataclass(frozen=True)
class CoefficientBounds:
    """Bounds keyed by aerodynamic coefficient name.

    Name-based bounds are used instead of positional tuples so experiments can
    optimize arbitrary subsets without relying on fragile index conventions.
    """

    by_name: Mapping[str, tuple[float, float]]

    def for_names(self, names: Sequence[str]) -> tuple[list[float], list[float]]:
        """Return lower/upper vectors in the requested coefficient order."""
        lows: list[float] = []
        highs: list[float] = []
        for name in names:
            if name not in self.by_name:
                raise KeyError(f"Missing bounds for coefficient '{name}'.")
            lo, hi = self.by_name[name]
            lows.append(float(lo))
            highs.append(float(hi))
        return lows, highs


@dataclass(frozen=True)
class CoefficientCodec:
    """Encode/decode coefficients using one canonical order.

    The canonical order is sourced from :class:`AeroCoefficients` so physics and
    identification share one definition of coefficient ordering.
    """

    order: tuple[str, ...] = AeroCoefficients.ORDER

    def to_vector(self, coeffs: Mapping[str, float] | AeroCoefficients) -> np.ndarray:
        """Encode dictionary/object coefficients into ordered vector form.

        GA operators are vector-based, so this is the only place where we allow
        positional encoding. Everything else remains name-addressable.
        """
        if isinstance(coeffs, AeroCoefficients):
            src = {name: getattr(coeffs, name) for name in self.order}
        else:
            src = coeffs
        return np.asarray([float(src[name]) for name in self.order], dtype=float)

    def to_dict(self, vector: Sequence[float]) -> dict[str, float]:
        """Decode ordered vector into dictionary form."""
        if len(vector) != len(self.order):
            raise ValueError(f"Expected vector of length {len(self.order)}, got {len(vector)}.")
        return {name: float(value) for name, value in zip(self.order, vector)}

    def merge_partial(
        self,
        base: Mapping[str, float] | AeroCoefficients,
        optimize_names: Sequence[str],
        optimize_values: Sequence[float],
    ) -> dict[str, float]:
        """Merge a partial optimization vector into a base coefficient set.

        This supports grouped optimization runs: non-optimized coefficients stay
        fixed at their base values while selected coefficients are replaced.
        """
        base_dict = self.to_dict(self.to_vector(base))
        if len(optimize_names) != len(optimize_values):
            raise ValueError("optimize_names and optimize_values must have equal length.")
        for name, value in zip(optimize_names, optimize_values):
            if name not in base_dict:
                raise KeyError(f"Unknown coefficient '{name}'.")
            base_dict[name] = float(value)
        return base_dict
