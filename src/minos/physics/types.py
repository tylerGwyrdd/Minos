"""Core dataclasses and validation utilities for the 6-DoF model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Sequence

import numpy as np


def as_vector3(value: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    """Return a validated 3-element float vector.

    Parameters
    ----------
    value
        Input values convertible to a flat vector of length 3.
    name
        Field name used in validation error messages.

    Returns
    -------
    np.ndarray
        A float array with shape ``(3,)``.
    """
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be length 3, got shape {arr.shape}.")
    return arr


def clamp_vector(vec: Sequence[float] | np.ndarray, max_abs: float = 1e3) -> np.ndarray:
    """Clamp non-finite or out-of-range vector values.

    Parameters
    ----------
    vec
        Vector-like input.
    max_abs
        Absolute limit applied to each component.

    Returns
    -------
    np.ndarray
        Stabilized array with NaNs replaced and values clipped to ``[-max_abs, max_abs]``.
    """
    arr = np.asarray(vec, dtype=float)
    if np.any(np.abs(arr) > max_abs) or np.any(np.isnan(arr)):
        arr = np.clip(arr, -max_abs, max_abs)
        arr = np.nan_to_num(arr, nan=0.0)
    return arr


@dataclass
class State:
    """Vehicle state used by the dynamics model.

    Attributes
    ----------
    position
        Position in model inertial coordinates (NED convention).
    velocity_body
        Body-frame translational velocity ``[u, v, w]``.
    eulers
        Euler angles ``[phi, theta, psi]`` in radians.
    angular_velocity
        Body-frame angular velocity ``[p, q, r]`` in rad/s.
    """

    position: np.ndarray
    velocity_body: np.ndarray
    eulers: np.ndarray
    angular_velocity: np.ndarray

    def __post_init__(self) -> None:
        self.position = as_vector3(self.position, "position")
        self.velocity_body = clamp_vector(as_vector3(self.velocity_body, "velocity_body"))
        self.eulers = as_vector3(self.eulers, "eulers")
        self.angular_velocity = clamp_vector(as_vector3(self.angular_velocity, "angular_velocity"))

    @classmethod
    def from_sequence(cls, values: Sequence[Sequence[float] | np.ndarray]) -> "State":
        """Build state from sequence form.

        Parameters
        ----------
        values
            Sequence ordered as ``[position, velocity_body, eulers, angular_velocity]``.

        Returns
        -------
        State
            Typed state instance.
        """
        return cls(position=values[0], velocity_body=values[1], eulers=values[2], angular_velocity=values[3])

    def as_sequence(self) -> list[np.ndarray]:
        """Return sequence form for ODE integration.

        Returns
        -------
        list[np.ndarray]
            Ordered as ``[position, velocity_body, eulers, angular_velocity]``.
        """
        return [
            self.position.copy(),
            self.velocity_body.copy(),
            self.eulers.copy(),
            self.angular_velocity.copy(),
        ]


@dataclass
class StateDerivative:
    """Time derivative of :class:`State`.

    Attributes
    ----------
    position_dot
        Inertial-frame velocity.
    velocity_dot
        Body-frame acceleration.
    eulers_dot
        Euler angle rates.
    angular_velocity_dot
        Body-frame angular acceleration.
    """

    position_dot: np.ndarray
    velocity_dot: np.ndarray
    eulers_dot: np.ndarray
    angular_velocity_dot: np.ndarray

    def __post_init__(self) -> None:
        self.position_dot = as_vector3(self.position_dot, "position_dot")
        self.velocity_dot = as_vector3(self.velocity_dot, "velocity_dot")
        self.eulers_dot = as_vector3(self.eulers_dot, "eulers_dot")
        self.angular_velocity_dot = as_vector3(self.angular_velocity_dot, "angular_velocity_dot")

    def as_sequence(self) -> list[np.ndarray]:
        """Return sequence form aligned with :meth:`State.as_sequence`.

        Returns
        -------
        list[np.ndarray]
            Ordered as ``[position_dot, velocity_dot, eulers_dot, angular_velocity_dot]``.
        """
        return [
            self.position_dot.copy(),
            self.velocity_dot.copy(),
            self.eulers_dot.copy(),
            self.angular_velocity_dot.copy(),
        ]


@dataclass
class Inputs:
    """External inputs at one simulation instant.

    Attributes
    ----------
    flap_left
        Left flap deflection command in radians.
    flap_right
        Right flap deflection command in radians.
    wind_inertial
        Inertial-frame wind vector in m/s.
    """

    flap_left: float
    flap_right: float
    wind_inertial: np.ndarray

    def __post_init__(self) -> None:
        self.flap_left = float(self.flap_left)
        self.flap_right = float(self.flap_right)
        self.wind_inertial = as_vector3(self.wind_inertial, "wind_inertial")


@dataclass
class AeroCoefficients:
    """Linearized aerodynamic coefficient set.

    Coefficients follow the model naming used in the dynamics equations
    (drag/lift/side-force and roll/pitch/yaw moment terms).
    """

    CDo: float = 0.25
    CDa: float = 0.12
    CD_sym: float = 0.2
    CLo: float = 0.091
    CLa: float = 0.90
    CL_sym: float = 0.2
    CYB: float = -0.23
    ClB: float = -0.036
    Clp: float = -0.84
    Clr: float = -0.082
    Cl_asym: float = -0.0035
    Cmo: float = 0.35
    Cma: float = -0.72
    Cmq: float = -1.49
    CnB: float = -0.0015
    Cn_p: float = -0.082
    Cn_r: float = -0.27
    Cn_asym: float = 0.0115

    ORDER: ClassVar[tuple[str, ...]] = (
        "CDo",
        "CDa",
        "CD_sym",
        "CLo",
        "CLa",
        "CL_sym",
        "CYB",
        "ClB",
        "Clp",
        "Clr",
        "Cl_asym",
        "Cmo",
        "Cma",
        "Cmq",
        "CnB",
        "Cn_p",
        "Cn_r",
        "Cn_asym",
    )

    def update(self, values: dict[str, float] | Sequence[float] | None) -> None:
        """Update coefficients from a mapping or ordered sequence.

        Parameters
        ----------
        values
            One of:
            - ``dict[str, float]`` keyed by coefficient name.
            - ordered sequence matching :attr:`ORDER`.
            - ``None`` (no-op).
        """
        if values is None:
            return
        if isinstance(values, dict):
            for key, val in values.items():
                if hasattr(self, key):
                    setattr(self, key, float(val))
            return
        if isinstance(values, (list, tuple)):
            if len(values) < len(self.ORDER):
                raise ValueError("Coefficient list must have at least 18 values.")
            for key, val in zip(self.ORDER, values):
                setattr(self, key, float(val))
            return
        raise TypeError("coefficients must be dict, list/tuple, or None")


@dataclass
class PhysicalParams:
    """Physical constants and geometry for the parafoil system.

    Notes
    -----
    Defaults represent the baseline model values used by this project.
    """

    dt: float = 0.1
    S: float = 1.0
    c: float = 0.75
    t: float = 0.075
    b: float = 1.35
    rigging_angle: float = np.radians(-12.0)
    m: float = 2.4
    Rp: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.11]))
    I: np.ndarray = field(
        default_factory=lambda: np.array([[0.42, 0.0, 0.03], [0.0, 0.4, 0.0], [0.03, 0.0, 0.053]])
    )
    initial_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    flap_time_constant: float = 1.0
    gravity: float = 9.81
    air_density: float = 1.293

    def __post_init__(self) -> None:
        self.Rp = as_vector3(self.Rp, "Rp")
        self.initial_pos = as_vector3(self.initial_pos, "initial_pos")
        self.I = np.asarray(self.I, dtype=float)
        if self.I.shape != (3, 3):
            raise ValueError(f"I must be 3x3, got shape {self.I.shape}.")
        self.dt = float(self.dt)
        self.S = float(self.S)
        self.c = float(self.c)
        self.t = float(self.t)
        self.b = float(self.b)
        self.rigging_angle = float(self.rigging_angle)
        self.m = float(self.m)
        self.flap_time_constant = float(self.flap_time_constant)
        self.gravity = float(self.gravity)
        self.air_density = float(self.air_density)

    @property
    def I_inv(self) -> np.ndarray:
        """Return inverse inertia tensor.

        Returns
        -------
        np.ndarray
            3x3 inverse of :attr:`I`.
        """
        return np.linalg.inv(self.I)

    @classmethod
    def from_mapping(cls, values: dict[str, object] | None) -> "PhysicalParams":
        """Construct parameters from a partially specified mapping.

        Parameters
        ----------
        values
            Mapping of parameter names to overrides. Unknown keys are ignored.

        Returns
        -------
        PhysicalParams
            Validated parameter set.
        """
        params = cls()
        if values is None:
            return params
        for key, value in values.items():
            if hasattr(params, key):
                setattr(params, key, value)
        params.__post_init__()
        return params
