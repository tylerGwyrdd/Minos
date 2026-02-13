"""Typed contracts for modular guidance, navigation, and control."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from minos.physics.types import Inputs, State, as_vector3


@dataclass
class Observation:
    """Immutable snapshot consumed by GnC components."""

    time_s: float
    state: State
    inertial_position: np.ndarray
    inertial_velocity: np.ndarray
    wind_inertial: np.ndarray
    euler_rates: np.ndarray

    def __post_init__(self) -> None:
        self.time_s = float(self.time_s)
        self.inertial_position = as_vector3(self.inertial_position, "inertial_position")
        self.inertial_velocity = as_vector3(self.inertial_velocity, "inertial_velocity")
        self.wind_inertial = as_vector3(self.wind_inertial, "wind_inertial")
        self.euler_rates = as_vector3(self.euler_rates, "euler_rates")


@dataclass
class NavigationEstimate:
    """Navigation output shared with guidance and phase logic."""

    wind_inertial_estimate: np.ndarray
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.wind_inertial_estimate = as_vector3(self.wind_inertial_estimate, "wind_inertial_estimate")


@dataclass
class GuidanceCommand:
    """High-level command produced by the guidance law."""

    desired_heading: float
    flare_magnitude: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.desired_heading = float(self.desired_heading)
        self.flare_magnitude = float(self.flare_magnitude)


@dataclass
class ControlCommand:
    """Low-level actuation command produced by the controller."""

    flap_left: float
    flap_right: float
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.flap_left = float(self.flap_left)
        self.flap_right = float(self.flap_right)

    def to_inputs(self, wind_inertial: np.ndarray) -> Inputs:
        """Convert to physics input container."""
        return Inputs(self.flap_left, self.flap_right, wind_inertial)


@dataclass
class MissionContext:
    """Mutable mission-level context shared across GnC components."""

    phase: str = "default"
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Navigator(Protocol):
    """State estimator contract."""

    def update(self, observation: Observation, dt: float, mission: MissionContext) -> NavigationEstimate: ...


@runtime_checkable
class GuidanceLaw(Protocol):
    """Guidance strategy contract."""

    def update(
        self,
        observation: Observation,
        nav: NavigationEstimate,
        mission: MissionContext,
        dt: float,
    ) -> GuidanceCommand: ...


@runtime_checkable
class Controller(Protocol):
    """Controller contract mapping guidance to flap deflections."""

    def update(
        self,
        observation: Observation,
        nav: NavigationEstimate,
        guidance: GuidanceCommand,
        mission: MissionContext,
        dt: float,
    ) -> ControlCommand: ...


@runtime_checkable
class PhaseManager(Protocol):
    """Optional mission phase manager contract."""

    def update(
        self,
        observation: Observation,
        nav: NavigationEstimate,
        mission: MissionContext,
        dt: float,
    ) -> MissionContext: ...
