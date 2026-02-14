"""Typed contracts for modular guidance, navigation, and control.

These interfaces define how navigation, guidance, control, and optional phase
management components exchange data each update step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from minos.physics.types import Inputs, State, as_vector3


@dataclass
class GuidanceMarker:
    """Represent one guidance-defined marker in 3D space.

    Guidance laws can publish these markers for plotting and diagnostics
    without coupling plotting utilities to guidance-specific dictionary keys.

    ### Parameters
    - `marker_id` (`str`): Stable identifier used for deterministic styling.
    - `label` (`str`): Human-readable marker label for plot legends.
    - `xyz` (`np.ndarray`): Marker position in inertial coordinates, shape `(3,)`.
    - `kind` (`str`): Marker semantic category (for example `"point"` or `"active"`).
    - `style_hint` (`dict[str, Any]`): Optional rendering hints consumed by visualizers.

    ### Returns
    - `GuidanceMarker`: Dataclass instance with normalized field types.

    ### Raises
    - `ValueError`: If `xyz` cannot be converted to a 3-vector.

    ### Notes
    - `xyz` is validated via `as_vector3`, enforcing shape and float coercion.
    - `style_hint` content is intentionally untyped to keep renderer coupling low.
    """

    marker_id: str
    label: str
    xyz: np.ndarray
    kind: str = "point"
    style_hint: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.marker_id = str(self.marker_id)
        self.label = str(self.label)
        self.kind = str(self.kind)
        self.xyz = as_vector3(self.xyz, "xyz")
        self.style_hint = dict(self.style_hint)


@dataclass
class GuidanceVisualization:
    """Bundle optional visualization metadata produced by guidance.

    This payload is passed alongside guidance commands so plotting helpers can
    render waypoints and mode styling using typed structures rather than
    hardcoded key lookups.

    ### Parameters
    - `markers` (`list[GuidanceMarker]`): Marker set to render.
    - `mode_styles` (`dict[str, dict[str, Any]]`): Optional per-mode style hints.

    ### Returns
    - `GuidanceVisualization`: Dataclass instance with copied containers.

    ### Raises
    - None directly. Downstream code may reject malformed style payloads.

    ### Notes
    - Containers are copied in `__post_init__` to avoid accidental aliasing.
    - `mode_styles` keys are coerced to `str` for deterministic lookups.
    """

    markers: list[GuidanceMarker] = field(default_factory=list)
    mode_styles: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.markers = list(self.markers)
        self.mode_styles = {str(k): dict(v) for k, v in self.mode_styles.items()}


@dataclass
class Observation:
    """Provide one normalized plant observation to GnC modules.

    This object is the canonical input to navigator/guidance/controller update
    calls for each simulation or control cycle.

    ### Parameters
    - `time_s` (`float`): Sample time in seconds.
    - `state` (`State`): Full body-frame state object from the physics layer.
    - `inertial_position` (`np.ndarray`): Position vector in inertial frame `(3,)`.
    - `inertial_velocity` (`np.ndarray`): Velocity vector in inertial frame `(3,)`.
    - `wind_inertial` (`np.ndarray`): Wind vector in inertial frame `(3,)`.
    - `euler_rates` (`np.ndarray`): Euler angle rates in body/kinematic convention `(3,)`.

    ### Returns
    - `Observation`: Dataclass instance with validated vector fields.

    ### Raises
    - `ValueError`: If any vector field cannot be converted to shape `(3,)`.

    ### Notes
    - Vector normalization ensures all modules receive consistent shapes/types.
    """

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
    """Contain navigation outputs shared with guidance and phase logic.

    ### Parameters
    - `wind_inertial_estimate` (`np.ndarray`): Estimated inertial-frame wind `(3,)`.
    - `extras` (`dict[str, Any]`): Optional estimator-specific telemetry.

    ### Returns
    - `NavigationEstimate`: Dataclass instance with normalized wind vector.

    ### Raises
    - `ValueError`: If `wind_inertial_estimate` is not a 3-vector.

    ### Notes
    - `extras` is intentionally open-ended for algorithm-specific diagnostics.
    """

    wind_inertial_estimate: np.ndarray
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.wind_inertial_estimate = as_vector3(self.wind_inertial_estimate, "wind_inertial_estimate")


@dataclass
class GuidanceCommand:
    """Represent high-level guidance output for the current cycle.

    ### Parameters
    - `desired_heading` (`float`): Desired yaw/heading command in radians.
    - `flare_magnitude` (`float`): Normalized flare command scalar.
    - `extras` (`dict[str, Any]`): Optional metadata for logging/telemetry.
    - `visualization` (`GuidanceVisualization | None`): Optional plot payload.

    ### Returns
    - `GuidanceCommand`: Dataclass instance with normalized scalar fields.

    ### Raises
    - None directly.

    ### Notes
    - Heading units are radians throughout the GnC stack.
    - `visualization` is optional to avoid forcing all guidance laws to provide it.
    """

    desired_heading: float
    flare_magnitude: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)
    visualization: GuidanceVisualization | None = None

    def __post_init__(self) -> None:
        self.desired_heading = float(self.desired_heading)
        self.flare_magnitude = float(self.flare_magnitude)


@dataclass
class ControlCommand:
    """Represent low-level actuator demands produced by the controller.

    ### Parameters
    - `flap_left` (`float`): Left flap command.
    - `flap_right` (`float`): Right flap command.
    - `extras` (`dict[str, Any]`): Optional controller telemetry.

    ### Returns
    - `ControlCommand`: Dataclass instance with float-normalized commands.

    ### Raises
    - None directly.

    ### Notes
    - Command units match the physics model's flap input convention.
    """

    flap_left: float
    flap_right: float
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.flap_left = float(self.flap_left)
        self.flap_right = float(self.flap_right)

    def to_inputs(self, wind_inertial: np.ndarray) -> Inputs:
        """Convert this command into the physics model input container.

        ### Parameters
        - `wind_inertial` (`np.ndarray`): Inertial-frame wind vector applied by plant.

        ### Returns
        - `Inputs`: Physics-layer input structure.

        ### Raises
        - `ValueError`: If `wind_inertial` is incompatible with `Inputs` validation.

        ### Notes
        - This keeps actuator and wind routing explicit at the stack boundary.
        """
        return Inputs(self.flap_left, self.flap_right, wind_inertial)


@dataclass
class MissionContext:
    """Store mutable mission state shared across GnC components.

    ### Parameters
    - `phase` (`str`): Current mission phase/mode name.
    - `extras` (`dict[str, Any]`): Shared mutable bag for mission/guidance state.

    ### Returns
    - `MissionContext`: Dataclass instance.

    ### Raises
    - None directly.

    ### Notes
    - `extras` enables low-friction state sharing but should stay structured by convention.
    """

    phase: str = "default"
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Navigator(Protocol):
    """Define the navigation/estimation module contract.

    Implementations consume observations and emit `NavigationEstimate` each
    cycle.

    ### Notes
    - `dt` is the estimator update period in seconds.
    - Returned estimates are expected in inertial frame.
    """

    def update(self, observation: Observation, dt: float, mission: MissionContext) -> NavigationEstimate: ...


@runtime_checkable
class GuidanceLaw(Protocol):
    """Define the guidance strategy contract.

    Implementations map observation and navigation state to high-level commands.

    ### Notes
    - Guidance output should remain frame-consistent with controller expectations.
    - `dt` is guidance update period in seconds.
    """

    def update(
        self,
        observation: Observation,
        nav: NavigationEstimate,
        mission: MissionContext,
        dt: float,
    ) -> GuidanceCommand: ...


@runtime_checkable
class Controller(Protocol):
    """Define the low-level controller contract.

    Implementations map guidance commands and current state to flap actuation.

    ### Notes
    - Controller should treat `dt` as the control integration/update period.
    """

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
    """Define optional mission phase manager contract.

    Phase managers can mutate/replace mission context between navigation and
    guidance updates.

    ### Notes
    - This hook is useful for supervisory logic independent of guidance internals.
    """

    def update(
        self,
        observation: Observation,
        nav: NavigationEstimate,
        mission: MissionContext,
        dt: float,
    ) -> MissionContext: ...
