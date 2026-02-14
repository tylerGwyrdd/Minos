"""GnC orchestration layer for simulator integration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from minos.physics.types import Inputs

from .interfaces import (
    ControlCommand,
    Controller,
    GuidanceLaw,
    MissionContext,
    NavigationEstimate,
    Navigator,
    Observation,
    PhaseManager,
)


@dataclass
class GncStackConfig:
    """Configuration for stack wiring and command limits."""

    max_flap_abs: float | None = 1.0
    use_nav_wind_estimate: bool = False


class GncStack:
    """Run navigation, guidance, and control in a single update call."""

    def __init__(
        self,
        navigator: Navigator,
        guidance: GuidanceLaw,
        controller: Controller,
        mission: MissionContext | None = None,
        phase_manager: PhaseManager | None = None,
        config: GncStackConfig | None = None,
    ) -> None:
        self.navigator = navigator
        self.guidance = guidance
        self.controller = controller
        self.phase_manager = phase_manager
        self.mission = MissionContext() if mission is None else mission
        self.config = GncStackConfig() if config is None else config

        self.last_nav: NavigationEstimate | None = None
        self.last_guidance = None
        self.last_control = None
        self.last_control_raw = None

    def update(self, observation: Observation, dt: float) -> Inputs:
        """Run one full GnC cycle and return physics inputs."""
        dt = float(dt)
        nav = self.navigator.update(observation=observation, dt=dt, mission=self.mission)
        if self.phase_manager is not None:
            self.mission = self.phase_manager.update(observation=observation, nav=nav, mission=self.mission, dt=dt)

        guidance_cmd = self.guidance.update(
            observation=observation,
            nav=nav,
            mission=self.mission,
            dt=dt,
        )
        control_cmd_raw = self.controller.update(
            observation=observation,
            nav=nav,
            guidance=guidance_cmd,
            mission=self.mission,
            dt=dt,
        )

        raw_left = float(control_cmd_raw.flap_left)
        raw_right = float(control_cmd_raw.flap_right)
        flap_left, flap_right = self._validate_and_clip_flaps(raw_left, raw_right)
        control_cmd = ControlCommand(
            flap_left=flap_left,
            flap_right=flap_right,
            extras=dict(control_cmd_raw.extras),
        )
        self.last_control_raw = ControlCommand(
            flap_left=raw_left,
            flap_right=raw_right,
            extras=dict(control_cmd_raw.extras),
        )

        wind = observation.wind_inertial
        if self.config.use_nav_wind_estimate:
            wind = nav.wind_inertial_estimate

        self.last_nav = nav
        self.last_guidance = guidance_cmd
        self.last_control = control_cmd
        return Inputs(flap_left=flap_left, flap_right=flap_right, wind_inertial=wind)

    def _validate_and_clip_flaps(self, flap_left: float, flap_right: float) -> tuple[float, float]:
        if not np.isfinite(flap_left) or not np.isfinite(flap_right):
            raise ValueError("Controller returned non-finite flap command.")
        left = float(flap_left)
        right = float(flap_right)
        if self.config.max_flap_abs is not None:
            lim = float(self.config.max_flap_abs)
            left = float(np.clip(left, -lim, lim))
            right = float(np.clip(right, -lim, lim))
        return left, right
