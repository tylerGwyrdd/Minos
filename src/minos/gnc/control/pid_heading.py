"""Heading controller strategy for flap commands."""

from __future__ import annotations

import numpy as np

from ..interfaces import ControlCommand, Controller, GuidanceCommand, MissionContext, NavigationEstimate, Observation


def _wrap_angle_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class PidHeadingController(Controller):
    """Map heading tracking error to flap deflections using a PD law."""

    def __init__(self, kp: float = 3.0, kd: float = 4.0, max_deflection: float = 0.6) -> None:
        self.kp = float(kp)
        self.kd = float(kd)
        self.max_deflection = float(max_deflection)

    def update(
        self,
        observation: Observation,
        nav: NavigationEstimate,
        guidance: GuidanceCommand,
        mission: MissionContext,
        dt: float,
    ) -> ControlCommand:
        del nav, mission, dt
        current_heading = _wrap_angle_pi(float(observation.state.eulers[2]))
        desired_heading = _wrap_angle_pi(float(guidance.desired_heading))
        heading_rate = float(observation.euler_rates[2])
        heading_error = _wrap_angle_pi(desired_heading - current_heading)

        control_effort = self.kp * heading_error - self.kd * heading_rate
        control_effort = float(np.clip(control_effort, -self.max_deflection, self.max_deflection))

        if control_effort > 0.0:
            flap_left, flap_right = 0.0, control_effort
        else:
            flap_left, flap_right = -control_effort, 0.0
        return ControlCommand(flap_left=flap_left, flap_right=flap_right)
