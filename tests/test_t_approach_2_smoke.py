"""Smoke tests for pseudocode-based TApproachGuidance2."""

from __future__ import annotations

import unittest

import numpy as np

from minos.gnc.guidance.t_approach_2 import TApproachGuidance2
from minos.gnc.interfaces import MissionContext, NavigationEstimate, Observation
from minos.physics.types import State


def _obs(position: tuple[float, float, float], yaw: float = 0.0) -> Observation:
    return Observation(
        time_s=0.0,
        state=State(
            position=np.zeros(3),
            velocity_body=np.array([6.0, 0.0, 0.0], dtype=float),
            eulers=np.array([0.0, 0.0, yaw], dtype=float),
            angular_velocity=np.zeros(3),
        ),
        inertial_position=np.array(position, dtype=float),
        inertial_velocity=np.array([6.0, 0.0, -4.0], dtype=float),
        wind_inertial=np.zeros(3),
        euler_rates=np.zeros(3),
    )


class TestTApproach2Smoke(unittest.TestCase):
    def test_update_returns_heading_and_waypoint(self) -> None:
        guidance = TApproachGuidance2()
        mission = MissionContext(
            phase="homing",
            extras={
                "mode": "homing",
                "IPI": np.array([0.0, 0.0, 0.0], dtype=float),
            },
        )
        nav = NavigationEstimate(wind_inertial_estimate=np.array([1.0, 0.0, 0.0], dtype=float))
        cmd = guidance.update(_obs((80.0, 20.0, 120.0)), nav, mission, 0.1)
        self.assertTrue(np.isfinite(cmd.desired_heading))
        self.assertIn("waypoint_xy", cmd.extras)

    def test_flare_mode_commands_flare(self) -> None:
        guidance = TApproachGuidance2()
        mission = MissionContext(
            phase="flare",
            extras={
                "mode": "flare",
                "IPI": np.array([0.0, 0.0, 0.0], dtype=float),
            },
        )
        nav = NavigationEstimate(wind_inertial_estimate=np.zeros(3))
        cmd = guidance.update(_obs((2.0, 1.0, 5.0)), nav, mission, 0.1)
        self.assertEqual(cmd.flare_magnitude, 1.0)


if __name__ == "__main__":
    unittest.main()
