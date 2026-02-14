"""Unit tests for the modular T-approach guidance implementation."""

from __future__ import annotations

import math
import unittest

import numpy as np

from minos.gnc.guidance.t_approach import (
    TApproachGuidance,
    _compute_required_heading,
    _smooth_heading_to_line_with_wind,
    _wrap_angle,
)
from minos.gnc.interfaces import MissionContext, NavigationEstimate, Observation
from minos.physics.types import State


def _make_observation(
    position: tuple[float, float, float] = (0.0, 0.0, 150.0),
    velocity: tuple[float, float, float] = (6.0, 0.0, -4.0),
    yaw_rad: float = 0.0,
) -> Observation:
    state = State(
        position=np.zeros(3),
        velocity_body=np.array([6.0, 0.0, 0.0], dtype=float),
        eulers=np.array([0.0, 0.0, yaw_rad], dtype=float),
        angular_velocity=np.zeros(3),
    )
    return Observation(
        time_s=0.0,
        state=state,
        inertial_position=np.asarray(position, dtype=float),
        inertial_velocity=np.asarray(velocity, dtype=float),
        wind_inertial=np.zeros(3),
        euler_rates=np.zeros(3),
    )


class TestGuidanceHelpers(unittest.TestCase):
    def test_wrap_angle_in_range(self) -> None:
        for angle in (-5.0 * np.pi, -0.1, 0.0, 0.5, 12.0):
            wrapped = _wrap_angle(angle)
            self.assertGreaterEqual(wrapped, 0.0)
            self.assertLess(wrapped, 2.0 * np.pi)

    def test_compute_required_heading_zero_target_vector(self) -> None:
        heading, air_vector = _compute_required_heading(
            wind_vector=np.array([1.0, 2.0], dtype=float),
            airspeed=6.0,
            target_vector=np.zeros(2),
        )
        self.assertEqual(heading, 0.0)
        np.testing.assert_allclose(air_vector, np.zeros(2))

    def test_compute_required_heading_unreachable_vector_falls_back_to_track(self) -> None:
        heading, air_vector = _compute_required_heading(
            wind_vector=np.array([20.0, 0.0], dtype=float),
            airspeed=5.0,
            target_vector=np.array([0.0, 30.0], dtype=float),
        )
        self.assertTrue(math.isfinite(heading))
        self.assertTrue(np.all(np.isfinite(air_vector)))
        self.assertAlmostEqual(heading, np.pi / 2.0, places=6)

    def test_smooth_heading_degenerate_line_direction_points_to_line_point(self) -> None:
        heading = _smooth_heading_to_line_with_wind(
            position=np.array([0.0, 0.0], dtype=float),
            line_point=np.array([10.0, 10.0], dtype=float),
            line_direction=np.zeros(2),
            lookahead_distance=10.0,
            wind_vector=np.zeros(2),
            airspeed=5.0,
        )
        self.assertAlmostEqual(heading, np.pi / 4.0, places=6)


class TestTApproachModes(unittest.TestCase):
    def setUp(self) -> None:
        self.guidance = TApproachGuidance()
        self.nav = NavigationEstimate(wind_inertial_estimate=np.zeros(3))

    def test_initialising_sets_start_heading(self) -> None:
        mission = MissionContext(phase="initialising", extras={})
        obs = _make_observation(yaw_rad=0.7)
        cmd = self.guidance.update(observation=obs, nav=self.nav, mission=mission, dt=0.1)

        self.assertTrue(mission.extras["initialised"])
        self.assertAlmostEqual(mission.extras["start_heading"], _wrap_angle(0.7), places=6)
        self.assertTrue(0.0 <= cmd.desired_heading < 2.0 * np.pi)

    def test_initialising_transitions_to_homing_after_full_rotation(self) -> None:
        mission = MissionContext(
            phase="initialising",
            extras={
                "mode": "initialising",
                "initialised": True,
                "start_heading": 0.0,
            },
        )
        # Keep position far so same-step homing->energy transition does not trigger.
        obs = _make_observation(position=(500.0, 0.0, 2000.0), yaw_rad=2.0 * np.pi + 0.2)
        self.guidance.update(observation=obs, nav=self.nav, mission=mission, dt=0.1)
        self.assertEqual(mission.extras["mode"], "homing")

    def test_homing_transitions_to_energy_management_when_near_tangent(self) -> None:
        mission = MissionContext(
            phase="homing",
            extras={
                "mode": "homing",
                "spirialing_radius": 20.0,
                "FTP_centre": np.array([10.0, 0.0], dtype=float),
                "IPI": np.array([0.0, 0.0, 0.0], dtype=float),
                "final_approach_height": 100.0,
                "sink_velocity": 4.9,
                "horizontal_velocity": 5.9,
                "wind_magnitude": 0.0,
                "wind_unit_vector": np.array([1.0, 0.0], dtype=float),
                "update_rate": 0.1,
                "desired_heading": 0.0,
                "initialised": True,
            },
        )
        obs = _make_observation(position=(0.0, 0.0, 150.0))
        self.guidance.update(observation=obs, nav=self.nav, mission=mission, dt=0.1)
        self.assertEqual(mission.extras["mode"], "energy_management")

    def test_energy_management_reverts_to_homing_when_far(self) -> None:
        mission = MissionContext(
            phase="energy_management",
            extras={
                "mode": "energy_management",
                "spirialing_radius": 10.0,
                "FTP_centre": np.array([0.0, 0.0], dtype=float),
                "IPI": np.array([0.0, 0.0, 0.0], dtype=float),
                "final_approach_height": 100.0,
                "sink_velocity": 4.9,
                "horizontal_velocity": 5.9,
                "wind_magnitude": 0.0,
                "wind_unit_vector": np.array([1.0, 0.0], dtype=float),
                "update_rate": 0.1,
                "desired_heading": 0.0,
            },
        )
        obs = _make_observation(position=(100.0, 0.0, 120.0))
        self.guidance.update(observation=obs, nav=self.nav, mission=mission, dt=0.1)
        self.assertEqual(mission.extras["mode"], "homing")

    def test_final_approach_sets_flare_when_below_flare_height(self) -> None:
        mission = MissionContext(
            phase="Final Approach",
            extras={
                "mode": "Final Approach",
                "flare_height": 20.0,
                "FTP_centre": np.array([0.0, 0.0], dtype=float),
                "IPI": np.array([0.0, 0.0, 0.0], dtype=float),
                "wind_unit_vector": np.array([1.0, 0.0], dtype=float),
                "wind_magnitude": 0.0,
                "horizontal_velocity": 5.9,
                "sink_velocity": 4.9,
                "final_approach_height": 100.0,
                "spirialing_radius": 20.0,
                "update_rate": 0.1,
                "desired_heading": 0.0,
            },
        )
        obs = _make_observation(position=(0.0, 0.0, 10.0))
        cmd = self.guidance.update(observation=obs, nav=self.nav, mission=mission, dt=0.1)
        self.assertEqual(mission.extras["mode"], "Final Approach")
        self.assertEqual(cmd.flare_magnitude, 1.0)

    def test_premature_ipi_branch_is_reachable(self) -> None:
        mission = MissionContext(
            phase="homing",
            extras={
                "mode": "homing",
                "IPI": np.array([200.0, 0.0, 0.0], dtype=float),
                "wind_unit_vector": np.array([1.0, 0.0], dtype=float),
                "wind_magnitude": 0.0,
                "horizontal_velocity": 5.0,
                "sink_velocity": 5.0,
                "final_approach_height": 100.0,
                "spirialing_radius": 20.0,
                "update_rate": 0.1,
                "flare_height": 20.0,
                "FTP_centre": np.array([0.0, 0.0], dtype=float),
                "desired_heading": 0.0,
            },
        )
        obs = _make_observation(position=(0.0, 0.0, 30.0))
        cmd = self.guidance.update(observation=obs, nav=self.nav, mission=mission, dt=0.1)
        self.assertEqual(mission.extras["mode"], "homing")
        self.assertTrue(0.0 <= cmd.desired_heading < 2.0 * np.pi)


if __name__ == "__main__":
    unittest.main()
