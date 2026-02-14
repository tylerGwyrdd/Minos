"""Minimal branch probes for T-approach guidance modes.

Each probe feeds a crafted observation/mission into one update step so a
specific guidance branch can be checked quickly.
"""

from __future__ import annotations

import numpy as np

from minos.gnc.guidance.t_approach import TApproachGuidance
from minos.gnc.interfaces import MissionContext, NavigationEstimate, Observation
from minos.physics.types import State


def _obs(position_xyz: tuple[float, float, float], yaw: float = 0.0) -> Observation:
    return Observation(
        time_s=0.0,
        state=State(
            position=np.zeros(3),
            velocity_body=np.array([6.0, 0.0, 0.0], dtype=float),
            eulers=np.array([0.0, 0.0, yaw], dtype=float),
            angular_velocity=np.zeros(3),
        ),
        inertial_position=np.array(position_xyz, dtype=float),
        inertial_velocity=np.array([6.0, 0.0, -4.9], dtype=float),
        wind_inertial=np.zeros(3),
        euler_rates=np.zeros(3),
    )


def _run_case(name: str, mission: MissionContext, observation: Observation) -> None:
    guidance = TApproachGuidance()
    nav = NavigationEstimate(wind_inertial_estimate=np.zeros(3))
    cmd = guidance.update(observation=observation, nav=nav, mission=mission, dt=0.1)
    print(
        f"{name}: mode={mission.extras.get('mode')}, "
        f"desired_heading_deg={np.rad2deg(cmd.desired_heading):.2f}, "
        f"flare={cmd.flare_magnitude:.1f}"
    )


def main() -> None:
    _run_case(
        "initialising",
        MissionContext(phase="initialising", extras={}),
        _obs((0.0, 0.0, 150.0), yaw=0.3),
    )
    _run_case(
        "full_rotation_to_homing",
        MissionContext(
            phase="initialising",
            extras={"mode": "initialising", "initialised": True, "start_heading": 0.0},
        ),
        _obs((0.0, 0.0, 150.0), yaw=2.0 * np.pi + 0.1),
    )
    _run_case(
        "homing_to_energy_management",
        MissionContext(
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
        ),
        _obs((0.0, 0.0, 150.0), yaw=0.0),
    )
    _run_case(
        "final_approach_with_flare",
        MissionContext(
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
        ),
        _obs((0.0, 0.0, 10.0), yaw=0.0),
    )


if __name__ == "__main__":
    main()
