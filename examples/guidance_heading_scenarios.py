"""Heading-guidance scenario runner with desired vs actual path comparison.

Scenarios:
1) Straight-line heading hold.
2) Step turn at a specified time.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from minos.gnc.control.pid_heading import PidHeadingController
from minos.gnc.interfaces import GuidanceCommand, MissionContext, NavigationEstimate, Observation
from minos.physics.types import State


def _wrap_angle_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class ScenarioResult:
    name: str
    times: np.ndarray
    desired_heading: np.ndarray
    actual_heading: np.ndarray
    desired_path: np.ndarray
    actual_path: np.ndarray
    rmse_m: float
    final_error_m: float


def _integrate_desired_path(times: np.ndarray, speed: float, heading_fn: Callable[[float], float]) -> tuple[np.ndarray, np.ndarray]:
    headings = np.array([heading_fn(t) for t in times], dtype=float)
    pos = np.zeros((len(times), 2), dtype=float)
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        v = speed * np.array([np.cos(headings[i - 1]), np.sin(headings[i - 1])], dtype=float)
        pos[i] = pos[i - 1] + v * dt
    return headings, pos


def run_heading_scenario(
    name: str,
    heading_fn: Callable[[float], float],
    total_time_s: float = 60.0,
    dt_s: float = 0.1,
    speed_mps: float = 6.0,
    yaw_tau_s: float = 0.9,
    turn_gain: float = 2.0,
) -> ScenarioResult:
    times = np.arange(0.0, total_time_s + dt_s, dt_s)
    desired_heading, desired_path = _integrate_desired_path(times, speed_mps, heading_fn)

    controller = PidHeadingController(kp=3.0, kd=4.0, max_deflection=0.6)
    mission = MissionContext()
    nav = NavigationEstimate(wind_inertial_estimate=np.zeros(3))

    actual_heading = np.zeros_like(times)
    actual_path = np.zeros((len(times), 2), dtype=float)
    yaw_rate = 0.0

    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        obs = Observation(
            time_s=times[i - 1],
            state=State(
                position=np.zeros(3),
                velocity_body=np.array([speed_mps, 0.0, 0.0], dtype=float),
                eulers=np.array([0.0, 0.0, actual_heading[i - 1]], dtype=float),
                angular_velocity=np.array([0.0, 0.0, yaw_rate], dtype=float),
            ),
            inertial_position=np.array([actual_path[i - 1, 0], actual_path[i - 1, 1], 0.0], dtype=float),
            inertial_velocity=np.array(
                [speed_mps * np.cos(actual_heading[i - 1]), speed_mps * np.sin(actual_heading[i - 1]), 0.0],
                dtype=float,
            ),
            wind_inertial=np.zeros(3),
            euler_rates=np.array([0.0, 0.0, yaw_rate], dtype=float),
        )
        cmd = controller.update(
            observation=obs,
            nav=nav,
            guidance=GuidanceCommand(desired_heading=desired_heading[i - 1]),
            mission=mission,
            dt=dt,
        )
        signed_flap = float(cmd.flap_right - cmd.flap_left)
        yaw_rate_target = turn_gain * signed_flap
        yaw_rate += (yaw_rate_target - yaw_rate) * (dt / yaw_tau_s)
        actual_heading[i] = _wrap_angle_pi(actual_heading[i - 1] + yaw_rate * dt)
        vel = speed_mps * np.array([np.cos(actual_heading[i]), np.sin(actual_heading[i])], dtype=float)
        actual_path[i] = actual_path[i - 1] + vel * dt

    diff = desired_path - actual_path
    rmse_m = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
    final_error_m = float(np.linalg.norm(diff[-1]))
    return ScenarioResult(
        name=name,
        times=times,
        desired_heading=desired_heading,
        actual_heading=actual_heading,
        desired_path=desired_path,
        actual_path=actual_path,
        rmse_m=rmse_m,
        final_error_m=final_error_m,
    )


def plot_results(results: list[ScenarioResult]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for result in results:
        axes[0].plot(result.desired_path[:, 0], result.desired_path[:, 1], "--", label=f"{result.name} desired")
        axes[0].plot(result.actual_path[:, 0], result.actual_path[:, 1], label=f"{result.name} actual")
    axes[0].set_title("Desired vs Actual XY Paths")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    axes[0].axis("equal")
    axes[0].grid(True)
    axes[0].legend()

    for result in results:
        axes[1].plot(result.times, np.rad2deg(result.desired_heading), "--", label=f"{result.name} desired")
        axes[1].plot(result.times, np.rad2deg(result.actual_heading), label=f"{result.name} actual")
    axes[1].set_title("Heading Tracking")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Heading (deg)")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()


def main(show_plot: bool = True) -> None:
    straight = run_heading_scenario(
        name="straight_line",
        heading_fn=lambda _t: 0.0,
        total_time_s=40.0,
    )
    step_turn = run_heading_scenario(
        name="step_turn_90deg_at_20s",
        heading_fn=lambda t: 0.0 if t < 20.0 else np.deg2rad(90.0),
        total_time_s=60.0,
    )
    results = [straight, step_turn]

    for result in results:
        print(
            f"{result.name}: RMSE={result.rmse_m:.3f} m, "
            f"FinalError={result.final_error_m:.3f} m"
        )

    if show_plot:
        plot_results(results)
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run heading-guidance scenarios.")
    parser.add_argument("--no-plot", action="store_true", help="Run scenarios without opening matplotlib windows.")
    args = parser.parse_args()
    main(show_plot=not args.no_plot)
