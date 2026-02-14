"""Reusable tuning helpers for heading step-response experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from minos.gnc.control.pid_heading import PidHeadingController
from minos.gnc.guidance.timed_heading_sequence import HeadingStep, TimedHeadingSequenceConfig, TimedHeadingSequenceGuidance
from minos.gnc.interfaces import MissionContext, NavigationEstimate, Navigator
from minos.gnc.stack import GncStack, GncStackConfig
from minos.physics.model import ParafoilModel6DOF
from minos.physics.types import Inputs, State
from minos.sim.runners import run_simulation_with_gnc


def _wrap_angle_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class ZeroWindNavigator(Navigator):
    """Navigator that always returns zero wind."""

    def update(self, observation, dt: float, mission: MissionContext) -> NavigationEstimate:  # type: ignore[override]
        del observation, dt, mission
        return NavigationEstimate(wind_inertial_estimate=np.zeros(3, dtype=float), extras={})


@dataclass(frozen=True)
class StepResponseMetrics:
    """Scalar step-response metrics used for candidate ranking."""

    score: float
    heading_rmse_deg: float
    heading_mae_deg: float
    heading_p95_deg: float


def run_pid_step_response(
    kp: float,
    kd: float,
    *,
    total_time_s: float = 40.0,
    dt: float = 0.1,
    max_deflection: float = 0.6,
    steps: list[tuple[float, float]] | None = None,
) -> dict[str, object]:
    """Run one zero-wind heading-step simulation and return traces + metrics."""
    # Step definitions are (time_s, heading_deg). Guidance converts to radians.
    # The test is intentionally wind-free so the score reflects controller
    # steering dynamics rather than navigation/wind-estimation quality.
    step_defs = steps if steps is not None else [(0.0, 0.0), (10.0, 90.0), (25.0, 0.0)]
    step_cfg = [HeadingStep(float(t), np.deg2rad(float(deg))) for t, deg in step_defs]
    n_steps = int(total_time_s / dt)

    # Initial position uses world frame offset (`initial_pos`) while state
    # position is model-local NED. This mirrors the main simulation conventions.
    sim = ParafoilModel6DOF(
        params={"initial_pos": np.array([0.0, 0.0, 600.0], dtype=float)},
        initial_state=State.from_sequence(
            [
                np.array([0.0, 0.0, 0.0], dtype=float),
                np.array([10.0, 0.0, 3.0], dtype=float),
                np.array([0.0, 0.0, 0.0], dtype=float),
                np.array([0.0, 0.0, 0.0], dtype=float),
            ]
        ),
        initial_inputs=Inputs(0.0, 0.0, np.zeros(3, dtype=float)),
    )
    stack = GncStack(
        navigator=ZeroWindNavigator(),
        guidance=TimedHeadingSequenceGuidance(TimedHeadingSequenceConfig(sequence=step_cfg)),
        controller=PidHeadingController(kp=float(kp), kd=float(kd), max_deflection=float(max_deflection)),
        mission=MissionContext(phase="heading-sequence", extras={}),
        config=GncStackConfig(max_flap_abs=1.0, use_nav_wind_estimate=False),
    )
    # Timing assumption: controller/guidance update period equals integrator
    # step (`dt`) in this benchmark.
    snapshots = run_simulation_with_gnc(sim=sim, steps=n_steps, dt=float(dt), gnc_stack=stack)
    if not snapshots:
        inf = float("inf")
        return {
            "score": inf,
            "heading_rmse_deg": inf,
            "heading_mae_deg": inf,
            "heading_p95_deg": inf,
            "times": np.array([], dtype=float),
            "desired_heading_rad": np.array([], dtype=float),
            "actual_heading_rad": np.array([], dtype=float),
        }

    times = np.array([s.time_s for s in snapshots], dtype=float)
    desired = np.array([s.heading_desired for s in snapshots], dtype=float)
    actual = np.array([s.heading_current for s in snapshots], dtype=float)
    n = int(min(len(desired), len(actual)))
    err_deg = np.degrees(_wrap_angle_pi(desired[:n] - actual[:n]))

    rmse = float(np.sqrt(np.mean(err_deg**2)))
    mae = float(np.mean(np.abs(err_deg)))
    p95 = float(np.percentile(np.abs(err_deg), 95))
    # Weighted score emphasizes RMS tracking while still penalizing persistent
    # bias (MAE) and large transient excursions (P95).
    score = 0.65 * rmse + 0.25 * mae + 0.10 * p95
    return {
        "score": float(score),
        "heading_rmse_deg": rmse,
        "heading_mae_deg": mae,
        "heading_p95_deg": p95,
        "times": times,
        "desired_heading_rad": desired,
        "actual_heading_rad": actual,
    }


def evaluate_pid_step_candidate(
    kp: float,
    kd: float,
    total_time_s: float = 40.0,
    dt: float = 0.1,
    max_deflection: float = 0.6,
) -> dict[str, float]:
    """Process-safe top-level function for parallel candidate evaluation."""
    out = run_pid_step_response(
        kp=float(kp),
        kd=float(kd),
        total_time_s=float(total_time_s),
        dt=float(dt),
        max_deflection=float(max_deflection),
    )
    return {
        "kp": float(kp),
        "kd": float(kd),
        "score": float(out["score"]),
        "heading_rmse_deg": float(out["heading_rmse_deg"]),
        "heading_mae_deg": float(out["heading_mae_deg"]),
        "heading_p95_deg": float(out["heading_p95_deg"]),
    }
