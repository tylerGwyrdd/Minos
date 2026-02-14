"""Tune PID heading gains using scripted left/right 90-degree step maneuvers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from minos.gnc import (
    GncStack,
    GncStackConfig,
    HeadingStep,
    MissionContext,
    NavigationEstimate,
    Navigator,
    PidHeadingController,
    TimedHeadingSequenceConfig,
    TimedHeadingSequenceGuidance,
)
from minos.physics.model import ParafoilModel6DOF
from minos.physics.types import Inputs, State
from minos.sim.runners import run_simulation_with_gnc


def _wrap_angle_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class ZeroWindNavigator(Navigator):
    """Navigator that always returns zero wind (for pure heading-step tests)."""

    def update(self, observation, dt: float, mission: MissionContext) -> NavigationEstimate:  # type: ignore[override]
        del observation, dt, mission
        return NavigationEstimate(wind_inertial_estimate=np.zeros(3, dtype=float), extras={})


@dataclass(frozen=True)
class CandidateResult:
    kp: float
    kd: float
    score: float
    heading_rmse_deg: float
    heading_mae_deg: float
    heading_p95_deg: float
    snapshots: list


def run_candidate(kp: float, kd: float, total_time_s: float, dt: float) -> CandidateResult:
    steps = int(total_time_s / dt)
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

    guidance = TimedHeadingSequenceGuidance(
        TimedHeadingSequenceConfig(
            sequence=[
                HeadingStep(0.0, np.deg2rad(0.0)),
                HeadingStep(10.0, np.deg2rad(90.0)),  # left 90 deg
                HeadingStep(25.0, np.deg2rad(0.0)),   # right 90 deg (back to 0)
            ]
        )
    )
    stack = GncStack(
        navigator=ZeroWindNavigator(),
        guidance=guidance,
        controller=PidHeadingController(kp=kp, kd=kd, max_deflection=0.6),
        mission=MissionContext(phase="heading-sequence", extras={}),
        config=GncStackConfig(max_flap_abs=1.0, use_nav_wind_estimate=False),
    )

    snapshots = run_simulation_with_gnc(sim=sim, steps=steps, dt=dt, gnc_stack=stack)
    if not snapshots:
        return CandidateResult(
            kp=kp,
            kd=kd,
            score=float("inf"),
            heading_rmse_deg=float("inf"),
            heading_mae_deg=float("inf"),
            heading_p95_deg=float("inf"),
            snapshots=[],
        )

    current = np.array([s.heading_current for s in snapshots if s.heading_current is not None], dtype=float)
    desired = np.array([s.heading_desired for s in snapshots if s.heading_desired is not None], dtype=float)
    n = min(len(current), len(desired))
    if n == 0:
        return CandidateResult(
            kp=kp,
            kd=kd,
            score=float("inf"),
            heading_rmse_deg=float("inf"),
            heading_mae_deg=float("inf"),
            heading_p95_deg=float("inf"),
            snapshots=snapshots,
        )

    err_deg = np.degrees(_wrap_angle_pi(desired[:n] - current[:n]))
    rmse = float(np.sqrt(np.mean(err_deg**2)))
    mae = float(np.mean(np.abs(err_deg)))
    p95 = float(np.percentile(np.abs(err_deg), 95))

    # Score only steering behavior.
    score = 0.65 * rmse + 0.25 * mae + 0.10 * p95
    return CandidateResult(
        kp=kp,
        kd=kd,
        score=score,
        heading_rmse_deg=rmse,
        heading_mae_deg=mae,
        heading_p95_deg=p95,
        snapshots=snapshots,
    )


def plot_best(best: CandidateResult) -> None:
    t = np.array([s.time_s for s in best.snapshots], dtype=float)
    desired = np.array([s.heading_desired for s in best.snapshots], dtype=float)
    actual = np.array([s.heading_current for s in best.snapshots], dtype=float)

    plt.figure(figsize=(10, 5))
    plt.plot(t, np.degrees(desired), "--", linewidth=2, label="Desired heading")
    plt.plot(t, np.degrees(actual), linewidth=2, label="Actual heading")
    plt.title(f"Best PID Step Response (kp={best.kp:.3f}, kd={best.kd:.3f})")
    plt.xlabel("Time (s)")
    plt.ylabel("Heading (deg)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def main(show_plot: bool = True) -> None:
    dt = 0.1
    total_time_s = 40.0

    kp_values = np.linspace(1.0, 8.0, 15)
    kd_values = np.linspace(0.5, 10.0, 20)

    best: CandidateResult | None = None
    for kp in kp_values:
        for kd in kd_values:
            result = run_candidate(float(kp), float(kd), total_time_s=total_time_s, dt=dt)
            if best is None or result.score < best.score:
                best = result

    if best is None:
        raise RuntimeError("No PID candidates evaluated.")

    print("Best candidate:")
    print(f"kp={best.kp:.4f}, kd={best.kd:.4f}")
    print(f"score={best.score:.4f}")
    print(f"heading_rmse_deg={best.heading_rmse_deg:.4f}")
    print(f"heading_mae_deg={best.heading_mae_deg:.4f}")
    print(f"heading_p95_deg={best.heading_p95_deg:.4f}")

    plot_best(best)
    if show_plot:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune PID on heading step maneuvers.")
    parser.add_argument("--no-show", action="store_true", help="Run tuning without opening a plot window.")
    args = parser.parse_args()
    main(show_plot=not args.no_show)
