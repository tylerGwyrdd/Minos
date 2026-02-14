"""Example: scenario-based GnC metrics and method comparison."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np

from minos.gnc import (
    GncStack,
    GncStackConfig,
    MissionContext,
    PidHeadingController,
    RlsWindEstimator,
    ScenarioConfig,
    TApproachGuidance,
    run_benchmark_suite,
)


def _mission_factory() -> MissionContext:
    return MissionContext(
        phase="initialising",
        extras={
            "deployment_pos": np.array([0.0, 40.0, 500.0], dtype=float),
            "final_approach_height": 100.0,
            "spirialing_radius": 20.0,
            "update_rate": 0.1,
            "wind_unit_vector": np.array([1.0, 0.0], dtype=float),
            "wind_magnitude": 0.0,
            "wind_heading": 0.0,
            "wind_v_list": [],
            "horizontal_velocity": 5.9,
            "sink_velocity": 4.9,
            "IPI": np.array([0.0, 0.0, 0.0], dtype=float),
            "flare_height": 20.0,
            "initialised": False,
            "mode": "initialising",
            "start_heading": 0.0,
            "desired_heading": 0.0,
            "FTP_centre": np.array([0.0, 0.0], dtype=float),
        },
    )


def _scenario() -> ScenarioConfig:
    return ScenarioConfig(
        name="baseline_wind_1_1",
        dt=0.1,
        steps=2500,
        params={"initial_pos": np.array([0.0, 40.0, 500.0], dtype=float)},
        initial_state=[
            np.array([0.0, 0.0, 0.0], dtype=float),
            np.array([10.0, 0.0, 3.0], dtype=float),
            np.array([0.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 0.0, 0.0], dtype=float),
        ],
        wind_inertial=np.array([1.0, 1.0, 0.0], dtype=float),
        mission_factory=_mission_factory,
        ipi=np.array([0.0, 0.0, 0.0], dtype=float),
        seed=7,
    )


def _stack_default(mission: MissionContext) -> GncStack:
    return GncStack(
        navigator=RlsWindEstimator(lambda_=1.0e-4, delta=1.0e13),
        guidance=TApproachGuidance(),
        controller=PidHeadingController(kp=3.0, kd=4.0, max_deflection=0.6),
        mission=mission,
        config=GncStackConfig(max_flap_abs=1.0, use_nav_wind_estimate=False),
    )


def _stack_tighter_heading(mission: MissionContext) -> GncStack:
    return GncStack(
        navigator=RlsWindEstimator(lambda_=1.0e-4, delta=1.0e13),
        guidance=TApproachGuidance(),
        controller=PidHeadingController(kp=4.0, kd=5.0, max_deflection=0.6),
        mission=mission,
        config=GncStackConfig(max_flap_abs=1.0, use_nav_wind_estimate=False),
    )


def main() -> None:
    scenarios = [_scenario()]
    methods = [
        ("pid_default", _stack_default),
        ("pid_tighter", _stack_tighter_heading),
    ]
    results, aggregates = run_benchmark_suite(scenarios, methods=methods)

    print("Per-run metrics:")
    for run in results:
        print(
            f"{run.method_name}/{run.scenario_name}: "
            f"landing_xy={run.metrics.landing_error_xy_m:.3f} m, "
            f"heading_rmse={run.metrics.heading_rmse_deg:.3f} deg, "
            f"control_l1={run.metrics.control_effort_l1:.3f}, "
            f"wind_rmse_xy={run.metrics.wind_rmse_xy_mps if run.metrics.wind_rmse_xy_mps is not None else float('nan'):.3f}"
        )

    print("\nAggregate metrics:")
    for method_name, agg in aggregates.items():
        print(f"{method_name}: success_rate={agg.success_rate:.2f}, means={agg.mean}")

    report = {
        "runs": [run.metrics.to_dict() for run in results],
        "aggregate": {k: asdict(v) for k, v in aggregates.items()},
    }
    with open("gnc_metrics_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\nWrote gnc_metrics_report.json")


if __name__ == "__main__":
    main()
