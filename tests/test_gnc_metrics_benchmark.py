"""Tests for GnC metrics and scenario benchmarking."""

from __future__ import annotations

import unittest

import numpy as np

from minos.gnc import (
    GncStack,
    GncStackConfig,
    MissionContext,
    PidHeadingController,
    RlsWindEstimator,
    ScenarioConfig,
    TApproachGuidance,
    run_benchmark_scenario,
)


def _mission_factory() -> MissionContext:
    return MissionContext(
        phase="initialising",
        extras={
            "deployment_pos": np.array([0.0, 20.0, 120.0], dtype=float),
            "final_approach_height": 60.0,
            "spirialing_radius": 15.0,
            "update_rate": 0.1,
            "wind_unit_vector": np.array([1.0, 0.0], dtype=float),
            "wind_magnitude": 0.0,
            "wind_heading": 0.0,
            "horizontal_velocity": 5.5,
            "sink_velocity": 4.9,
            "IPI": np.array([0.0, 0.0, 0.0], dtype=float),
            "flare_height": 10.0,
            "initialised": False,
            "mode": "initialising",
            "start_heading": 0.0,
            "desired_heading": 0.0,
            "FTP_centre": np.array([0.0, 0.0], dtype=float),
        },
    )


class TestGncBenchmarkMetrics(unittest.TestCase):
    def test_benchmark_produces_metrics(self) -> None:
        scenario = ScenarioConfig(
            name="unit_small",
            dt=0.1,
            steps=200,
            params={"initial_pos": np.array([0.0, 20.0, 120.0], dtype=float)},
            initial_state=[
                np.array([0.0, 0.0, 0.0], dtype=float),
                np.array([8.0, 0.0, 2.0], dtype=float),
                np.array([0.0, 0.0, 0.0], dtype=float),
                np.array([0.0, 0.0, 0.0], dtype=float),
            ],
            wind_inertial=np.array([0.5, 0.2, 0.0], dtype=float),
            mission_factory=_mission_factory,
            ipi=np.array([0.0, 0.0, 0.0], dtype=float),
            seed=1,
        )

        def _factory(mission: MissionContext) -> GncStack:
            return GncStack(
                navigator=RlsWindEstimator(lambda_=1.0e-4, delta=1.0e13),
                guidance=TApproachGuidance(),
                controller=PidHeadingController(),
                mission=mission,
                config=GncStackConfig(max_flap_abs=1.0, use_nav_wind_estimate=False),
            )

        result = run_benchmark_scenario(scenario, method_name="default_pid", stack_factory=_factory)
        self.assertGreater(len(result.snapshots), 0)
        self.assertTrue(np.isfinite(result.metrics.heading_rmse_deg) or np.isnan(result.metrics.heading_rmse_deg))
        self.assertGreaterEqual(result.metrics.control_effort_l1, 0.0)
        self.assertGreaterEqual(result.metrics.saturation_fraction, 0.0)
        self.assertLessEqual(result.metrics.saturation_fraction, 1.0)


if __name__ == "__main__":
    unittest.main()
