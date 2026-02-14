"""Tests shared metrics integration in identification evaluator."""

from __future__ import annotations

import unittest

import numpy as np

from minos.identification.core import FlightDataset
from minos.identification.evaluator import TrajectoryEvaluator
from minos.metrics import compute_position_error_metrics
from minos.sim.runners import bare_simulate_model


class TestIdentificationMetricsBridge(unittest.TestCase):
    def test_compute_position_error_metrics_basic(self) -> None:
        ref = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)
        meas = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 1.0]], dtype=float)
        metrics = compute_position_error_metrics(reference=ref, measured=meas)
        self.assertEqual(metrics.sample_count, 2)
        self.assertGreater(metrics.rmse_m, 0.0)
        self.assertGreaterEqual(metrics.p95_m, metrics.mae_m)

    def test_evaluator_exposes_shared_metrics(self) -> None:
        params = {"initial_pos": [0.0, 0.0, 200.0]}
        initial_conditions = [
            np.array([0.0, 0.0, 0.0], dtype=float),
            np.array([9.0, 0.0, 2.0], dtype=float),
            np.array([0.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 0.0, 0.0], dtype=float),
        ]
        time_s = np.linspace(0.0, 5.0, 51)
        flap_left = np.zeros_like(time_s)
        flap_right = np.zeros_like(time_s)
        wind = np.zeros((len(time_s), 3), dtype=float)
        sim_inputs = [flap_left, flap_right, [w.copy() for w in wind]]
        measured_positions, _ = bare_simulate_model(
            time_s, initial_conditions, sim_inputs, params, inertial=True, coefficients=None, broke_on=True
        )
        dataset = FlightDataset(
            time_s=time_s,
            flap_left=flap_left,
            flap_right=flap_right,
            wind_inertial=wind,
            measured_positions=np.asarray(measured_positions, dtype=float),
        )
        evaluator = TrajectoryEvaluator(dataset=dataset, params=params, initial_conditions=initial_conditions)
        result = evaluator.evaluate({})
        self.assertIsNotNone(result.position_metrics)
        self.assertTrue(np.isfinite(result.position_metrics.rmse_m))
        self.assertGreaterEqual(result.penalty_factor, 1.0)


if __name__ == "__main__":
    unittest.main()
