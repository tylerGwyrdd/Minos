"""Example: identification evaluator metrics using shared trajectory metrics."""

from __future__ import annotations

import numpy as np

from minos.identification.core import FlightDataset
from minos.identification.evaluator import TrajectoryEvaluator
from minos.sim.runners import bare_simulate_model


def _build_dataset() -> tuple[FlightDataset, dict[str, object], list[np.ndarray]]:
    params = {"initial_pos": [0.0, 0.0, 400.0]}
    initial_conditions = [
        np.array([0.0, 0.0, 0.0], dtype=float),
        np.array([10.0, 0.0, 3.0], dtype=float),
        np.array([0.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 0.0, 0.0], dtype=float),
    ]
    time_s = np.linspace(0.0, 10.0, 101)
    flap_left = np.zeros_like(time_s)
    flap_right = np.zeros_like(time_s)
    flap_left[15:30] = 0.2
    flap_right[60:80] = 0.25
    wind = np.zeros((len(time_s), 3), dtype=float)
    sim_inputs = [flap_left, flap_right, [w.copy() for w in wind]]
    measured_positions, _ = bare_simulate_model(
        time_s,
        initial_conditions,
        sim_inputs,
        params,
        inertial=True,
        coefficients=None,
        broke_on=True,
    )
    measured_positions = np.asarray(measured_positions, dtype=float)
    dataset = FlightDataset(
        time_s=time_s,
        flap_left=flap_left,
        flap_right=flap_right,
        wind_inertial=wind,
        measured_positions=measured_positions,
    )
    return dataset, params, initial_conditions


def main() -> None:
    dataset, params, initial_conditions = _build_dataset()
    evaluator = TrajectoryEvaluator(
        dataset=dataset,
        params=params,
        initial_conditions=initial_conditions,
        inertial=True,
        break_penalty_scale=1.0,
    )

    baseline = evaluator.evaluate({}, sample_stride=1)
    perturbed = evaluator.evaluate({"CDo": 0.4, "CLa": 1.2, "Cmq": -2.0}, sample_stride=1)

    print("Baseline coefficients:")
    print(
        f"cost={baseline.cost:.6f}, rmse={baseline.position_metrics.rmse_m:.6f}, "
        f"p95={baseline.position_metrics.p95_m:.6f}, penalty={baseline.penalty_factor:.3f}"
    )
    print("Perturbed coefficients:")
    print(
        f"cost={perturbed.cost:.6f}, rmse={perturbed.position_metrics.rmse_m:.6f}, "
        f"p95={perturbed.position_metrics.p95_m:.6f}, penalty={perturbed.penalty_factor:.3f}"
    )


if __name__ == "__main__":
    main()
