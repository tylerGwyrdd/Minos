"""High-level GA runner for position-based aerodynamic identification.

This module provides a stable entry point for external callers. Scripts can
change, but this function signature is intended to remain predictable.
"""

from __future__ import annotations

import numpy as np

from minos.identification.core import CoefficientBounds, FlightDataset
from minos.identification.deap_ga import GAConfig, GAResult, optimize_coefficients_ga
from minos.identification.evaluator import TrajectoryEvaluator


DEFAULT_BOUNDS = CoefficientBounds(
    by_name={
        "CDo": (0.0, 0.5),
        "CDa": (0.0, 0.25),
        "CD_sym": (0.0, 0.5),
        "CLo": (0.0, 0.2),
        "CLa": (0.0, 1.8),
        "CL_sym": (0.0, 0.5),
        "CYB": (-0.5, 0.0),
        "ClB": (-0.1, 0.0),
        "Clp": (-1.7, 0.0),
        "Clr": (-0.2, 0.0),
        "Cl_asym": (-0.01, 0.0),
        "Cmo": (0.0, 0.7),
        "Cma": (-1.5, 0.0),
        "Cmq": (-3.0, 0.0),
        "CnB": (-0.01, 0.0),
        "Cn_p": (-0.2, 0.0),
        "Cn_r": (-0.6, 0.0),
        "Cn_asym": (0.0, 0.03),
    }
)


def run_position_identification_ga(
    dataset: FlightDataset,
    *,
    params: dict[str, object],
    initial_conditions: list[np.ndarray],
    optimize_names: list[str] | None = None,
    base_coefficients: dict[str, float] | None = None,
    bounds: CoefficientBounds = DEFAULT_BOUNDS,
    ga_config: GAConfig = GAConfig(),
) -> GAResult:
    """Run a DEAP GA using position RMSE against measured trajectory data.

    Why this wrapper exists
    -----------------------
    It hides wiring details (dataset -> evaluator -> optimizer) so higher-level
    tools can invoke identification with minimal boilerplate.
    """
    evaluator = TrajectoryEvaluator(
        dataset=dataset,
        params=params,
        initial_conditions=initial_conditions,
        inertial=True,
        break_penalty_scale=1.0,
    )
    return optimize_coefficients_ga(
        evaluate_coefficients=evaluator.evaluate,
        bounds=bounds,
        optimize_names=optimize_names,
        base_coefficients=base_coefficients,
        config=ga_config,
    )


if __name__ == "__main__":
    from minos.sim.runners import bare_simulate_model

    params = {"initial_pos": [0.0, 0.0, 500.0]}
    initial_conditions = [
        np.array([0.0, 0.0, 0.0]),
        np.array([10.0, 0.0, 3.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
    ]
    time_s = np.linspace(0.0, 12.0, 121)
    flap_left = np.zeros_like(time_s)
    flap_right = np.zeros_like(time_s)
    flap_left[20:40] = 0.2
    flap_right[60:80] = 0.25
    wind = np.zeros((time_s.size, 3), dtype=float)
    sim_inputs = [flap_left, flap_right, [w.copy() for w in wind]]
    measured_positions, _ = bare_simulate_model(
        time_s, initial_conditions, sim_inputs, params, inertial=True, coefficients=None
    )
    # Synthetic noise is intentionally injected here so the demo resembles
    # flight-test fitting rather than perfect model recovery.
    measured_positions = np.asarray(measured_positions) + np.random.normal(0.0, 0.1, size=(time_s.size, 3))

    dataset = FlightDataset(
        time_s=time_s,
        flap_left=flap_left,
        flap_right=flap_right,
        wind_inertial=wind,
        measured_positions=measured_positions,
    )
    result = run_position_identification_ga(
        dataset,
        params=params,
        initial_conditions=initial_conditions,
        optimize_names=["CDo", "CDa", "CLo", "CLa"],
        ga_config=GAConfig(population_size=20, generations=30, seed=7),
    )
    print("Best cost:", result.best_cost)
    print("Best subset:", {k: result.best_coefficients[k] for k in ["CDo", "CDa", "CLo", "CLa"]})
