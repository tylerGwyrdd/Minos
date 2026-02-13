"""Single-run GA aerodynamic coefficient identification experiment.

This script preserves the original Simple_GA intent while using the refactored
identification API.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from minos.identification.core import CoefficientBounds, FlightDataset
from minos.identification.deap_ga import GAConfig, optimize_coefficients_ga
from minos.identification.evaluator import TrajectoryEvaluator
from minos.identification.TSGA_utils import (
    aero_coeffs,
    bounds_dict,
    coeff_names,
    evaluate_partial_coeffs_error,
    generate_complete_flight,
    ideal_coeffs,
)
from minos.sim.runners import bare_simulate_model, sim_with_noise

OUTPUT_BEST = Path("best_coefficients.json")
OUTPUT_SHARED = Path("shared_data.npz")


PARAMS = {"initial_pos": [0.0, 0.0, 0.0]}
INITIAL_CONDITIONS = [
    np.array([0.0, 0.0, 0.0]),
    np.array([10.0, 0.0, 3.0]),
    np.array([0.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 0.0]),
]


def _build_dataset(time_vector: np.ndarray, with_noise: bool) -> FlightDataset:
    """Build a reproducible trajectory-fitting dataset for a full-vector run."""
    left, right = generate_complete_flight(time_vector)
    wind = np.zeros((time_vector.size, 3), dtype=float)
    inputs = [np.asarray(left, dtype=float), np.asarray(right, dtype=float), [w.copy() for w in wind]]

    # Named mapping avoids silent coefficient-order mismatches.
    target_coeffs = {name: ideal_coeffs[name] for name in coeff_names}
    if with_noise:
        measured_positions, _ = sim_with_noise(
            time_vector,
            INITIAL_CONDITIONS,
            inputs,
            PARAMS,
            inertial=True,
            coefficients=target_coeffs,
        )
    else:
        measured_positions, _ = bare_simulate_model(
            time_vector,
            INITIAL_CONDITIONS,
            inputs,
            PARAMS,
            inertial=True,
            coefficients=target_coeffs,
            broke_on=True,
        )

    return FlightDataset(
        time_s=time_vector,
        flap_left=np.asarray(left, dtype=float),
        flap_right=np.asarray(right, dtype=float),
        wind_inertial=wind,
        measured_positions=np.asarray(measured_positions, dtype=float),
    )


def run_simple_ga() -> dict[str, object]:
    """Run one full-coefficient GA experiment and persist key artifacts.

    `shared_data.npz` is intentionally kept for backwards compatibility with
    legacy analysis tools that still read this file.
    """
    time_vector = np.linspace(0.0, 50.0, 500)
    dataset = _build_dataset(time_vector, with_noise=True)

    np.savez(
        OUTPUT_SHARED,
        time_vector=dataset.time_s,
        inputs=np.array(dataset.sim_inputs(), dtype=object),
        real_data=dataset.measured_positions,
    )

    # Centralized bounds keep all experiments aligned to the same search limits.
    bounds = CoefficientBounds(by_name=bounds_dict)
    evaluator = TrajectoryEvaluator(
        dataset=dataset,
        params=PARAMS,
        initial_conditions=INITIAL_CONDITIONS,
        inertial=True,
        break_penalty_scale=1.0,
    )

    result = optimize_coefficients_ga(
        evaluate_coefficients=evaluator.evaluate,
        bounds=bounds,
        optimize_names=coeff_names,
        base_coefficients=aero_coeffs,
        config=GAConfig(
            population_size=20,
            generations=300,
            cxpb=0.5,
            mutpb=0.35,
            mutation_eta=10.0,
            mutation_indpb=0.25,
            elite_size=2,
            # n_jobs=0 => auto-select available CPU workers.
            n_jobs=0,
            # Coarse-first evaluation reduces early-generation compute.
            enable_multi_fidelity=True,
            coarse_stride=4,
            coarse_until_fraction=0.7,
            coarse_top_k_full=2,
            seed=7,
        ),
    )

    with OUTPUT_BEST.open("w", encoding="utf-8") as f:
        json.dump(result.best_coefficients, f, indent=4)

    final_errors = evaluate_partial_coeffs_error(
        [result.best_coefficients[name] for name in coeff_names],
        coeff_names,
    )
    summary = {
        "best_cost": result.best_cost,
        "best_coefficients": result.best_coefficients,
        "coefficient_percent_error": final_errors,
        "history": result.history,
    }
    return summary


if __name__ == "__main__":
    out = run_simple_ga()
    print("Simple_GA complete")
    print(f"Best cost: {out['best_cost']:.6f}")
