"""Stepwise grouped GA coefficient identification.

This replaces the legacy TSGA_modified script with the refactored
identification stack.
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
    evaluate_partial_coeffs_error,
    generate_complete_flight,
    ideal_coeffs,
)
from minos.sim.runners import sim_with_noise

OUTPUT_BEST = Path("tsga_step1_best.json")
OUTPUT_METRICS = Path("tsga_step1_metrics.json")

PARAMS = {"initial_pos": [0.0, 0.0, 0.0]}
INITIAL_CONDITIONS = [
    np.array([0.0, 0.0, 0.0]),
    np.array([10.0, 0.0, 3.0]),
    np.array([0.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 0.0]),
]

GROUPED_NAMES = [
    ["CDo", "CDa", "CYB", "CLo", "CLa"],
    ["CD_sym", "CL_sym", "Cl_asym", "Cn_asym"],
    ["Cmo", "Cma", "Cmq"],
    ["ClB", "Clp", "Clr"],
    ["CnB", "Cn_p", "Cn_r"],
]


def _build_dataset(time_vector: np.ndarray) -> FlightDataset:
    """Create a noisy long-horizon dataset for staged grouped fitting."""
    left, right = generate_complete_flight(time_vector)
    wind = np.zeros((time_vector.size, 3), dtype=float)
    inputs = [np.asarray(left, dtype=float), np.asarray(right, dtype=float), [w.copy() for w in wind]]

    measured, _ = sim_with_noise(
        time_vector,
        INITIAL_CONDITIONS,
        inputs,
        PARAMS,
        inertial=True,
        coefficients=ideal_coeffs,
    )

    return FlightDataset(
        time_s=time_vector,
        flap_left=np.asarray(left, dtype=float),
        flap_right=np.asarray(right, dtype=float),
        wind_inertial=wind,
        measured_positions=np.asarray(measured, dtype=float),
    )


def optimize_group(
    names: list[str],
    dataset: FlightDataset,
    base_coefficients: dict[str, float],
    seed: int,
) -> dict[str, object]:
    """Optimize one group using current best values as fixed context."""
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
        optimize_names=names,
        base_coefficients=base_coefficients,
        config=GAConfig(
            population_size=20,
            generations=100,
            cxpb=0.5,
            mutpb=0.3,
            mutation_eta=10.0,
            mutation_indpb=0.25,
            elite_size=2,
            # Auto worker count keeps script portable between machines.
            n_jobs=0,
            # Coarse-to-fine minimizes wasted high-fidelity evaluations.
            enable_multi_fidelity=True,
            coarse_stride=3,
            coarse_until_fraction=0.6,
            coarse_top_k_full=2,
            seed=seed,
        ),
    )

    return {
        "best_values": {k: result.best_coefficients[k] for k in names},
        "fitness_over_time": result.history,
        "actual_error": [evaluate_partial_coeffs_error([result.best_coefficients[n] for n in names], names)],
        "best_cost": result.best_cost,
        "full_coefficients": result.best_coefficients,
    }


def run_tsga_modified() -> tuple[dict[str, object], dict[str, object]]:
    """Run sequential grouped optimization and update running coefficient state."""
    time_vector = np.linspace(0.0, 50.0, 500)
    dataset = _build_dataset(time_vector)

    running_coeffs = dict(aero_coeffs)
    best_coeffs: dict[str, dict[str, float]] = {}
    metrics: dict[str, dict[str, object]] = {}

    for idx, names in enumerate(GROUPED_NAMES):
        group_key = " + ".join(names)
        result = optimize_group(names, dataset, running_coeffs, seed=100 + idx)

        # Later groups see improvements discovered in earlier groups.
        for name in names:
            running_coeffs[name] = float(result["best_values"][name])

        best_coeffs[group_key] = result["best_values"]
        metrics[group_key] = {
            "fitness_over_time": result["fitness_over_time"],
            "time_seconds": None,
            "actual_error": result["actual_error"],
            "best_cost": result["best_cost"],
        }

    with OUTPUT_BEST.open("w", encoding="utf-8") as f:
        json.dump(best_coeffs, f, indent=4)

    with OUTPUT_METRICS.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    return best_coeffs, metrics


if __name__ == "__main__":
    best, _ = run_tsga_modified()
    print("TSGA_modified complete")
    for group, vals in best.items():
        print(group)
        for name, value in vals.items():
            print(f"  {name}: {value:.6f} (ideal: {ideal_coeffs[name]:.6f})")
