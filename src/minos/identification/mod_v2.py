"""Grouped coefficient identification using refactored identification APIs.

This is the modern implementation of the original mod_v2 experiment.
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
    generate_real_data,
    generate_straight_flight,
    ideal_coeffs,
)

OUTPUT_BEST = Path("modified_tsga_best_coeffs.json")
OUTPUT_METRICS = Path("modified_tsga_metrics.json")

PARAMS = {"initial_pos": [0.0, 0.0, 500.0]}
INITIAL_CONDITIONS = [
    np.array([0.0, 0.0, 0.0]),
    np.array([10.0, 0.0, 3.0]),
    np.array([0.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 0.0]),
]


GROUPED_NAMES = [
    ["CDo", "CDa", "CLo", "CLa", "Cmo", "Cma", "Cmq", "CL_sym", "CD_sym"],
]


def _build_dataset(time_vector: np.ndarray) -> FlightDataset:
    """Create dataset for the straight-flight grouped-identification scenario."""
    left, right = generate_straight_flight(time_vector)
    wind = np.zeros((time_vector.size, 3), dtype=float)
    wind_list = [w.copy() for w in wind]

    measured = generate_real_data(
        time_vector,
        left,
        right,
        wind_list,
        PARAMS,
        np.asarray(INITIAL_CONDITIONS, dtype=object),
    )

    return FlightDataset(
        time_s=time_vector,
        flap_left=np.asarray(left, dtype=float),
        flap_right=np.asarray(right, dtype=float),
        wind_inertial=wind,
        measured_positions=np.asarray(measured, dtype=float),
    )


def optimize_group(names: list[str], dataset: FlightDataset) -> dict[str, object]:
    """Optimize one coefficient group while freezing all others at baseline."""
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
        base_coefficients=aero_coeffs,
        config=GAConfig(
            population_size=20,
            generations=120,
            cxpb=0.5,
            mutpb=0.35,
            mutation_eta=10.0,
            mutation_indpb=0.25,
            elite_size=3,
            # Auto parallelism because objective evaluation dominates runtime.
            n_jobs=0,
            # Grouped runs benefit from aggressive coarse passes early on.
            enable_multi_fidelity=True,
            coarse_stride=3,
            coarse_until_fraction=0.6,
            coarse_top_k_full=2,
            seed=13,
        ),
    )

    partial_errors = evaluate_partial_coeffs_error([result.best_coefficients[n] for n in names], names)
    return {
        "best_values": {k: result.best_coefficients[k] for k in names},
        "actual_errors": [partial_errors],
        "fitness_over_time": result.history,
        "best_cost": result.best_cost,
    }


def run_mod_v2() -> tuple[dict[str, object], dict[str, object]]:
    """Run grouped-identification workflow and emit legacy-compatible outputs."""
    time_vector = np.arange(0.0, 10.1, 0.1)
    dataset = _build_dataset(time_vector)

    best_coeffs: dict[str, dict[str, float]] = {}
    metrics: dict[str, dict[str, object]] = {}

    for names in GROUPED_NAMES:
        key = " + ".join(names)
        # Each group is reported independently to match historical metrics files.
        run_data = optimize_group(names, dataset)
        best_coeffs[key] = run_data["best_values"]
        metrics[key] = {
            "fitness_over_time": run_data["fitness_over_time"],
            "actual_error": run_data["actual_errors"],
            "best_cost": run_data["best_cost"],
        }

    with OUTPUT_BEST.open("w", encoding="utf-8") as f:
        json.dump(best_coeffs, f, indent=4)

    with OUTPUT_METRICS.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    return best_coeffs, metrics


if __name__ == "__main__":
    best, _ = run_mod_v2()
    print("mod_v2 complete")
    for group, vals in best.items():
        print(group)
        for name, value in vals.items():
            print(f"  {name}: {value:.6f} (ideal: {ideal_coeffs[name]:.6f})")
