"""Objective evaluation for parafoil aerodynamic coefficient identification.

This module isolates objective design from optimizer implementation. That keeps
GA code reusable and allows changing metrics/penalties without touching search
operators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from minos.identification.core import CoefficientCodec, FlightDataset
from minos.sim.runners import bare_simulate_model


@dataclass(frozen=True)
class EvaluationResult:
    """Result for one coefficient evaluation.

    Keeping diagnostics in a structured result allows experimentation with
    richer objectives later (state error, regularization, constraints).
    """

    cost: float
    simulated_positions: np.ndarray
    completed_steps: int


class TrajectoryEvaluator:
    """Evaluate coefficients by trajectory error against measured positions.

    Design notes
    ------------
    1. Uses the same simulation runner as production scripts to avoid model drift.
    2. Supports downsampled evaluation for coarse/fine multi-fidelity GA passes.
    3. Penalizes early termination to discourage numerically unstable candidates.
    """

    def __init__(
        self,
        dataset: FlightDataset,
        params: Mapping[str, object],
        initial_conditions: Sequence[np.ndarray],
        *,
        inertial: bool = True,
        break_penalty_scale: float = 1.0,
    ) -> None:
        self.dataset = dataset
        self.params = dict(params)
        self.initial_conditions = initial_conditions
        self.inertial = bool(inertial)
        self.break_penalty_scale = float(break_penalty_scale)
        # Kept as part of evaluator state to keep a single place for coefficient
        # representation concerns as objectives evolve.
        self.codec = CoefficientCodec()

    def _evaluation_slice(self, sample_stride: int) -> tuple[np.ndarray, list[object], np.ndarray]:
        """Build strided views of time, inputs, and measurements.

        Why keep the final sample
        -------------------------
        Trajectory endpoints are often highly informative; always including the
        last sample improves comparability between coarse and full evaluations.
        """
        stride = max(1, int(sample_stride))
        if stride == 1:
            return self.dataset.time_s, self.dataset.sim_inputs(), self.dataset.measured_positions

        idx = np.arange(0, len(self.dataset.time_s), stride, dtype=int)
        if idx[-1] != len(self.dataset.time_s) - 1:
            idx = np.append(idx, len(self.dataset.time_s) - 1)
        if idx.size < 2:
            idx = np.array([0, len(self.dataset.time_s) - 1], dtype=int)

        t = np.asarray(self.dataset.time_s)[idx]
        flap_l = np.asarray(self.dataset.flap_left)[idx]
        flap_r = np.asarray(self.dataset.flap_right)[idx]
        wind = np.asarray(self.dataset.wind_inertial)[idx]
        measured = np.asarray(self.dataset.measured_positions)[idx]
        sim_inputs = [flap_l, flap_r, [np.asarray(w, dtype=float) for w in wind]]
        return t, sim_inputs, measured

    def evaluate(self, coefficients: Mapping[str, float] | Sequence[float], sample_stride: int = 1) -> EvaluationResult:
        """Run simulation and compute objective cost.

        Objective currently uses position RMSE plus an early-stop penalty.
        """
        coeff_obj: Mapping[str, float] | Sequence[float]
        if isinstance(coefficients, Mapping):
            coeff_obj = coefficients
        else:
            coeff_obj = list(coefficients)

        eval_time, eval_inputs, eval_measured = self._evaluation_slice(sample_stride)
        simulated, completed = bare_simulate_model(
            eval_time,
            self.initial_conditions,
            eval_inputs,
            self.params,
            inertial=self.inertial,
            coefficients=coeff_obj,
            broke_on=True,
        )

        sim_arr = np.asarray(simulated, dtype=float)
        n = min(len(sim_arr), len(eval_measured))
        if n == 0:
            return EvaluationResult(cost=1e6, simulated_positions=np.empty((0, 3)), completed_steps=0)

        sim_cut = sim_arr[:n]
        real_cut = eval_measured[:n]
        rmse = float(np.sqrt(np.mean(np.sum((sim_cut - real_cut) ** 2, axis=1))))

        # If a simulation breaks early, missing points indicate instability or
        # physically implausible parameter sets. We scale RMSE by completion.
        missing = max(0, len(eval_time) - int(completed))
        miss_ratio = missing / len(eval_time)
        penalty = 1.0 + self.break_penalty_scale * miss_ratio
        cost = rmse * penalty

        if not np.isfinite(cost):
            cost = 1e6
        return EvaluationResult(cost=float(cost), simulated_positions=sim_arr, completed_steps=int(completed))
