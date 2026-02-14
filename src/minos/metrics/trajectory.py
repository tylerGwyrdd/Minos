"""Generic trajectory error metrics and penalties."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PositionErrorMetrics:
    """Position-tracking error summary."""

    rmse_m: float
    mae_m: float
    p95_m: float
    final_error_m: float
    sample_count: int


def compute_position_error_metrics(reference: np.ndarray, measured: np.ndarray) -> PositionErrorMetrics:
    """Compute scalar trajectory error metrics from 3D point series.

    The two series are aligned by truncating to the shortest sequence length.
    """
    ref = np.asarray(reference, dtype=float)
    meas = np.asarray(measured, dtype=float)
    n = int(min(len(ref), len(meas)))
    if n <= 0:
        return PositionErrorMetrics(rmse_m=float("inf"), mae_m=float("inf"), p95_m=float("inf"), final_error_m=float("inf"), sample_count=0)

    err = ref[:n] - meas[:n]
    dist = np.linalg.norm(err, axis=1)
    return PositionErrorMetrics(
        rmse_m=float(np.sqrt(np.mean(dist**2))),
        mae_m=float(np.mean(np.abs(dist))),
        p95_m=float(np.percentile(dist, 95)),
        final_error_m=float(dist[-1]),
        sample_count=n,
    )


def apply_early_stop_penalty(base_cost: float, total_steps: int, completed_steps: int, scale: float) -> tuple[float, float]:
    """Scale objective cost based on incomplete runs.

    Returns ``(penalized_cost, penalty_factor)``.
    """
    total = max(1, int(total_steps))
    completed = max(0, int(completed_steps))
    missing = max(0, total - completed)
    miss_ratio = missing / total
    penalty_factor = 1.0 + float(scale) * miss_ratio
    return float(base_cost) * penalty_factor, penalty_factor
