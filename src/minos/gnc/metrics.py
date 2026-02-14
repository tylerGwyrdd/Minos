"""Run-level and aggregate metrics for GnC benchmarking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np

from minos.utilities.snapshots import SimulationSnapshot


def _wrap_angle_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class GncRunMetrics:
    """Metrics for one scenario-method closed-loop run."""

    scenario_name: str
    method_name: str
    samples: int
    sim_time_s: float
    landed: bool
    landing_error_x_m: float
    landing_error_y_m: float
    landing_error_xy_m: float
    heading_mae_deg: float
    heading_rmse_deg: float
    heading_p95_deg: float
    control_effort_l1: float
    control_rate_l1: float
    saturation_fraction: float
    clip_events: int
    wind_rmse_xy_mps: float | None
    phase_transition_count: int
    phase_durations_s: dict[str, float] = field(default_factory=dict)
    flare_started: bool = False
    flare_start_time_s: float | None = None
    flare_start_altitude_m: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize for JSON/CSV reporting."""
        return asdict(self)


@dataclass(frozen=True)
class GncAggregateMetrics:
    """Aggregate statistics across multiple runs."""

    n_runs: int
    success_rate: float
    mean: dict[str, float]
    std: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        """Serialize aggregate report."""
        return asdict(self)


def compute_run_metrics(
    snapshots: Sequence[SimulationSnapshot],
    *,
    scenario_name: str = "",
    method_name: str = "",
    ipi: np.ndarray | None = None,
    max_flap_abs: float | None = None,
    wind_truth: np.ndarray | None = None,
) -> GncRunMetrics:
    """Compute summary metrics from a closed-loop snapshot series."""
    if not snapshots:
        return GncRunMetrics(
            scenario_name=scenario_name,
            method_name=method_name,
            samples=0,
            sim_time_s=0.0,
            landed=False,
            landing_error_x_m=float("nan"),
            landing_error_y_m=float("nan"),
            landing_error_xy_m=float("nan"),
            heading_mae_deg=float("nan"),
            heading_rmse_deg=float("nan"),
            heading_p95_deg=float("nan"),
            control_effort_l1=0.0,
            control_rate_l1=0.0,
            saturation_fraction=0.0,
            clip_events=0,
            wind_rmse_xy_mps=None,
            phase_transition_count=0,
        )

    times = np.array([s.time_s for s in snapshots], dtype=float)
    if len(times) > 1:
        dt = np.diff(times)
        dt = np.append(dt, dt[-1])
    else:
        dt = np.array([0.0], dtype=float)
    sim_time_s = float(max(times[-1] - times[0], 0.0))

    landing_error_x = float("nan")
    landing_error_y = float("nan")
    landing_error_xy = float("nan")
    landed = False
    if ipi is not None:
        ipi_vec = np.asarray(ipi, dtype=float).reshape(3)
        final_pos = snapshots[-1].inertial_position
        err_xy = ipi_vec[:2] - final_pos[:2]
        landing_error_x = float(err_xy[0])
        landing_error_y = float(err_xy[1])
        landing_error_xy = float(np.linalg.norm(err_xy))
        landed = bool(final_pos[2] <= ipi_vec[2])

    heading_pairs = np.array(
        [
            [s.heading_current, s.heading_desired]
            for s in snapshots
            if s.heading_current is not None and s.heading_desired is not None
        ],
        dtype=float,
    )
    if heading_pairs.size > 0:
        heading_err = _wrap_angle_pi(heading_pairs[:, 1] - heading_pairs[:, 0])
        heading_err_deg = np.degrees(heading_err)
        heading_mae_deg = float(np.mean(np.abs(heading_err_deg)))
        heading_rmse_deg = float(np.sqrt(np.mean(heading_err_deg**2)))
        heading_p95_deg = float(np.percentile(np.abs(heading_err_deg), 95))
    else:
        heading_mae_deg = float("nan")
        heading_rmse_deg = float("nan")
        heading_p95_deg = float("nan")

    flap_cmd = np.array(
        [
            [
                s.flap_left_command if s.flap_left_command is not None else s.flap_left,
                s.flap_right_command if s.flap_right_command is not None else s.flap_right,
            ]
            for s in snapshots
        ],
        dtype=float,
    )
    control_effort_l1 = float(np.sum((np.abs(flap_cmd[:, 0]) + np.abs(flap_cmd[:, 1])) * dt))
    if len(flap_cmd) > 1:
        diffs = np.abs(np.diff(flap_cmd, axis=0))
        control_rate_l1 = float(np.sum(diffs[:, 0] + diffs[:, 1]))
    else:
        control_rate_l1 = 0.0

    saturation_fraction = 0.0
    if max_flap_abs is not None and max_flap_abs > 0:
        at_limit = np.logical_or(
            np.abs(flap_cmd[:, 0]) >= float(max_flap_abs) - 1.0e-9,
            np.abs(flap_cmd[:, 1]) >= float(max_flap_abs) - 1.0e-9,
        )
        saturation_fraction = float(np.mean(at_limit))

    clip_events = 0
    for s in snapshots:
        if s.flap_left_command_raw is None or s.flap_right_command_raw is None:
            continue
        left_cmd = s.flap_left_command if s.flap_left_command is not None else s.flap_left
        right_cmd = s.flap_right_command if s.flap_right_command is not None else s.flap_right
        if abs(s.flap_left_command_raw - left_cmd) > 1.0e-9 or abs(s.flap_right_command_raw - right_cmd) > 1.0e-9:
            clip_events += 1

    wind_rmse_xy_mps: float | None = None
    if wind_truth is not None:
        w_truth = np.asarray(wind_truth, dtype=float).reshape(3)
        wind_est_series = np.array(
            [s.wind_estimate[:2] for s in snapshots if s.wind_estimate is not None],
            dtype=float,
        )
        if wind_est_series.size > 0:
            truth_xy = np.tile(w_truth[:2], (len(wind_est_series), 1))
            wind_rmse_xy_mps = float(np.sqrt(np.mean(np.sum((wind_est_series - truth_xy) ** 2, axis=1))))

    phases = [s.phase for s in snapshots if s.phase]
    phase_durations: dict[str, float] = {}
    for idx, s in enumerate(snapshots):
        if s.phase is None:
            continue
        phase_durations[s.phase] = phase_durations.get(s.phase, 0.0) + float(dt[idx])
    phase_transition_count = 0
    if phases:
        phase_transition_count = int(sum(1 for i in range(1, len(phases)) if phases[i] != phases[i - 1]))

    flare_started = False
    flare_start_time: float | None = None
    flare_start_altitude: float | None = None
    for s in snapshots:
        if s.flare_magnitude is not None and s.flare_magnitude > 0.0:
            flare_started = True
            flare_start_time = float(s.time_s)
            flare_start_altitude = float(s.inertial_position[2])
            break

    return GncRunMetrics(
        scenario_name=scenario_name,
        method_name=method_name,
        samples=len(snapshots),
        sim_time_s=sim_time_s,
        landed=landed,
        landing_error_x_m=landing_error_x,
        landing_error_y_m=landing_error_y,
        landing_error_xy_m=landing_error_xy,
        heading_mae_deg=heading_mae_deg,
        heading_rmse_deg=heading_rmse_deg,
        heading_p95_deg=heading_p95_deg,
        control_effort_l1=control_effort_l1,
        control_rate_l1=control_rate_l1,
        saturation_fraction=saturation_fraction,
        clip_events=clip_events,
        wind_rmse_xy_mps=wind_rmse_xy_mps,
        phase_transition_count=phase_transition_count,
        phase_durations_s=phase_durations,
        flare_started=flare_started,
        flare_start_time_s=flare_start_time,
        flare_start_altitude_m=flare_start_altitude,
    )


def aggregate_metrics(runs: Sequence[GncRunMetrics]) -> GncAggregateMetrics:
    """Compute mean/std summary across runs."""
    if not runs:
        return GncAggregateMetrics(n_runs=0, success_rate=0.0, mean={}, std={})

    keys = [
        "landing_error_xy_m",
        "heading_mae_deg",
        "heading_rmse_deg",
        "control_effort_l1",
        "control_rate_l1",
        "saturation_fraction",
        "clip_events",
        "sim_time_s",
    ]
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    for key in keys:
        vals = np.array([getattr(r, key) for r in runs], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        mean[key] = float(np.mean(vals))
        std[key] = float(np.std(vals))
    success_rate = float(np.mean([1.0 if r.landed else 0.0 for r in runs]))
    return GncAggregateMetrics(n_runs=len(runs), success_rate=success_rate, mean=mean, std=std)
