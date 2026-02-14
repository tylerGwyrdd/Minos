"""Parallel PID sweep for GnC heading step-response."""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from minos.gnc.tuning import run_pid_step_response
from minos.parallel import ParallelRunner, TaskSpec


def _progress(done: int, total: int, outcome) -> None:
    if done == total or done % max(1, total // 10) == 0:
        print(f"Progress: {done}/{total} ({100.0 * done / total:.0f}%)")


def main(show_plot: bool = True, workers: int | None = None, backend: str = "process") -> None:
    # Gains are dimensionless controller coefficients for yaw tracking. The
    # sweep bounds are intentionally broad to expose unstable/overdamped regions
    # before narrowing in follow-up searches.
    kp_values = np.linspace(1.0, 8.0, 15)
    kd_values = np.linspace(0.5, 10.0, 20)
    # Timing assumptions: step sequence and controller updates both run at a
    # fixed 0.1 s cadence in `run_pid_step_response`.
    total_time_s = 40.0
    dt = 0.1

    tasks: list[TaskSpec] = []
    for kp in kp_values:
        for kd in kd_values:
            tasks.append(
                TaskSpec(
                    task_id=f"kp={kp:.4f}|kd={kd:.4f}",
                    # Coupling note: callable path must remain a top-level symbol
                    # for process backend pickling compatibility.
                    callable_path="minos.gnc.tuning:evaluate_pid_step_candidate",
                    args=(float(kp), float(kd), float(total_time_s), float(dt), 0.6),
                    kwargs={},
                )
            )

    if workers is None:
        # Leave one core for OS/UI responsiveness during large sweeps.
        workers = max(1, (os.cpu_count() or 2) - 1)
    runner = ParallelRunner(backend=backend)
    outcomes = runner.run(tasks, max_workers=workers, fail_fast=False, progress_cb=_progress)

    ok = [o for o in outcomes if o.status == "ok"]
    if not ok:
        raise RuntimeError("No successful candidates. Check errors in task outcomes.")
    best = min(ok, key=lambda o: float(o.result["score"]))
    best_kp = float(best.result["kp"])
    best_kd = float(best.result["kd"])

    print("\nBest candidate:")
    print(f"kp={best_kp:.4f}, kd={best_kd:.4f}")
    print(f"score={float(best.result['score']):.4f}")
    print(f"heading_rmse_deg={float(best.result['heading_rmse_deg']):.4f}")
    print(f"heading_mae_deg={float(best.result['heading_mae_deg']):.4f}")
    print(f"heading_p95_deg={float(best.result['heading_p95_deg']):.4f}")

    # Re-run best candidate once in-process to collect a coherent trace for
    # plotting. Worker results only return scalar metrics by design.
    trace = run_pid_step_response(best_kp, best_kd, total_time_s=total_time_s, dt=dt)
    t = np.asarray(trace["times"], dtype=float)
    desired = np.asarray(trace["desired_heading_rad"], dtype=float)
    actual = np.asarray(trace["actual_heading_rad"], dtype=float)

    plt.figure(figsize=(10, 5))
    # Heading channels are radians internally; convert to degrees for operator
    # readability when comparing desired vs actual tracks.
    plt.plot(t, np.degrees(desired), "--", linewidth=2, label="Desired heading")
    plt.plot(t, np.degrees(actual), linewidth=2, label="Actual heading")
    plt.title(f"Parallel PID Sweep Best Response (kp={best_kp:.3f}, kd={best_kd:.3f})")
    plt.xlabel("Time (s)")
    plt.ylabel("Heading (deg)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if show_plot:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel PID tuning for heading step response.")
    parser.add_argument("--no-show", action="store_true", help="Run without opening the plot window.")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes/threads.")
    parser.add_argument("--backend", choices=["process", "thread", "sequential"], default="process")
    args = parser.parse_args()
    main(show_plot=not args.no_show, workers=args.workers, backend=args.backend)
