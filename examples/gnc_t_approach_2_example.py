"""Example: closed-loop run using the TApproachGuidance2 strategy."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from minos.gnc import (
    GncStack,
    GncStackConfig,
    MissionContext,
    PidHeadingController,
    RlsWindEstimator,
    TApproachGuidance2,
    compute_run_metrics,
)
from minos.physics.model import ParafoilModel6DOF
from minos.physics.types import Inputs, State
from minos.sim.runners import run_simulation_with_gnc
from minos.utilities.graphs import PlotConfig, PlotKey, plot_selected_parameters
from minos.utilities.visual_3D import visualize_parafoil_pose


def build_mission(deployment_pos: np.ndarray, dt: float) -> MissionContext:
    """Build mission context for TApproachGuidance2."""
    return MissionContext(
        phase="homing",
        extras={
            "deployment_pos": deployment_pos.copy(),
            "final_approach_height": 100.0,
            "flare_height": 20.0,
            "backup_height": 60.0,
            "spirialing_radius": 20.0,
            "emc_offset": 40.0,
            "reach_radius": 12.0,
            "close_ipi_radius": 20.0,
            "low_height_threshold": 90.0,
            "backup_leg_distance": 50.0,
            "update_rate": dt,
            "wind_unit_vector": np.array([1.0, 0.0], dtype=float),
            "wind_magnitude": 0.0,
            "wind_heading": 0.0,
            "horizontal_velocity": 5.9,
            "sink_velocity": 4.9,
            "IPI": np.array([0.0, 0.0, 0.0], dtype=float),
            "mode": "homing",
            "desired_heading": 0.0,
            "em_heading": None,
        },
    )


def main(show_plots: bool = True) -> None:
    dt = 0.1
    steps = 3500
    initial_state = [
        np.array([0.0, 0.0, 0.0]),
        np.array([10.0, 0.0, 3.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
    ]
    deployment_pos = np.array([0.0, 50.0, 500.0])
    wind_truth = np.array([1.0, 1.0, 0.0])
    ipi = np.array([0.0, 0.0, 0.0])

    sim = ParafoilModel6DOF(
        params={"initial_pos": deployment_pos},
        initial_state=State.from_sequence(initial_state),
        initial_inputs=Inputs(0.0, 0.0, wind_truth),
    )
    mission = build_mission(deployment_pos, dt)
    gnc_stack = GncStack(
        navigator=RlsWindEstimator(lambda_=1.0e-4, delta=1.0e13),
        guidance=TApproachGuidance2(),
        controller=PidHeadingController(6,6),
        mission=mission,
        config=GncStackConfig(max_flap_abs=1.0, use_nav_wind_estimate=False),
    )

    snapshots = run_simulation_with_gnc(sim=sim, steps=steps, dt=dt, gnc_stack=gnc_stack)

    plot_selected_parameters(
        snapshots,
        PlotConfig(
            enabled={
            PlotKey.POSITION,
            PlotKey.EULER_ANGLES,
            PlotKey.WIND_VECTOR,
            PlotKey.DEFLECTION,
            PlotKey.HEADINGS,
            PlotKey.INERTIAL_POSITION_3D,
            PlotKey.INERTIAL_POSITION_XY,
            },
            include_guidance_annotations=True,
        ),
    )

    eulers = np.array([s.state.eulers for s in snapshots])
    ned_positions = np.array([s.state.position for s in snapshots])
    winds = np.array([s.wind_inertial for s in snapshots])
    visualize_parafoil_pose(
        euler_series=eulers,
        position_series=ned_positions,
        angle_unit="rad",
        frame="NED",
        show=show_plots,
        mode_series=[s.phase for s in snapshots],
        wind_inertial_series=winds,
    )

    metrics = compute_run_metrics(
        snapshots,
        scenario_name="t_approach_2_demo",
        method_name="TApproachGuidance2",
        ipi=ipi,
        max_flap_abs=gnc_stack.config.max_flap_abs,
        wind_truth=wind_truth,
    )

    if show_plots:
        plt.show()
    plt.close("all")

    print("TApproachGuidance2 example completed successfully.")
    print(f"Steps simulated: {len(snapshots)}")
    print(f"Landing XY error: {metrics.landing_error_xy_m:.3f} m")
    print(f"Heading RMSE: {metrics.heading_rmse_deg:.3f} deg")
    print(f"Control effort L1: {metrics.control_effort_l1:.3f}")
    if metrics.wind_rmse_xy_mps is not None:
        print(f"Wind estimate RMSE XY: {metrics.wind_rmse_xy_mps:.3f} m/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TApproachGuidance2 closed-loop example.")
    parser.add_argument("--no-show", action="store_true", help="Run without opening GUI windows.")
    args = parser.parse_args()
    main(show_plots=not args.no_show)
