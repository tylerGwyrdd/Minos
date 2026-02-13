"""Example: modular GnC closed-loop run with plots + 3D animation.

This example mirrors `physics_model_3d_example.py`, but instead of feeding
pre-built flap schedules, it runs the new modular GnC stack:

    Observation -> Navigator -> Guidance -> Controller -> Physics Inputs

Run from repo root:
    python examples/gnc_3d_example.py

Or without opening GUI windows (useful for quick smoke checks):
    python examples/gnc_3d_example.py --no-show
"""

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
    TApproachGuidance,
)
from minos.physics.model import ParafoilModel6DOF
from minos.physics.types import Inputs, State
from minos.sim.runners import run_simulation_with_gnc
from minos.utilities.graphs import PlotKey, plot_selected_parameters
from minos.utilities.visual_3D import visualize_parafoil_pose


def build_default_mission(deployment_pos: np.ndarray, dt: float) -> MissionContext:
    """Create the mission dictionary used by the T-approach guidance law.

    Notes:
    - These keys map directly to the parameters consumed by the guidance
      strategy.
    - Keeping this as a single function makes strategy comparison easy:
      duplicate this function and change only a few fields.
    """
    guidance_params = {
        # Inertial deployment point (world frame).
        "deployment_pos": deployment_pos.copy(),
        # Altitude where final approach should begin.
        "final_approach_height": 100.0,
        # Radius used during spiral/energy-management phases.
        "spirialing_radius": 20.0,
        # Guidance update rate (same as sim step in this example).
        "update_rate": dt,
        # Wind estimate state (initially unknown).
        "wind_unit_vector": np.array([1.0, 0.0]),
        "wind_magnitude": 0.0,
        "wind_v_list": [],
        # Approximate kinematic values used in guidance geometry.
        "horizontal_velocity": 5.9,
        "sink_velocity": 4.9,
        # Desired impact point indicator (ground target).
        "IPI": np.array([0.0, 0.0, 0.0]),
        # Height to begin flare command.
        "flare_height": 20.0,
        # Internal state flags for the phase machine.
        "initialised": False,
        "mode": "initialising",
        "start_heading": 0.0,
        "desired_heading": 0.0,
        "FTP_centre": np.array([0.0, 0.0]),
    }
    return MissionContext(phase="initialising", extras=guidance_params)


def main(show_plots: bool = True) -> None:
    # ------------------------------
    # 1) Simulation timing settings
    # ------------------------------
    # Closed-loop fixed-step simulation.
    dt = 0.1
    steps = 3500

    # ---------------------------------------------------------
    # 2) Physics model setup (plant) and initial truth-wind
    # ---------------------------------------------------------
    # Physics model state format:
    # [position_NED_local, velocity_body, eulers(rad), omega_body]
    initial_state = [
        np.array([0.0, 0.0, 0.0]),
        np.array([10.0, 0.0, 3.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
    ]
    # Inertial deployment position (world coordinates).
    deployment_pos = np.array([0.0, 50.0, 500.0])
    # Truth wind used by the plant (m/s). The navigator estimates this.
    wind_truth = np.array([1.0, 1.0, 0.0])

    sim = ParafoilModel6DOF(
        params={"initial_pos": deployment_pos},
        initial_state=State.from_sequence(initial_state),
        initial_inputs=Inputs(0.0, 0.0, wind_truth),
    )

    # ---------------------------------------
    # 3) Build mission state and GnC modules
    # ---------------------------------------
    # Mission context holds mutable phase/config state for guidance.
    mission = build_default_mission(deployment_pos=deployment_pos, dt=dt)
    # GnC stack wiring:
    # - Navigator: estimates wind from observed inertial velocity.
    # - Guidance: computes desired heading from mission geometry.
    # - Controller: converts heading command to left/right flap deflections.
    gnc_stack = GncStack(
        navigator=RlsWindEstimator(lambda_=0.0001, delta=10000000000000),
        guidance=TApproachGuidance(),
        controller=PidHeadingController(),
        mission=mission,
        config=GncStackConfig(
            max_flap_abs=1.0,
            # Keep this False so plant uses truth wind, not estimated wind.
            # This preserves realistic estimation error in closed-loop tests.
            use_nav_wind_estimate=False,
        ),
    )

    # -------------------------------------------
    # 4) Run modular closed-loop GnC simulation
    # -------------------------------------------
    # Each loop step inside run_simulation_with_gnc does:
    #   observation_from_sim(sim) -> gnc_stack.update(obs, dt) -> sim.set_inputs -> sim.step
    snapshots = run_simulation_with_gnc(sim=sim, steps=steps, dt=dt, gnc_stack=gnc_stack)

    # -------------------------------------------------
    # 5) Generate standard diagnostic time-series plots
    # -------------------------------------------------
    plot_selected_parameters(
        snapshots,
        {
            PlotKey.POSITION,
            PlotKey.EULER_ANGLES,
            PlotKey.WIND_VECTOR,
            PlotKey.DEFLECTION,
            PlotKey.HEADINGS,
            PlotKey.INERTIAL_POSITION_3D,
        },
    )

    # ---------------------------------------------
    # 6) Build channels for interactive 3D animation
    # ---------------------------------------------
    eulers = np.array([s.state.eulers for s in snapshots])
    # Viewer expects position series aligned with euler samples.
    # Here we use model-local NED position from state.
    ned_positions = np.array([s.state.position for s in snapshots])

    # Show 3D motion viewer:
    # - angle_unit="rad": state eulers are in radians.
    # - frame="NED": keep same frame convention as model internals.
    visualize_parafoil_pose(
        euler_series=eulers,
        position_series=ned_positions,
        angle_unit="rad",
        frame="NED",
        show=show_plots,
    )

    # ----------------------------------------------------
    # 7) Optional matplotlib window display + text summary
    # ----------------------------------------------------
    if show_plots:
        plt.show()
    plt.close("all")

    if snapshots:
        final = snapshots[-1]
        ipi = mission.extras["IPI"]
        final_error_xy = ipi[:2] - final.inertial_position[:2]
        print("GnC example completed successfully.")
        print(f"Steps simulated: {len(snapshots)}")
        print(f"Final inertial position: {final.inertial_position}")
        print(f"Final IPI XY error: {final_error_xy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run modular GnC + 3D example.")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Run simulation and plotting logic without opening GUI windows.",
    )
    args = parser.parse_args()
    main(show_plots=not args.no_show)
