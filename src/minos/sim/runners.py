"""Canonical simulation runners for open-loop and closed-loop GnC."""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

from minos.gnc.adapters import observation_from_sim
from minos.gnc.interfaces import GuidanceVisualization
from minos.gnc.stack import GncStack
from minos.physics.model import ParafoilModel6DOF
from minos.physics.types import Inputs, State
from minos.utilities.snapshots import SimulationSnapshot


def _build_model(
    params: dict,
    initial_conditions: Iterable[np.ndarray],
    model_input: list[object],
) -> ParafoilModel6DOF:
    """Create a typed model from runner input vectors."""
    return ParafoilModel6DOF(
        params=params,
        initial_state=State.from_sequence(initial_conditions),
        initial_inputs=Inputs(
            flap_left=float(model_input[0][0]),
            flap_right=float(model_input[0][1]),
            wind_inertial=np.asarray(model_input[1], dtype=float),
        ),
    )


def _inertial_state(sim: ParafoilModel6DOF) -> list[np.ndarray]:
    """Return inertial state in historical layout."""
    return [
        sim.inertial_position.copy(),
        sim.inertial_velocity.copy(),
        sim.state.eulers.copy(),
        sim.euler_rates.copy(),
    ]


def _snapshot(
    sim: ParafoilModel6DOF,
    t: float,
    heading_pair: list[float] | None = None,
    phase: str | None = None,
    flare_magnitude: float | None = None,
    wind_estimate: np.ndarray | None = None,
    flap_command: tuple[float, float] | None = None,
    flap_command_raw: tuple[float, float] | None = None,
    guidance_visualization: GuidanceVisualization | None = None,
) -> SimulationSnapshot:
    """Build one typed log snapshot from the current simulator state."""
    diag = sim.last_diagnostics
    heading_current = None
    heading_desired = None
    if heading_pair is not None:
        heading_current = float(heading_pair[0])
        heading_desired = float(heading_pair[1])
    return SimulationSnapshot(
        time_s=t,
        state=State.from_sequence(sim.state.as_sequence()),
        inertial_position=sim.inertial_position.copy(),
        inertial_velocity=sim.inertial_velocity.copy(),
        euler_rates=sim.euler_rates.copy(),
        angle_of_attack=diag.angle_of_attack,
        sideslip_angle=diag.sideslip_angle,
        angular_acc=diag.angular_acc.copy(),
        acc=diag.acc.copy(),
        CL=diag.CL,
        CD=diag.CD,
        Cl=diag.Cl,
        Cn=diag.Cn,
        Cm=diag.Cm,
        F_aero=diag.F_aero.copy(),
        F_g=diag.F_g.copy(),
        F_fictious=diag.F_fictious.copy(),
        M_aero=diag.M_aero.copy(),
        M_f_aero=diag.M_f_aero.copy(),
        M_fictious=diag.M_fictious.copy(),
        M_total=diag.M_total.copy(),
        va=diag.va.copy(),
        wind_inertial=diag.w.copy(),
        flap_left=sim.flap_left,
        flap_right=sim.flap_right,
        heading_current=heading_current,
        heading_desired=heading_desired,
        phase=phase,
        flare_magnitude=flare_magnitude,
        wind_estimate=None if wind_estimate is None else np.asarray(wind_estimate, dtype=float),
        flap_left_command=None if flap_command is None else float(flap_command[0]),
        flap_right_command=None if flap_command is None else float(flap_command[1]),
        flap_left_command_raw=None if flap_command_raw is None else float(flap_command_raw[0]),
        flap_right_command_raw=None if flap_command_raw is None else float(flap_command_raw[1]),
        guidance_visualization=guidance_visualization,
    )


def bare_simulate_model(
    time_list,
    initial_conditions,
    inputs,
    params,
    inertial: bool = False,
    coefficients=None,
    broke_on: bool = True,
):
    """Run simulation and collect position history only."""
    positions = []
    if not broke_on:
        if inertial:
            positions = [np.array(params["initial_pos"], dtype=float) for _ in time_list]
        else:
            positions = [np.zeros(3) for _ in time_list]
    model_input = [[inputs[0][0], inputs[1][0]], inputs[2][0]]
    sim = _build_model(params, initial_conditions, model_input)
    sim.set_coefficients(coefficients)
    for i, t in enumerate(time_list):
        if inertial:
            if not broke_on:
                positions[i] = sim.inertial_position.copy()
            else:
                positions.append(sim.inertial_position.copy())
        else:
            if not broke_on:
                positions[i] = sim.state.position.copy()
            else:
                positions.append(sim.state.position.copy())

        model_input = [[inputs[0][i], inputs[1][i]], inputs[2][i]]
        sim.set_inputs(Inputs(model_input[0][0], model_input[0][1], np.asarray(model_input[1], dtype=float)))
        dt = t - time_list[i - 1] if i > 0 else time_list[1] - time_list[0]
        sim.step(dt)
        if sim.error_flag and broke_on:
            return positions, i
    return positions, len(time_list)


def sim_with_noise(time_vector, initial_conditions, inputs, params, inertial: bool = False, coefficients=None):
    """Run position simulation and add Gaussian measurement noise."""
    gps_noise_std = np.array([0.2, 0.2, 0.2])
    ideal_positions, complete_idx = bare_simulate_model(
        time_vector, initial_conditions, inputs, params, inertial, coefficients
    )
    noisy = [p + np.random.normal(0.0, gps_noise_std, size=3) for p in ideal_positions]
    return noisy, complete_idx


def sim_state_with_noise(time_vector, initial_conditions, inputs, params, inertial: bool = False, coefficients=None):
    """Run state simulation and add Gaussian noise to each state component."""
    ideal_states, broken = multi_obj_sim(time_vector, initial_conditions, inputs, params, inertial, coefficients)
    noisy_states = [
        [
            state[0] + np.random.normal(0, 0.1, size=3),
            state[1] + np.random.normal(0, 0.1, size=3),
            state[2] + np.random.normal(0, 0.1, size=3),
            state[3] + np.random.normal(0, 0.1, size=3),
        ]
        for state in ideal_states
    ]
    return noisy_states, broken


def multi_obj_sim(time_vector, initial_conditions, inputs, params, inertial: bool = False, coefficients=None):
    """Run the model and return state history."""
    states = []
    model_input = [[inputs[0][0], inputs[1][0]], inputs[2][0]]
    sim = _build_model(params, initial_conditions, model_input)
    sim.set_coefficients(coefficients)
    for i, t in enumerate(time_vector):
        if inertial:
            states.append(_inertial_state(sim))
        else:
            states.append(sim.state.as_sequence())
        model_input = [[inputs[0][i], inputs[1][i]], inputs[2][i]]
        sim.set_inputs(Inputs(model_input[0][0], model_input[0][1], np.asarray(model_input[1], dtype=float)))
        dt = t - time_vector[i - 1] if i > 0 else time_vector[1] - time_vector[0]
        sim.step(dt)
        if sim.error_flag:
            return states, i
    return states, len(time_vector)


def simulate_model(
    time_vector,
    initial_conditions,
    inputs,
    params,
    inertial: bool = False,
    coefficients=None,
):
    """Run simulation and return snapshots + state history."""
    states = []
    snapshots: list[SimulationSnapshot] = []
    model_input = [[inputs[0][0], inputs[1][0]], inputs[2][0]]
    sim = _build_model(params, initial_conditions, model_input)
    sim.set_coefficients(coefficients)

    for i, t in enumerate(time_vector):
        snapshots.append(_snapshot(sim, float(t)))
        if inertial:
            states.append(_inertial_state(sim))
        else:
            states.append(sim.state.as_sequence())
        model_input = [[inputs[0][i], inputs[1][i]], inputs[2][i]]
        sim.set_inputs(Inputs(model_input[0][0], model_input[0][1], np.asarray(model_input[1], dtype=float)))
        dt = t - time_vector[i - 1] if i > 0 else time_vector[1] - time_vector[0]
        sim.step(dt)

    return snapshots, states


def run_simulation_with_gnc(sim: ParafoilModel6DOF, steps: int, dt: float, gnc_stack: GncStack) -> list[SimulationSnapshot]:
    """Run closed-loop simulation using the modular GnC stack."""
    data: list[SimulationSnapshot] = []
    t = 0.0
    for _ in range(steps):
        obs = observation_from_sim(sim, t)
        impact_z = float(gnc_stack.mission.extras.get("IPI", np.array([0.0, 0.0, 0.0]))[2])
        if obs.inertial_position[2] < impact_z:
            logging.info("Parafoil has hit the ground, stopping simulation.")
            logging.info(f"Final Position: {[f'{coord:.3g}' for coord in sim.state.position]}")
            ipi_error = gnc_stack.mission.extras.get("IPI", np.array([0.0, 0.0, 0.0]))[:2] - obs.inertial_position[:2]
            logging.info(f"Final IPI Error: {[f'{e:.3g}' for e in ipi_error]}")
            if gnc_stack.last_nav is not None:
                est_wind = gnc_stack.last_nav.wind_inertial_estimate
                logging.info(f"Estimated Wind: {[f'{w:.3g}' for w in est_wind]}")
            logging.info(f"Actual Wind: {[f'{w:.3g}' for w in obs.wind_inertial]}")
            break

        current_heading = float(obs.state.eulers[2])
        inputs = gnc_stack.update(obs, dt)
        desired_heading = float(gnc_stack.last_guidance.desired_heading)
        phase = getattr(gnc_stack.mission, "phase", None)
        flare = float(gnc_stack.last_guidance.flare_magnitude) if gnc_stack.last_guidance is not None else None
        guidance_visualization = None if gnc_stack.last_guidance is None else gnc_stack.last_guidance.visualization
        wind_est = None if gnc_stack.last_nav is None else gnc_stack.last_nav.wind_inertial_estimate
        flap_cmd = None
        if gnc_stack.last_control is not None:
            flap_cmd = (float(gnc_stack.last_control.flap_left), float(gnc_stack.last_control.flap_right))
        flap_cmd_raw = None
        if gnc_stack.last_control_raw is not None:
            flap_cmd_raw = (float(gnc_stack.last_control_raw.flap_left), float(gnc_stack.last_control_raw.flap_right))

        sim.set_inputs(inputs)
        sim.step(dt)
        data.append(
            _snapshot(
                sim,
                t,
                heading_pair=[current_heading, desired_heading],
                phase=phase,
                flare_magnitude=flare,
                wind_estimate=wind_est,
                flap_command=flap_cmd,
                flap_command_raw=flap_cmd_raw,
                guidance_visualization=guidance_visualization,
            )
        )
        t += dt
    return data
