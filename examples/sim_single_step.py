from minos.physics.model import ParafoilModel6DOF
from minos.physics.types import Inputs, State
import numpy as np

def single_step(dt, params, old_state = None, control_input = None, wind = None):
    """
    Simulate one RK4 step of the typed 6-DoF parafoil model.

    Parameters
    ----------
    dt : float
        Integration step in seconds.
    params : dictionary
        Physical parameter overrides for :class:`ParafoilModel6DOF`.
        Typically includes ``initial_pos`` in world frame.
    old_state : list[np.ndarray] | None
        Previous state in sequence form
        ``[position, velocity_body, eulers, angular_velocity]``.
        If ``None``, a nominal default state is used.
    control_input : list
        Left/right flap deflection command ``[left, right]`` in radians.
        If ``None``, zero deflection is applied.
    wind : np.ndarray | None
        Inertial wind vector ``[wx, wy, wz]`` in m/s.
        If ``None``, zero wind is used.
    
    Returns
    -------
    new_state : list
        Updated state sequence in model coordinates.
    inertial_state : list
        Inertial-state sequence
        ``[position_inertial, velocity_inertial, eulers, euler_rates]``.
    """
    # general stuff you can change if you want
    if wind is None:
        wind = np.array([0, 0, 0])

    # have we got a state to start from?
    if old_state is None:
        # Initialize the state if not provided
        old_state = [np.array([0, 0, 0]), 
                     np.array([10, 0, 3]), 
                     np.radians(np.array([0, 0, 0])),
                     np.array([0, 0, 0])]
    # got control input?
    if control_input is None:
        # no deflections, just a straight line
        control_input = np.array([0, 0])
    # set up instance of simulator
    sim = ParafoilModel6DOF(
        params=params,
        initial_state=State.from_sequence(old_state),
        initial_inputs=Inputs(control_input[0], control_input[1], wind),
    )
    new_state = sim.step(dt)
    inertial_state = [
        sim.inertial_position.copy(),
        sim.inertial_velocity.copy(),
        new_state.eulers.copy(),
        sim.euler_rates.copy(),
    ]
    return new_state.as_sequence(), inertial_state
