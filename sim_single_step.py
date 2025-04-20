import six_DoF_simulator as simulator
import numpy as np
from utils import rk4

def single_step(dt, old_state = None, control_input = None):
    """
    Simulate a single step of the 6-DoF simulator.

    Parameters:
    dt (float): The time step for the simulation.
    old_state (np.ndarray): The previous state of the system.
    If None, initializes to a default state.
    control_input (np.ndarray): The control input to apply.
    If None, initializes to a default input.

    Returns:
    np.ndarray: The new state after dt seconds of applying the control input.
    """
    # general stuff you can change if you want
    wind = np.array([0, 0, 0]) # is there wind?
    params = {} # mass, inertia, etc of system. dont change unless sure

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
    sim = simulator.ParafoilSimulation_6Dof(params, old_state, [control_input, wind])
    # generate new state using rk4 ode solver
    new_state = rk4(old_state, sim.get_solver_derivatives, dt)
    return new_state