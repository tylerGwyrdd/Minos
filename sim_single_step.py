import six_DoF_simulator as simulator
import numpy as np
from utils import rk4

def single_step(dt, params, old_state = None, control_input = None, wind = None):
    """
    Simulate a single step of the 6-DoF simulator.

    Parameters:
    dt (float): The time step for the simulation.
    params (python dictionary): Sets up all the masses etc of the system. Main thing to note:
    params['initial_pos'] = np.array([x,y,z]) - MUST BE SET TO THE POSITION WHERE PARAFOIL DEPLOYED IN WORLD FRAME e.g [100,200,500]
    old_state (np.ndarray): The previous state of the system (BODY FRAME).
    control_input list: [L-deflection, R-deflection] - The control input to apply.
    If None, assumes there is no control input
    wind (np.darray): 1x3 array of wind velocities [x_wind, y_wind, z_wind]
    If none, assumes no wind
    
    Returns:
    new_state: the new state in the BODY FRAME
    inertial_state: the new state in the WORLD FRAME!
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
    sim = simulator.ParafoilSimulation_6Dof(params, old_state, [control_input, wind])
    # generate new state using rk4 ode solver
    new_state = rk4(old_state, sim.get_solver_derivatives, dt)
    sim.set_state(new_state)
    inertial_state = sim.get_inertial_state()
    return new_state, inertial_state