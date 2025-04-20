from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R
import matplotlib.animation as animation
import six_DoF_simulator as simulator
from matplotlib.widgets import Slider, Button
import logging
from utils import visualize_parafoil_pose
from utils import rk4 
from guidance import T_approach
from guidance import Control

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(message)s')

@dataclass
class State:
    pos: np.ndarray
    vel: np.ndarray
    eulers: np.ndarray
    omega: np.ndarray


def run_simulation(sim, control, guidance, steps, dt):
    data = []
    t = 0
    for i in range(steps):
        state = sim.get_state()

        # Sim only calculates the derivatives, 
        # uses rk4 to work out how much derivititve to add - its is pretty good
        new_state = rk4(state, sim.get_solver_derivatives, dt)
        # update sim
        sim.set_state(new_state)

        # update inputs
        current_heading = sim.eulers[2] # assumption B is minimal
        desired_heading,_ = guidance.update(sim.get_inertial_state())
        new_inputs = control.simple_heading([sim.flap_l, sim.flap_r], desired_heading, dt)
        
        # save the current data
        data.append([
            t, state, sim.angle_of_attack, sim.sideslip_angle,
            sim.angular_acc, sim.acc, sim.CL, sim.CD, sim.Cl, sim.Cn, sim.Cm,
            sim.F_aero, sim.F_g, sim.F_fictious, sim.M_aero, sim.M_f_aero, 
            sim.M_fictious, sim.va, sim.w, [sim.flap_l, sim.flap_r], new_inputs,
            [current_heading, desired_heading]
        ])

        # update the control inputs
        sim.set_inputs([new_inputs, sim.wind])

        # calculate the derivatives for the next step
        sim.calculate_derivatives()
        # update the time
        t += dt
        # Log every 10 steps
        if i % 10 == 0:
            logging.info(f"t = {t:.3f}")
            logging.info("  Position: %s", state[0])
            logging.info("  Velocity (body): %s", state[1])
            logging.info("  Euler angles (deg): %s", np.degrees(state[2]))
            logging.info("  Angular velocity (deg/s): %s", np.degrees(state[3]))
    return data

def plot_state_over_time(data, labels, title, ylabel, times):
    plt.figure()
    if data.ndim == 1:
        data = data[:, np.newaxis]
    for i in range(data.shape[1]):
        plt.plot(times, data[:, i], label=labels[i])
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

def plot_force_moment_components_shared(times, data, field_indices, field_names, title_prefix, ylabel):
    colors = ['tab:red', 'tab:blue', 'tab:green']
    linestyles = ['-', '--', ':']
    components = ['X', 'Y', 'Z']

    plt.figure()
    for idx, name in zip(field_indices, field_names):
        vec_series = np.array([entry[idx] for entry in data])
        for i in range(3):
            label = f"{name} {components[i]}"
            plt.plot(times, vec_series[:, i], label=label, linestyle=linestyles[i], color=colors[field_indices.index(idx)])
    plt.title(f"{title_prefix} Components (All-in-One)")
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

def plot_selected_parameters(data, plots_to_show):
    times = [entry[0] for entry in data]

    parameters = {
        'Position': (np.array([entry[1][0] for entry in data]), ['X', 'Y', 'Z'], 'Position (m)'),
        'Velocity': (np.array([entry[1][1] for entry in data]), ['u', 'v', 'w'], 'Velocity (m/s)'),
        'Acceleration': (np.array([entry[5] for entry in data]), ['u', 'v', 'w'], 'Acceleration (m/s²)'),
        'Euler Angles': (np.degrees(np.array([entry[1][2] for entry in data])), ['Roll (ϕ)', 'Pitch (θ)', 'Yaw (ψ)'], 'Angle (deg)'),
        'Angular Velocity': (np.degrees(np.array([entry[1][3] for entry in data])), ['p', 'q', 'r'], 'Angular rate (deg/s)'),
        'Angular Acceleration': (np.degrees(np.array([entry[4] for entry in data])), ['p', 'q', 'r'], 'Angular rate (deg/s²)'),
        'Angle of Attack': (np.degrees(np.array([entry[2] for entry in data])), ['AoA'], 'Angle (deg)'),
        'Sideslip Angle': (np.degrees(np.array([entry[3] for entry in data])), ['β'], 'Angle (deg)'),
        'Force Coefficients': (np.column_stack(([entry[6] for entry in data], [entry[7] for entry in data])), ['CL', 'CD'], 'Coefficient'),
        'Moment Coefficients': (np.column_stack(([entry[8] for entry in data], [entry[9] for entry in data], [entry[10] for entry in data])), ['Cl', 'Cm', 'Cn'], 'Coefficient'),
        'Forces': (np.column_stack((
            [np.linalg.norm(entry[11]) for entry in data],
            [np.linalg.norm(entry[12]) for entry in data],
            [np.linalg.norm(entry[13]) for entry in data]
        )), ['F_aero', 'F_g', 'F_fictious'], 'Force (N)'),
        'Moments': (np.column_stack((
            [np.linalg.norm(entry[14]) for entry in data],
            [np.linalg.norm(entry[15]) for entry in data],
            [np.linalg.norm(entry[16]) for entry in data]
        )), ['M_aero', 'M_f_aero', 'M_fictious'], 'Moment (Nm)'),
        'Airspeed Vector': (np.array([entry[17] for entry in data]), ['Vx', 'Vy', 'Vz'], 'Airspeed (m/s)'),
        'Wind Vector': (np.array([entry[18] for entry in data]), ['Wind X', 'Wind Y', 'Wind Z'], 'Wind Velocity (m/s)'),
        'Deflection': (np.array([entry[19] for entry in data]), ['Flap Left', 'Flap Right'], 'Deflection (rad)')
    }

    for title, (data_array, labels, ylabel) in parameters.items():
        if plots_to_show.get(title, False):
            plot_state_over_time(data_array, labels, f"{title} vs Time", ylabel, times)

    if plots_to_show.get('Forces', False):
        plot_force_moment_components_shared(times, data, [11, 12, 13], ['F_aero', 'F_g', 'F_fictious'], "Force", 'Force (N)')

    if plots_to_show.get('Moments', False):
        plot_force_moment_components_shared(times, data, [14, 15, 16], ['M_aero', 'M_f_aero', 'M_fictious'], "Moment", 'Moment (Nm)')

def main():
    # first we need an initial state to base the sim off
    # Note sim uses NED Coordinate system, so Z is up
    init_state = State(
        pos=np.array([0, 0, 0]),
        vel=np.array([10, 0, 3]), # this is stable
        eulers=np.radians(np.array([0, 0, 0])),
        omega=np.array([0, 0, 0])
    )

    # lets just glide first, no wind
    initial_inputs = [[0.0, 0.0], np.array([0, 0, 0])]

    # params: you can chase the specifics using this. its a dict
    params = {}

    # teperal resolution of the sim
    dt = 0.1
    # number of steps to run the sim for
    steps = 1000 

    # lets make the objects
    sim = simulator.ParafoilSimulation_6Dof(params, 
                                            [init_state.pos, init_state.vel, init_state.eulers, init_state.omega],
                                            initial_inputs)
    guidance = T_approach([init_state.pos + np.array([0, 0, 300])], dt)

    control = Control()

    # run the sim
    data = run_simulation(sim, control, guidance, steps, dt)

    # only get data between 20 and 50 seconds
    wind_data = [entry for entry in data if 20 < entry[0] < 45]
    vel_b = np.array([entry[1][1] for entry in wind_data])
    eulers = np.array([entry[1][2] for entry in wind_data])
    vel_inertial =[]
    for i, vel in enumerate(vel_b):
        R = sim.get_CDM(eulers[i])
        vel_inertial.append(R @ vel)
    vel_inertial = np.array(vel_inertial)  # Now it's a proper 2D array
    #wind_estimate = least_squares_wind_calc(vel_inertial)
    #print(f"Wind Estimate: {wind_estimate}")
    # Convert data to numpy array for easier manipulation

    plots_to_show = {
        'Position': True,
        'Velocity': True,
        'Acceleration': True,
        'Euler Angles': True,
        'Angular Velocity': True,
        'Angular Acceleration': True,
        'Angle of Attack': True,
        'Sideslip Angle': True,
        'Force Coefficients': True,
        'Moment Coefficients': True,
        'Forces': True,
        'Moments': True,
        'Airspeed Vector': True,
        'Wind Vector': True,
        'Deflection': True
    }

    plot_selected_parameters(data, plots_to_show)
    eulers = np.degrees(np.array([entry[1][2] for entry in data]))
    positions = np.array([entry[1][0] for entry in data])
    # Visualize parafoil pose
    visualize_parafoil_pose(
        euler_series=eulers,
        position_series=positions,
        interval=100,
        slowmo_factor=1.0,
        save_path=False
    )

if __name__ == "__main__":
    main()