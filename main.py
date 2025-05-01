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
import guidance_v2
import copy
# Logging setup
logging.basicConfig(level=logging.INFO, format='%(message)s')

def simple_time_control(time):
    # Simple control logic for demonstration purposes
    if time < 15:
        return [0, 0]  # Flap deflections
    elif time < 40:
        return [0.0, 0.3]
    else:
        return [0.0, 0.0]

def run_simulation(sim, guidance_params, steps, dt):
    data = []
    t = 0
    for i in range(steps):
        # get state
        state = sim.get_state()
        inertial_state = sim.get_inertial_state()
        # check if we have hit the ground
        if(inertial_state[0][2] < guidance_params["IPI"][2]):
            logging.info("Parafoil has hit the ground, stopping simulation.")
            logging.info(f"Final Position: {[f'{coord:.3g}' for coord in state[0]]}")
            IPI_error = guidance_params['IPI'][:2] - state[0][:2]
            logging.info(f"Final IPI Error: {[f'{e:.3g}' for e in IPI_error]}")
            estimated_wind_vector = guidance_params['wind_unit_vector'] * guidance_params['wind_magnitude']
            logging.info(f"Estimated Wind: {[f'{w:.3g}' for w in estimated_wind_vector]}")
            logging.info(f"Actual Wind: {[f'{w:.3g}' for w in sim.w]}")
            break

        # ============ path planning coding ============================
        current_heading = state[2][2]
        current_euler_rate = inertial_state[3][2]
        desired_heading,_ = guidance_v2.guidance_update(guidance_params, inertial_state)
        new_inputs = guidance_v2.PID_control(current_heading, desired_heading,current_euler_rate)
        # logging.info(f"Current heading: {current_heading:.2f}, Desired heading: {desired_heading:.2f}, new inputs: {new_inputs}")
        
        # ================ simple physics sim =======================
        #new_inputs = simple_control(t)
        #desired_heading, current_heading = 0, 0

        # update the control inputs
        sim.set_desired_flaps(new_inputs)
        sim.update_flaps(dt)
        # uses rk4 to work out how much derivititve to add - its is pretty good
        new_state = rk4(state, sim.get_solver_derivatives, dt)
        # update sim
        sim.set_state(new_state)

        # save the current data
        data.append([
            t, state, sim.angle_of_attack, sim.sideslip_angle,
            sim.angular_acc, sim.acc, sim.CL, sim.CD, sim.Cl, sim.Cn, sim.Cm,
            sim.F_aero, sim.F_g, sim.F_fictious, sim.M_aero, sim.M_f_aero, 
            sim.M_fictious, sim.va, sim.w, [sim.flap_l, sim.flap_r],
            [current_heading, desired_heading], sim.get_inertial_position(), 
            sim.get_euler_rates()
        ])
        # update the time
        t += dt
        """        # Log every 10 steps
                if i % 10 == 0:
                    logging.info(f"t = {t:.3f}")
                    logging.info("  Position: %s", state[0])
                    logging.info("  Velocity (body): %s", state[1])
                    logging.info("  Euler angles (deg): %s", np.degrees(state[2]))
                    logging.info("  Angular velocity (deg/s): %s", np.degrees(state[3]))"""
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
        'Deflection': (np.array([entry[19] for entry in data]), ['Flap Left', 'Flap Right'], 'Deflection (rad)'),
        'headings':  (np.degrees(np.array([entry[20] for entry in data])), ['current', 'desired'], 'degrees'),
        'Euler Rates': (np.degrees(np.array([entry[22] for entry in data])), ['Roll (ϕ)', 'Pitch (θ)', 'Yaw (ψ)'], 'Angular rate (deg/s)'),
    }

    for title, (data_array, labels, ylabel) in parameters.items():
        if plots_to_show.get(title, False):
            plot_state_over_time(data_array, labels, f"{title} vs Time", ylabel, times)

    if plots_to_show.get('Forces', False):
        plot_force_moment_components_shared(times, data, [11, 12, 13], ['F_aero', 'F_g', 'F_fictious'], "Force", 'Force (N)')

    if plots_to_show.get('Moments', False):
        plot_force_moment_components_shared(times, data, [14, 15, 16], ['M_aero', 'M_f_aero', 'M_fictious'], "Moment", 'Moment (Nm)')

def main():
    # sim params
    # teperal resolution of the sim
    dt = 0.1
    # number of steps to run the sim for
    steps = 5000 

    # -------------- sim init -----------------------
    # Note sim uses NED Coordinate system, so Z is up
    init_body_pos=np.array([0, 0, 0])
    init_body_vel=np.array([10, 0, 3]) # this is stable
    init_eulers=np.array([0, 0, 0])
    init_omega=np.array([0, 0, 0])
    init_state = [init_body_pos,init_body_vel,init_eulers,init_omega]

    deployment_pos_inertial = np.array([0, 50, 500])

    wind = np.array([1, 1, 0])

    initial_inputs = [[0.0, 0.0], wind]

    # params: you can chase the specifics using this. its a dict
    sim_params = {
        'initial_pos': deployment_pos_inertial ,
    }
    # guidance
    guidance_params = {
    'deployment_pos': deployment_pos_inertial,
    'final_approach_height': 100,
    'spirialing_radius': 20,
    'update_rate': dt,
    'wind_unit_vector': np.array([1, 0]),
    'wind_magnitude': 0.0,
    'wind_v_list': [],
    'horizontal_velocity': 5.9,
    'sink_velocity': 4.9,
    'IPI': np.array([0, 0, 0]),
    'flare_height': 20,
    'initialised': False,
    'mode': 'initialising',  # initalising, homing, final approach, energy management
    'start_heading': 0,  # radians
    'desired_heading': np.deg2rad(0),  # radians
    'FTP_centre': np.array([0,0]),  # Final target point
    }
    # save for later
    ideal_guidance_params = copy.deepcopy(guidance_params)
    ideal_guidance_path = []
    # ----------------------- running --------------------------
    # lets make the objects
    sim = simulator.ParafoilSimulation_6Dof(sim_params, init_state, initial_inputs)

    # run the sim
    data = run_simulation(sim,guidance_params, steps, dt)
    guidance_init_state = init_state
    guidance_init_state[0] = deployment_pos_inertial
    ideal_guidance_path = guidance_v2.ideal_guidance(ideal_guidance_params, wind, init_state, 10, steps)
    
    # ------------------------ post ---------------------------
    plots_to_show = {
        'Position': True,
        'Velocity': False,
        'Acceleration': False,
        'Euler Angles': True,
        'Angular Velocity': False,
        'Angular Acceleration': False,
        'Angle of Attack': False,
        'Sideslip Angle': True,
        'Force Coefficients': False,
        'Moment Coefficients': False,
        'Forces': False,
        'Moments': False,
        'Airspeed Vector': True,
        'Wind Vector': True,
        'Deflection': True,
        'Euler Rates': True,
        'headings': True
    }

    plot_selected_parameters(data, plots_to_show)
    eulers = np.degrees(np.array([entry[1][2] for entry in data]))
    positions = np.array([entry[1][0] for entry in data])
    inertial_positions = np.array([entry[21] for entry in data])
    # Visualize parafoil pose
    visualize_parafoil_pose(
        euler_series=eulers,
        position_series=positions,
        interval=100,
        slowmo_factor=1.0,
        save_path=False
    )
    # ------------------- ideal guidance ----------------------
    # lets generate ideal guidance path

    guidance_v2.plot_3D_position(inertial_positions,guidance_params, ideal_guidance_path)

if __name__ == "__main__":
    main()
