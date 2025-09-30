import six_DoF_simulator
from utils import rk4
import numpy as np
import main
import matplotlib.pyplot as plt

def bare_simulate_model(time_list, initial_conditions, inputs, params, inertial = False, coefficients = None, broke_on = True):
    """
    Time list : list of steps in seconds
    initial_conditions: starting state vector (body frame)
    inputs: [left, right, wind]
    coefficients: ORDERDED or as DICTIONARY
    """
    positions = []
    if broke_on is False:
        if inertial:
            positions = [np.array(params['initial_pos']) for _ in time_list]
        else:
            positions = [np.array([0,0,0]) for _ in time_list]
    input = [[inputs[0][0], inputs[1][0]], inputs[2][0]]
    sim = six_DoF_simulator.ParafoilSimulation_6Dof(params, initial_conditions, input)
   # print("coeffs = ",coefficients)
    sim.set_coefficients(coefficients)
    for i,t in enumerate(time_list):
        state = sim.get_state()
        if inertial:
            if broke_on is False:
                positions[i] = sim.get_inertial_position()
            else:
                positions.append(sim.get_inertial_position())
        else:
            if broke_on is False:
                positions[i] = sim.p
            else:
                positions.append(sim.p)

        input = [[inputs[0][i], inputs[1][i]], inputs[2][i]]
        sim.set_inputs(input)
        dt = t - time_list[i - 1] if i > 0 else time_list[1] - time_list[0]
        new_state = rk4(state, sim.get_solver_derivatives, dt)
        sim.set_state(new_state)
        if sim.error is True:
            # we reached error. No point continuing
            # print("hello")
            if broke_on:
                return positions, i
    
    return positions, len(time_list)

def sim_with_noise(time_vector, initial_conditions, inputs, params, inertial = False, coefficients = None):
    gps_noise_std = np.array([0.2, 0.2, 0.2])  # standard deviations in meters
    ideal_positions = bare_simulate_model(time_vector, initial_conditions, inputs, params, inertial, coefficients)
    return ideal_positions + np.random.normal(0,gps_noise_std)

def sim_state_with_noise(time_vector,initial_conditions,inputs,params,inertial = False, coefficients = None):
    ideal_states, broken = multi_obj_sim(time_vector, initial_conditions, inputs, params, inertial, coefficients)
    noisy_states = [
        [
            state[0] + np.random.normal(0, 0.1, size=3),  # position noise
            state[1] + np.random.normal(0, 0.1, size=3),  # velocity noise
            state[2] + np.random.normal(0, 0.1, size=3),  # attitude noise
            state[3] + np.random.normal(0, 0.1, size=3),  # angular velocity noise
        ]
        for state in ideal_states
    ]
    return noisy_states,broken

def multi_obj_sim(time_vector,initial_conditions,inputs,params,inertial = False, coefficients = None):
    """
    Run the 6-DOF simulator with the given coefficients and return the simulated states.
    """
    states = []
    input = [[inputs[0][0], inputs[1][0]], inputs[2][0]]
    sim = six_DoF_simulator.ParafoilSimulation_6Dof(params, initial_conditions, input)
    sim.set_coefficients(coefficients)
    for i,t in enumerate(time_vector):
        state = sim.get_state()
        # get state
        if inertial:
            states.append(sim.get_inertial_state())
        else:
            states.append(state)
        input = [[inputs[0][i], inputs[1][i]], inputs[2][i]]
        sim.set_inputs(input)
        dt = t - time_vector[i - 1] if i > 0 else time_vector[1] - time_vector[0]
        new_state = rk4(state, sim.get_solver_derivatives, dt)
        if sim.error is True:
            # we reached error. No point continuing
            return states, i
        sim.set_state(new_state)
    return states, len(time_vector)

def simulate_model(time_vector, initial_conditions, inputs, params, inertial = False, coefficients = None):
    """
    Run the 6-DOF simulator with the given coefficients and return the simulated states.
    """
    states = []
    data = []
    input = [[inputs[0][0], inputs[1][0]], inputs[2][0]]
    sim = six_DoF_simulator.ParafoilSimulation_6Dof(params, initial_conditions, input)
    sim.set_coefficients(coefficients)

    for i,t in enumerate(time_vector):
        state = sim.get_state()
        # get data
        data.append([
            t, state, sim.angle_of_attack, sim.sideslip_angle,
            sim.angular_acc, sim.acc, sim.CL, sim.CD, sim.Cl, sim.Cn, sim.Cm,
            sim.F_aero, sim.F_g, sim.F_fictious, sim.M_aero, sim.M_f_aero, 
            sim.M_fictious, sim.va, sim.w, [sim.flap_l, sim.flap_r], sim.get_euler_rates() ,sim.get_inertial_position(), sim.M_total, [sim.angle_of_attack, sim.sideslip_angle]
        ])
        # get state
        if inertial:
            states.append(sim.get_inertial_state())
        else:
            states.append(state)
        input = [[inputs[0][i], inputs[1][i]], inputs[2][i]]
        sim.set_inputs(input)
        dt = t - time_vector[i - 1] if i > 0 else time_vector[1] - time_vector[0]
        new_state = rk4(state, sim.get_solver_derivatives, dt)
        sim.set_state(new_state)
    return data, states

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
        'Forces_components': (np.column_stack((
            [np.linalg.norm(entry[11]) for entry in data],
            [np.linalg.norm(entry[12]) for entry in data],
            [np.linalg.norm(entry[13]) for entry in data]
        )), ['F_aero', 'F_g', 'F_fictious'], 'Force (N)'),
        'Moments_components': (np.column_stack((
            [np.linalg.norm(entry[14]) for entry in data],
            [np.linalg.norm(entry[16]) for entry in data]
        )), ['M_aero', 'M_fictious'], 'Moment (Nm)'),
        'Airspeed Vector': (np.array([entry[17] for entry in data]), ['Vx', 'Vy', 'Vz'], 'Airspeed (m/s)'),
        'Wind Vector': (np.array([entry[18] for entry in data]), ['Wind X', 'Wind Y', 'Wind Z'], 'Wind Velocity (m/s)'),
        'Deflection': (np.array([entry[19] for entry in data]), ['Flap Left', 'Flap Right'], 'Deflection (rad)'),
        'Euler Rates': (np.degrees(np.array([entry[20] for entry in data])), ['Roll (ϕ)', 'Pitch (θ)', 'Yaw (ψ)'], 'Angular rate (deg/s)'),
        'Moments': (np.array([entry[22] for entry in data]), ['l', 'm', 'n'], 'Total moment'),
        'Angles': (np.degrees(np.array([entry[23] for entry in data])), ['AoA', 'β'], 'Angles (deg)'),
    }


    for title, (data_array, labels, ylabel) in parameters.items():
        if plots_to_show.get(title, False):
            plot_state_over_time(data_array, labels, f"{title} vs Time", ylabel, times)

    if plots_to_show.get('Moments_components', False):
       plot_force_moment_components_shared(times, data, [11, 12, 13], ['F_aero', 'F_g', 'F_fictious'], "Force", 'Force (N)')

    if plots_to_show.get('Forces_components', False):
       plot_force_moment_components_shared(times, data, [14, 16], ['M_aero', 'M_fictious'], "Moment", 'Moment (Nm)')
   
    plt.show()

    params_3d = {
                'Position': (np.array([entry[1] for entry in data]), 'Position'),
                'inertial Position': (np.array([entry[21] for entry in data]), 'Inertial Position'),
    }

    for title, (data_array, ylabel) in params_3d.items():
        if plots_to_show.get(title, False):
            plot_3d_position(data_array, ylabel)
    plt.show()

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
    
def plot_3d_position(inertial_positions, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # plotting
    ax.plot(inertial_positions[:, 0], inertial_positions[:, 1], inertial_positions[:, 2], label="Parafoil Path", color='blue', linewidth=2)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")
    ax.set_title(title)
    ax.legend()

    # Set equal x and y axis limits
    x_min, x_max = np.min(inertial_positions[:, 0]), np.max(inertial_positions[:, 0])
    y_min, y_max = np.min(inertial_positions[:, 1]), np.max(inertial_positions[:, 1])
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)

    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2

    ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
    ax.set_ylim(y_center - max_range / 2, y_center + max_range / 2)

    plt.show()

if __name__ == "__main__":
    # Example usage
    print("Running simulation...")
    wind_vect = np.array([0, 0, 0])
    initial_conditions = np.array([np.array([0,0,0]), np.array([10,0,3]), np.array([0,0,0]), np.array([0,0,0])])
    snowflake_params = {
        'initial_pos': [0, 0, 500],
    }
    Icarus_params = {
        'initial_pos': [0, 0, 500],
        # 'rigging_angle': np.radians(-7),
        'm': 3.0,
        'Rp': np.array([0, 0, -2.31]),
        'I': np.array([[2.74, 0, 0.001],[0, 2.59, 0],[0.001, 0, 0.158]])
    }
    
    END_TIME = 100
    DT = 0.1
    
    time_vector = np.linspace(0, END_TIME, int(END_TIME / DT))  # Example time vector (0 to 10 seconds, 100 steps)

    L_input = np.zeros_like(time_vector)
    R_input = np.zeros_like(time_vector)
    wind_list = [wind_vect.copy() for _ in range(time_vector.size)]

    def set_input_section(array, start_time, time_duration, value):
        start_index = int(start_time / DT)
        duration = int(time_duration / DT)
        end_index = start_index + duration
        array[start_index:end_index] = value
        return array
    
    # set up the inputs
    L_input = set_input_section(L_input, 10, 30, 0.3)
    R_input = set_input_section(R_input, 50, 70, 0.8)
    inputs = [L_input, R_input, wind_list]

    ic_data, _ = simulate_model(time_vector, initial_conditions, inputs, snowflake_params, True)
    # np.savez('test_data_1.npz', time_vector=time_vector, inputs=inputs, simulated_states=simulated_states)
    #snow_data, _ = simulate_model(time_vector, initial_conditions, inputs, snowflake_params, True)
    # ================ graphing ================

    plots_to_show = {
        'Position': False,
        'Velocity': True,
        'Acceleration': False,
        'Euler Angles': True,
        'Angular Velocity': True,
        'Angular Acceleration': True,
        'Angle of Attack': True,
        'Sideslip Angle': True,
        'Force Coefficients': False,
        'Moment Coefficients': True,
        'Forces_components': False,
        'Moments': True,
        'Airspeed Vector': True,
        'Wind Vector': False,
        'Deflection': True,
        'inertial Position': True,
        'Euler Rates': True,
        'Moments_components': False,
        'Angles': True
    }
    
    plot_selected_parameters(ic_data, plots_to_show)
    #plot_selected_parameters(snow_data, plots_to_show)
    plt.show()

    eulers = np.degrees(np.array([entry[1][2] for entry in ic_data]))
    positions = np.array([entry[1][0] for entry in ic_data])
    inertial_positions = np.array([entry[21] for entry in ic_data])
    # Visualize parafoil pose
    if True:
        main.visualize_parafoil_pose(
            euler_series=eulers,
            position_series=positions,
            interval=100,
            slowmo_factor=1.0,
            save_path=False
        )


"""    # for three graphs stacked vertically
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))  # (nrows=3, ncols=1)
    
    ax1.plot(times, parameters['Deflection'][0][:, 0], label='Flap left', color='r')
    ax1.plot(times, parameters['Deflection'][0][:, 1], label='Flap right', color='b')
    ax1.set_ylabel(parameters['Deflection'][2])  # 'Deflection (rad)'
    ax1.set_xlabel('Time (s)')
    ax1.legend()
    ax1.set_title('Control Surface Deflections')

    ax2.plot(times, parameters['Angle of Attack'][0], label='Angle of Attack', color='g')
    ax2.plot(times, parameters['Sideslip Angle'][0], label='Sideslip Angle', color='m')
    ax2.set_ylabel(parameters['Sideslip Angle'][2])  # 'Deflection (rad)'
    ax2.set_xlabel('Time (s)')
    ax2.legend()
    ax2.set_title('Angle of Attack and Sideslip Angle')

    ax3.plot(times, parameters['Euler Angles'][0][:, 0], label='Roll (ϕ)', color='c')
    ax3.plot(times, parameters['Euler Angles'][0][:, 1], label='Pitch (θ)', color='y')
    ax3.plot(times, parameters['Euler Angles'][0][:, 2], label='Yaw (ψ)', color='orange')
    ax3.set_ylabel(parameters['Euler Angles'][2])  # 'Angle (deg)'
    ax3.set_xlabel('Time (s)')
    ax3.legend()
    ax3.set_title('Euler Angles')
    # Adjust spacing
    plt.tight_layout()

    # Show plot
    plt.show()"""