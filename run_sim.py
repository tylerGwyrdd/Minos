import six_DoF_simulator
from utils import rk4
import numpy as np
import main
import matplotlib.pyplot as plt

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
            sim.M_fictious, sim.va, sim.w, [sim.flap_l, sim.flap_r], sim.get_inertial_position()
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


if __name__ == "__main__":
    # Example usage
    print("Running simulation...")
    wind_vect = np.array([0, 0, 0])
    initial_conditions = np.array([np.array([0,0,0]), np.array([10,0,3]), np.array([0,0,0]), np.array([0,0,0])])
    params = {
        'initial_pos': [0, 0, 0],
    }
    time_vector = np.linspace(0, 25, 250)  # Example time vector (0 to 10 seconds, 100 steps)

    L_input = np.zeros_like(time_vector)
    R_input = np.zeros_like(time_vector)
    wind_list = [wind_vect.copy() for _ in range(time_vector.size)]

    print("  ")
    def set_input_section(array, start_index, duration, value):
        end_index = start_index + duration
        array[start_index:end_index] = value
        return array
    
    # set up the inputs
    L_input = set_input_section(L_input, 50, 100, 0.3)
    #R_input = set_input_section(R_input, 130, 230, 0.3)
    inputs = [L_input, R_input, wind_list]
    data = []

    data, simulated_states = simulate_model(time_vector, initial_conditions, params, inputs, True)
    # np.savez('test_data_1.npz', time_vector=time_vector, inputs=inputs, simulated_states=simulated_states)
    print(simulated_states)

    # ================ graphing ================

    plots_to_show = {
        'Position': False,
        'Velocity': False,
        'Acceleration': False,
        'Euler Angles': True,
        'Angular Velocity': False,
        'Angular Acceleration': False,
        'Angle of Attack': True,
        'Sideslip Angle': True,
        'Force Coefficients': False,
        'Moment Coefficients': False,
        'Forces': False,
        'Moments': False,
        'Airspeed Vector': False,
        'Wind Vector': False,
        'Deflection': True
    }
    plot = False
    if plot:
        main.plot_selected_parameters(data, plots_to_show)
        eulers = np.degrees(np.array([entry[1][2] for entry in data]))
        positions = np.array([entry[1][0] for entry in data])
        inertial_positions = np.array([entry[-1] for entry in data])
        # Visualize parafoil pose

        main.visualize_parafoil_pose(
            euler_series=eulers,
            position_series=positions,
            interval=100,
            slowmo_factor=1.0,
            save_path=False
        )
    # ==================== neat graphing =========================
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

    # for three graphs stacked vertically
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
    plt.show()