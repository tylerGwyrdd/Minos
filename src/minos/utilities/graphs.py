import numpy as np
import matplotlib.pyplot as plt
import minos.physics.six_DoF_simulator as simulator
import logging

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
