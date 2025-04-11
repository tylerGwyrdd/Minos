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

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(message)s')


@dataclass
class State:
    pos: np.ndarray
    vel: np.ndarray
    eulers: np.ndarray
    omega: np.ndarray


# ODE solver, simplest one, error prone
def forward_euler(state, derivatives, dt):
    dx = [dt * der for der in derivatives]
    new_state = [s + a for s, a in zip(state, dx)]
    return new_state

def rk4(state, derivative_func, dt):
    """
    Runge-Kutta 4th order integrator.

    - `state`: list of np.array state vectors [pos, vb, eulers, omega]
    - `derivative_func`: function that returns derivatives given a state
    - `dt`: timestep
    """

    def add_scaled(state, derivative, scale):
        return [s + scale * d for s, d in zip(state, derivative)]

    # k1
    k1 = derivative_func(state)

    # k2
    state_k2 = add_scaled(state, k1, dt / 2)
    k2 = derivative_func(state_k2)

    # k3
    state_k3 = add_scaled(state, k2, dt / 2)
    k3 = derivative_func(state_k3)

    # k4
    state_k4 = add_scaled(state, k3, dt)
    k4 = derivative_func(state_k4)

    # Weighted average
    new_state = [
        s + (dt / 6) * (d1 + 2 * d2 + 2 * d3 + d4)
        for s, d1, d2, d3, d4 in zip(state, k1, k2, k3, k4)
    ]

    return new_state

# ----------- system definitions ----------------
def visualize_parafoil_pose(
    euler_series,
    position_series,
    interval=100,
    slowmo_factor=1.0,
    zoom=10,
    save_path=None,
    fixed_camera=False
):
    """
    Visualize the position and rotation of a parafoil over time with trail, rigging angle,
    center-of-mass offset, optional GIF saving (via Pillow), and optional fixed camera view.

    Parameters:
    - euler_series: (N, 3) array-like of [roll, pitch, yaw] in degrees.
    - position_series: (N, 3) array-like of [x, y, z] positions.
    - interval: base time between frames in milliseconds (before slowmo).
    - slowmo_factor: multiplier for playback speed (e.g., 2 = slower, 0.5 = faster).
    - zoom: half-width of the visible region in each axis direction (larger = zoomed out).
    - save_path: filename to save animation (must be .gif). If None, shows live.
    - fixed_camera: if True, camera view stays fixed around origin at zoom range.
    """
    euler_series = np.array(euler_series)
    position_series = np.array(position_series)
    assert len(euler_series) == len(position_series), "Rotation and position series must be the same length."

    num_frames = len(euler_series)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.2)

    ax.set_xlabel('Global X')
    ax.set_ylabel('Global Y')
    ax.set_zlabel('Global Z')
    ax.set_title("Parafoil Pose with Trail + Controls")

    axes = np.eye(3)
    quivers, labels = [], []
    parafoil_poly = None
    trail_line = None
    trail_positions = []

    wing_span = 2.0
    chord_length = 0.6
    half_span = wing_span / 2
    half_chord = chord_length / 2
    parafoil_body = np.array([
        [ half_chord, -half_span, 0],
        [ half_chord,  half_span, 0],
        [-half_chord,  half_span, 0],
        [-half_chord, -half_span, 0]
    ])
    Rp = np.array([0, 0, -1.11])
    rigging_angle_deg = -12
    rigging_rotation = R.from_euler('y', rigging_angle_deg, degrees=True)
    parafoil_body = rigging_rotation.apply(parafoil_body)
    Rp = rigging_rotation.apply(Rp)

    paused = [False]
    current_frame = [0]

    def draw_frame(frame):
        nonlocal quivers, labels, parafoil_poly, trail_line, trail_positions

        current_frame[0] = frame

        if frame == 0:
            trail_positions.clear()

        roll, pitch, yaw = euler_series[frame]
        pos = position_series[frame]
        rotation = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)

        for q in quivers: q.remove()
        for l in labels: l.remove()
        if parafoil_poly: parafoil_poly.remove()
        if trail_line: trail_line.remove()

        rotated_axes = rotation.apply(axes)
        quivers, labels = [], []
        for i, color in enumerate(['r', 'g', 'b']):
            vec = rotated_axes[i]
            q = ax.quiver(*pos, *vec, color=color, length=1)
            label = ax.text(*(pos + vec * 1.5), ["Forward (X)", "Right (Y)", "Down (Z)"][i], color=color)
            quivers.append(q)
            labels.append(label)

        parafoil_pos = pos + rotation.apply(Rp)
        rotated_parafoil = rotation.apply(parafoil_body) + parafoil_pos
        parafoil_poly = Poly3DCollection([rotated_parafoil], alpha=0.4, color='gray')
        ax.add_collection3d(parafoil_poly)

        trail_positions.append(pos)
        trail_array = np.array(trail_positions)
        trail_line = ax.plot3D(trail_array[:, 0], trail_array[:, 1], trail_array[:, 2], color='black')[0]

        if fixed_camera:
            view = 10 * zoom
            ax.set_xlim([-view, view])
            ax.set_ylim([-view, view])
            ax.set_zlim([-view, view])
        else:
            ax.set_xlim([pos[0] - zoom, pos[0] + zoom])
            ax.set_ylim([pos[1] - zoom, pos[1] + zoom])
            ax.set_zlim([pos[2] - zoom, pos[2] + zoom])

        fig.canvas.draw_idle()

    # Slider setup (smaller and lower)
    slider_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    frame_slider = Slider(slider_ax, 'Frame', 0, num_frames - 1, valinit=0, valfmt='%0.0f')

    # Button setup (small play/pause)
    button_ax = fig.add_axes([0.82, 0.045, 0.1, 0.04])
    play_button = Button(button_ax, 'Pause')

    def slider_update(val):
        paused[0] = True
        play_button.label.set_text('Play')
        frame = int(frame_slider.val)
        draw_frame(frame)

    def toggle_play(event):
        paused[0] = not paused[0]
        play_button.label.set_text('Pause' if not paused[0] else 'Play')

    frame_slider.on_changed(slider_update)
    play_button.on_clicked(toggle_play)

    # Animation function
    actual_interval = int(interval * slowmo_factor)
    def update(frame):
        if not paused[0]:
            draw_frame(current_frame[0])
            current_frame[0] = (current_frame[0] + 1) % num_frames
            frame_slider.set_val(current_frame[0])

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=actual_interval, blit=False, repeat=True)

    if save_path:
        from matplotlib.animation import PillowWriter
        writer = PillowWriter(fps=1000 // actual_interval)
        ani.save(save_path, dpi=200, writer=writer)
    else:
        plt.show()

def run_simulation(sim, steps, dt):
    data = []
    t = 0
    fired = False
    for i in range(steps):
        state = sim.get_state()
        data.append([
            t, state, sim.angle_of_attack, sim.sideslip_angle,
            sim.angular_acc, sim.acc, sim.CL, sim.CD, sim.Cl, sim.Cn, sim.Cm,
            sim.F_aero, sim.F_g, sim.F_fictious, sim.M_aero, sim.M_f_aero, sim.M_fictious, sim.va, sim.w, [sim.flap_l, sim.flap_r]
        ])
        sim.set_state(rk4(state, sim.get_solver_derivatives, dt))
        sim.calculate_derivatives()
        t += dt
        if t >= 5 and not fired:
            fired = True
            sim.set_inputs([[0.00, 0.02], np.array([0, 0, 0])])
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
        'Moment Coefficients': (np.column_stack(([entry[8] for entry in data], [entry[9] for entry in data], [entry[10] for entry in data])), ['Cl', 'Cn', 'Cm'], 'Coefficient'),
        'Forces': (np.column_stack((
            [np.linalg.norm(entry[11]) for entry in data],
            [np.linalg.norm(entry[12]) for entry in data],
            [np.linalg.norm(entry[13]) for entry in data]
        )), ['F_aero', 'F_g', 'F_fictious'], 'Force (N)'),
        'Moments': (np.column_stack((
            [np.linalg.norm(entry[14]) for entry in data],
            [np.linalg.norm(entry[15]) for entry in data],
            [np.linalg.norm(entry[16]) for entry in data]
        )), ['M_aero', 'M_f_aero', 'M_fictious'], 'Moment (Nm)')
    }

    for title, (data_array, labels, ylabel) in parameters.items():
        if plots_to_show.get(title, False):
            plot_state_over_time(data_array, labels, f"{title} vs Time", ylabel, times)

    if plots_to_show.get('Forces', False):
        plot_force_moment_components_shared(times, data, [11, 12, 13], ['F_aero', 'F_g', 'F_fictious'], "Force", 'Force (N)')

    if plots_to_show.get('Moments', False):
        plot_force_moment_components_shared(times, data, [14, 15, 16], ['M_aero', 'M_f_aero', 'M_fictious'], "Moment", 'Moment (Nm)')

def main():
    # Initial state
    init_state = State(
        pos=np.array([0, 0, 0]),
        vel=np.array([10, 0, 3]),
        eulers=np.radians(np.array([0, 0, 0])),
        omega=np.array([0, 0, 0])
    )

    initial_inputs = [[0.0, 0.0], np.array([0, 0, 0])]
    params = {}

    sim = simulator.ParafoilSimulation_6Dof(params, [init_state.pos, init_state.vel, init_state.eulers, init_state.omega], initial_inputs)
    dt = 0.1
    steps = 1000
    data = run_simulation(sim, steps, dt)

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
        'Wind Vector': False
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
        zoom=10,
        save_path=False,
        fixed_camera=True
    )

if __name__ == "__main__":
    main()