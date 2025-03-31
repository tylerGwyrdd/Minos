import six_DoF_simulator as simulator
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R
import matplotlib.animation as animation
import math

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
# ------------------------------------------
#  ----------- Simulating  ----------------
# ------------------------------------------

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

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Global X')
    ax.set_ylabel('Global Y')
    ax.set_zlabel('Global Z')
    ax.set_title("Parafoil Pose with Trail + Slow Motion")

    axes = np.eye(3)
    quivers, labels = [], []
    parafoil_poly = None
    trail_line = None
    trail_positions = []

    # Parafoil body geometry
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

    # Center of mass offset vector in body frame
    Rp = np.array([0, 0, -1.11])

    # Rigging angle: rotate parafoil downward about Y-axis
    rigging_angle_deg = -12
    rigging_rotation = R.from_euler('y', rigging_angle_deg, degrees=True)
    parafoil_body = rigging_rotation.apply(parafoil_body)
    Rp = rigging_rotation.apply(Rp)

    def update(frame):
        nonlocal quivers, labels, parafoil_poly, trail_line, trail_positions

        if frame == 0:
            trail_positions.clear()

        roll, pitch, yaw = euler_series[frame]
        pos = position_series[frame]
        rotation = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)

        for q in quivers:
            q.remove()
        for l in labels:
            l.remove()
        if parafoil_poly:
            parafoil_poly.remove()
        if trail_line:
            trail_line.remove()

        # Rotate axes
        rotated_axes = rotation.apply(axes)
        quivers = []
        labels = []
        for i, color in enumerate(['r', 'g', 'b']):
            vec = rotated_axes[i]
            q = ax.quiver(*pos, *vec, color=color, length=1)
            label = ax.text(*(pos + vec * 1.5), ["Forward (X)", "Right (Y)", "Down (Z)"][i], color=color)
            quivers.append(q)
            labels.append(label)

        # Apply rotation and CoM offset
        parafoil_pos = pos + rotation.apply(Rp)
        rotated_parafoil = rotation.apply(parafoil_body) + parafoil_pos

        parafoil_poly = Poly3DCollection([rotated_parafoil], alpha=0.4, color='gray')
        ax.add_collection3d(parafoil_poly)

        # Trail
        trail_positions.append(pos)
        trail_array = np.array(trail_positions)
        trail_line = ax.plot3D(trail_array[:,0], trail_array[:,1], trail_array[:,2], color='black')[0]

        # View window
        if fixed_camera:
            view = 10 * zoom
            ax.set_xlim([-view, view])
            ax.set_ylim([-view, view])
            ax.set_zlim([-view, view])
        else:
            ax.set_xlim([pos[0] - zoom, pos[0] + zoom])
            ax.set_ylim([pos[1] - zoom, pos[1] + zoom])
            ax.set_zlim([pos[2] - zoom, pos[2] + zoom])

        return quivers + labels + [parafoil_poly, trail_line]

    actual_interval = int(interval * slowmo_factor)
    ani = animation.FuncAnimation(fig, update, frames=len(euler_series), interval=actual_interval, blit=False, repeat=True)

    if save_path:
        from matplotlib.animation import PillowWriter
        writer = PillowWriter(fps=1000 // actual_interval)
        ani.save(save_path, dpi=200, writer=writer)
    else:
        plt.show()

# system definitions
# use defalts for now
params = {
}

# ----------- inital state definitions ----------------
# inertial frame positions
p_inital = np.array([0,0,0]) # inertial frame position

v_inital = np.array([10,0,3])

# euler angles of body fixed frame
eulers = np.radians(np.array([0,0,0]))# euler angles

# angular velocity of body fixed frame
angular_velocity = np.array([0,0,0]) # angular velocity

# inital state
inital_state = [p_inital, v_inital, eulers, angular_velocity]

# wind vector
wind_vector = np.array([0,0,0]) # wind vector

# flap deflections
l_flap = 0.1 # left flap angle
r_flap = 0.1 # right flap angle

#inital inputs
inital_inputs = [[l_flap, r_flap], wind_vector]


# -------------- simulation ----------------

data = []

# Time step
dt = 0.1
t = 0

# Run simulation
sim = simulator.ParafoilSimulation_6Dof(params, inital_state, inital_inputs)


state = inital_state
for i in range(100):
    data.append([t,sim.get_state(),sim.angle_of_attack,sim.sideslip_angle,sim.angular_acc,sim.acc,sim.CL,sim.CD,sim.Cl,sim.Cn,sim.Cm,sim.F_areo, sim.F_g, sim.F_tishious, sim.M_aero, sim.M_f_areo, sim.M_fictious,sim.va])
    new_state = rk4(sim.get_state(), sim.get_solver_derivities, dt)
    sim.set_state(new_state)
    sim.calculate_derivitives()
    t += dt
    if i % 10 == 0:
        print(f"t = {t:.3f}")
        print("  Position:", new_state[0])
        print("  Velocity (body):", new_state[1])
        print("  Euler angles (deg):", np.degrees(new_state[2]))
        print("  Angular velocity (deg/s):", np.degrees(new_state[3]))
    
# Unpack all states
times = [entry[0] for entry in data]
positions     = [entry[1][0] for entry in data]  # inertial position
velocities    = [entry[1][1] for entry in data]  # body-frame velocity
accelerations = [entry[5] for entry in data]
eulers        = [entry[1][2] for entry in data]  # Euler angles (rad)
angular_vels  = [entry[1][3] for entry in data]  # angular velocity (rad/s)
angular_acc = [entry[4] for entry in data]

angles_of_attack = [entry[2] for entry in data]
sideslip_angle = [entry[3] for entry in data]

CL = [entry[6] for entry in data]
CD = [entry[7] for entry in data]
Cl = [entry[8] for entry in data]
Cn = [entry[9] for entry in data]
Cm = [entry[10] for entry in data]



F_areo = [entry[11] for entry in data]
F_g = [entry[12] for entry in data]
F_fictious = [entry[13] for entry in data]

M_areo = [entry[14] for entry in data]
M_f_areo = [entry[15] for entry in data]
M_fictious = [entry[16] for entry in data]

wind_vector = [entry[17] for entry in data]

F_areo = [np.linalg.norm(f) for f in F_areo]
F_g = [np.linalg.norm(f) for f in F_g]
F_fictious = [np.linalg.norm(f) for f in F_fictious]

M_areo = [np.linalg.norm(m) for m in M_areo]
M_f_areo = [np.linalg.norm(m) for m in M_f_areo]
M_fictious = [np.linalg.norm(m) for m in M_fictious]

# Convert lists of vectors to arrays for easy indexing
positions = np.array(positions)
velocities = np.array(velocities)
accelerations = np.array(accelerations)
eulers = np.degrees(np.array(eulers))  # Convert radians to degrees for readability
angular_vels = np.degrees(np.array(angular_vels))  # Also degrees/sec for clarity
angular_acc = np.degrees(np.array(angular_acc))

angles_of_attack = np.degrees(np.array(angles_of_attack))  # Convert radians to degrees for readability
sideslip_angle = np.degrees(np.array(sideslip_angle))  # Convert radians to degrees for readability

force_coeffs = np.column_stack((CL,CD))
moments_coeffs = np.column_stack((Cl,Cn,Cm))

forces = np.column_stack((F_areo,F_g,F_fictious))
moments = np.column_stack((M_areo,M_f_areo,M_fictious))

wind_vector = np.array(wind_vector)
## 3d graph

x_pos = [pos[0] for pos in positions]
y_pos = [pos[1] for pos in positions]
z_pos = [pos[2] for pos in positions]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_pos, y_pos,z_pos)
ax.scatter(x_pos[0], y_pos[0], z_pos[0], color='red', s=50, marker='o', label='Start Point')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Parafoil simulation')
# Show the plot
plt.show()



visualize_parafoil_pose(eulers,positions,dt * 1000,2,6,None,False)

# Plotting utility
def plot_state_over_time(data, labels, title, ylabel):
    plt.figure()
    if data.ndim == 1:
        data = data[:, np.newaxis]  # shape (N,) → (N,1)
    num_series = data.shape[1]
    for i in range(num_series):
        plt.plot(times, data[:, i], label=labels[i])
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

# Plot each group
plot_state_over_time(positions, ['X', 'Y', 'Z'], 'Position vs Time', 'Position (m)')
plot_state_over_time(velocities, ['u', 'v', 'w'], 'Body-frame Velocity vs Time', 'Velocity (m/s)')
plot_state_over_time(accelerations, ['u', 'v', 'w'], 'Body-frame accelerations vs Time', 'acceleration (m/s^2)')
plot_state_over_time(eulers, ['Roll (ϕ)', 'Pitch (θ)', 'Yaw (ψ)'], 'Euler Angles vs Time', 'Angle (degrees)')
plot_state_over_time(angular_vels, ['p', 'q', 'r'], 'Angular Velocity vs Time', 'Angular rate (deg/s)')
plot_state_over_time(angular_acc, ['p', 'q', 'r'], 'Angular accelerations vs Time', 'Angular rate (deg/s^2)')

plot_state_over_time(angles_of_attack, ['angles_of_attack'], 'angles_of_attack vs Time', 'Angle (degrees)')
plot_state_over_time(sideslip_angle,  ['sideslip_angle'], 'sideslip_angle vs Time', 'Angle (degrees)')

plot_state_over_time(force_coeffs, ['CL','CD'], 'force coefficients vs Time', 'number')
plot_state_over_time(moments_coeffs, ['Cl','Cn','Cm'], 'moment coefficients vs Time', 'number')

plot_state_over_time(forces, ['F_areo','F_g','F_fictious'], 'force vs Time', 'Newtons')
plot_state_over_time(moments, ['M_areo','M_f_areo','M_fictious'], 'moment vs Time', 'Newton Meter')

plot_state_over_time(wind_vector, ['u', 'v', 'w'], 'Body-frame Velocity vs Time', 'Velocity (m/s)')

plt.show()