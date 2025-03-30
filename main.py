import six_DoF_simulator as simulator
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

# system definitions
# use defalts for now
params = {
}

# ----------- inital state definitions ----------------
# inertial frame positions
p_inital = np.array([0,0,100]) # inertial frame position

v_inital = np.array([9,0,3])

# euler angles of body fixed frame
eulers = np.radians(np.array([0,30,0]))# euler angles

# angular velocity of body fixed frame
angular_velocity = np.array([0,0,0]) # angular velocity

# inital state
inital_state = [p_inital, v_inital, eulers, angular_velocity]

# wind vector
wind_vector = np.array([0,0,0]) # wind vector

# flap deflections
l_flap = 0.0 # left flap angle
r_flap = 0.0 # right flap angle

#inital inputs
inital_inputs = [[l_flap, r_flap], wind_vector]


# -------------- simulation ----------------

data = []

# Time step
dt = 0.01
t = 0

# Run simulation
sim = simulator.ParafoilSimulation_6Dof(params, inital_state, inital_inputs)


state = inital_state
for i in range(250):
    data.append([t,sim.get_state()])
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
    
# might not be working because consideration for negitive angle of attacks are not considered. 

position = [np.array(state[0]) for _, state in data]
x_pos = [pos[0] for pos in position]
y_pos = [pos[1] for pos in position]
z_pos = [pos[2] for pos in position]

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

# Unpack all states
times = [entry[0] for entry in data]
positions     = [entry[1][0] for entry in data]  # inertial position
velocities    = [entry[1][1] for entry in data]  # body-frame velocity
eulers        = [entry[1][2] for entry in data]  # Euler angles (rad)
angular_vels  = [entry[1][3] for entry in data]  # angular velocity (rad/s)

# Convert lists of vectors to arrays for easy indexing
positions = np.array(positions)
velocities = np.array(velocities)
eulers = np.degrees(np.array(eulers))  # Convert radians to degrees for readability
angular_vels = np.degrees(np.array(angular_vels))  # Also degrees/sec for clarity

# Plotting utility
def plot_state_over_time(data, labels, title, ylabel):
    plt.figure()
    for i in range(3):
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
plot_state_over_time(eulers, ['Roll (ϕ)', 'Pitch (θ)', 'Yaw (ψ)'], 'Euler Angles vs Time', 'Angle (degrees)')
plot_state_over_time(angular_vels, ['p', 'q', 'r'], 'Angular Velocity vs Time', 'Angular rate (deg/s)')

plt.show()