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
for i in range(20):
    print(f"t = {t:.3f}")
    data.append([t,sim.get_state()])
    new_state = sim.update_state(forward_euler, dt)
    t += dt
    
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