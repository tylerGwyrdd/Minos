import numpy as np
import matplotlib.pyplot as plt
from guidance_v2 import guidance_update
import logging
import wind_estimation
# Simulation parameters
actual_wind_vector = np.array([-6, 0])  # some wind
initial_position = np.array([140, 0, 715])
initial_velocity = np.array([0, 0.0, 0])  # Downward velocity
lookahead_distance = 20


# Simulate path
num_steps = 1500
state = [initial_position, initial_velocity, np.array([0, 0, 0])]
# control parameters
max_turn_rate = np.deg2rad(10)  # radians per step

params = {
    'deployment_pos': initial_position,
    'final_approach_height': 100,
    'spirialing_radius': 20,
    'update_rate': 0.1,
    'wind_unit_vector': np.array([1, 0]),
    'wind_magnitude': 0.0,
    'horizontal_velocity': 6,
    'sink_velocity': 5,
    'IPI': np.array([0, 0, 0]),
    'flare_height': 20,
    'initialised': False,
    'mode': 'initialising',  # initalising, homing, final approach, energy management
    'start_heading': np.deg2rad(0),  # radians
    'desired_heading': np.deg2rad(0),  # radians
    'FTP_centre': np.array([0,0]),  # Final target point
    'RLS': wind_estimation.WindRLS(0.0001,10000000000000)
}

# for plotting
positions = [initial_position]
velocities = [[]]
dt = params['update_rate']  # time step


def get_wind(step):
    if step < 500:
        return np.array([-4.4, 0])
    else:
        return np.array([-2, 0])
def wrap_angle(angle):
    """
    Wrap angle to the range [0, 2pi].
    """
    return (angle + 2*np.pi) % (2 * np.pi)

wind = []
wind_esti = wind_estimation.HorizontalWindModel()
for i in range(num_steps):
    # stop sim if we have landed on ground
    estimated_wind_vector = params['wind_unit_vector'] * params['wind_magnitude']
    if state[0][2] < 0:
        print("Parafoil has landed.")
        print(f"Final Position: {[f'{coord:.3g}' for coord in state[0]]}")
        IPI_error = params['IPI'][:2] - state[0][:2]
        print(f"Final IPI Error: {[f'{e:.3g}' for e in IPI_error]}")
        estimated_wind_vector = params['wind_unit_vector'] * params['wind_magnitude']
        print(f"Estimated Wind: {[f'{w:.3g}' for w in estimated_wind_vector]}")
        print(f"Actual Wind: {[f'{w:.3g}' for w in get_wind(i)]}")
        #print(f"ftp_centre: {[f'{w:.3g}' for w in params['FTP_centre']]}")
        break

    # ================== guidance update ================
    desired_heading,_ = guidance_update(params, state)

    # ================== Control Logic ==================
    # Adjust current heading toward desired heading, limited by max_turn_rate
    current_heading = state[2][2]
    heading_error = (desired_heading - current_heading + np.pi) % (2 * np.pi) - np.pi # Normalize to [-pi, pi]
    heading_change = np.clip(heading_error, -max_turn_rate, max_turn_rate)
    new_heading = current_heading + heading_change
    print(f"Heading Error: {np.degrees(heading_error)}, Current Heading: {np.degrees(wrap_angle(current_heading))}, Desired Heading: {np.degrees(wrap_angle(desired_heading))}, New Heading: {np.degrees(wrap_angle(new_heading))}")
    
    # ================= Motion update ===================
    # Move in the direction of the current heading + wind
    vel_x = params['horizontal_velocity'] * np.cos(new_heading) + wind_esti.wind_at_altitude(state[0][2])[0]
    vel_y = params['horizontal_velocity'] * np.sin(new_heading) + wind_esti.wind_at_altitude(state[0][2])[1]    

    #vel_x = params['horizontal_velocity'] * np.cos(new_heading) + get_wind(i)[0]
    #vel_y = params['horizontal_velocity'] * np.sin(new_heading) + get_wind(i)[1]
    vel_z = params['sink_velocity']  # Assuming constant sink velocity
    new_x = vel_x * dt + state[0][0]  # Update x position
    new_y = vel_y * dt + state[0][1]  # Update y position
    new_z = state[0][2] - vel_z * dt  # Update z position
    print(f"New Position: {new_x}, {new_y}, {new_z}")
    print(f"New Velocity: {vel_x}, {vel_y}, {vel_z}")
    state = [np.array([new_x,new_y,new_z]),
             np.array([vel_x,vel_y,vel_z]),
             np.array([0,0,new_heading])
             ]  # Update state
    wind.append([wind_esti.wind_at_altitude(state[0][2])[0], estimated_wind_vector[0]])
    positions.append(state[0])  # Store position for plotting

# ================= Graphs ==================
# Plotting (3D)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def plot_3D_position(data,guidance_params,estimated_path = None):
    positions = np.array([entry[1][0] for entry in data])
    inertial_positions = np.array([entry[0] + guidance_params["deployment_pos"]  for entry in positions])
    
    # plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(inertial_positions[:, 0], inertial_positions[:, 1], inertial_positions[:, 2], label="Parafoil Path", linewidth=2)
    if estimated_path is not None:
        print("hello")  
        ax.plot(estimated_path[:, 0], estimated_path[:, 1], estimated_path[:, 2], color = 'green', label="Parafoil Path", linewidth=2)
    
    ax.scatter(guidance_params["deployment_pos"][0], guidance_params["deployment_pos"][1], guidance_params["deployment_pos"][2], color='red', label="Start Position")
    ax.scatter(guidance_params['IPI'][0], guidance_params['IPI'][1], guidance_params['IPI'][2], color='green', label="Impact Point Indicator (IPI)")
    ax.scatter(guidance_params['FTP_centre'][0], guidance_params['FTP_centre'][1], guidance_params['final_approach_height'], color='blue', label="Final Target Point (FTP)")
    wind_line = guidance_params['IPI'] + np.array([-guidance_params['wind_unit_vector'][0] * 100,guidance_params['wind_unit_vector'][1] * 100, 0])
    print(f"Wind Line: {wind_line}")
    ax.plot([guidance_params['IPI'][0], wind_line[0]],[guidance_params['IPI'][1], wind_line[1]],[guidance_params['IPI'][2],wind_line[2]], color='orange', label="Wind Vector", linewidth=2, alpha=0.5)
    # Labels and title
    ax.set_title("3D Parafoil Path")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")
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
    return

positions = np.array(positions)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot path and start position
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label="Icarus Path", linewidth=2)
ax.scatter(initial_position[0], initial_position[1], initial_position[2], color='red', label="Start Position")
ax.scatter(params['IPI'][0], params['IPI'][1], params['IPI'][2], color='green', label="Impact Point Indicator (IPI)")
ax.scatter(params['FTP_centre'][0], params['FTP_centre'][1], params['final_approach_height'], color='blue', label="Final Target Point (FTP)")
#wind_line = params['IPI'] + np.array([params['wind_unit_vector'][0] * 100,params['wind_unit_vector'][1] * 100, 0])
#print(f"Wind Line: {wind_line}")
#ax.plot([params['IPI'][0], wind_line[0]],[params['IPI'][1], wind_line[1]],[params['IPI'][2],wind_line[2]], color='orange', label="Wind Vector", linewidth=2, alpha=0.5)
# Labels and title
ax.set_title("3D Parafoil Path")
ax.set_xlabel("East Postion")
ax.set_ylabel("North Position")
ax.set_zlabel("Altitude")
ax.legend()

# Set equal x and y axis limits
x_min, x_max = np.min(positions[:, 0]), np.max(positions[:, 0])
y_min, y_max = np.min(positions[:, 1]), np.max(positions[:, 1])
x_range = x_max - x_min
y_range = y_max - y_min
max_range = max(x_range, y_range)

x_center = (x_max + x_min) / 2
y_center = (y_max + y_min) / 2

ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
ax.set_ylim(y_center - max_range / 2, y_center + max_range / 2)

plt.show()

# plot two d
time = np.arange(0, 0.1 * len(wind), 0.1)
# Separate the values for plotting
true_wind = [w[0] for w in wind]
estimated_wind = [w[1] for w in wind]

Error = rms_error = np.sqrt(np.mean((np.array(true_wind) - np.array(estimated_wind))**2))
print("EROOR: ", Error)
# Plot
plt.plot(time, true_wind, label='True Wind')
plt.plot(time, estimated_wind, label='Estimated Wind')
plt.xlabel('Time step')
plt.ylabel('Wind Value')
plt.title('True vs Estimated Wind Over Time')
plt.legend()
plt.grid(True)
plt.show()


"""plt.plot(positions[:, 0], positions[:, 1], label='y = sin(x)')
plt.scatter(initial_position[0], initial_position[1], color='red', label="Start Position")
plt.scatter(params['IPI'][0], params['IPI'][1], color='green', label="Impact Point Indicator (IPI)")
plt.scatter(params['FTP_centre'][0], params['FTP_centre'][1], color='blue', label="Final Target Point (FTP)")
plt.title('x - y view of path')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
"""
