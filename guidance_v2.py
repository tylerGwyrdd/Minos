import numpy as np
import logging
from wind_estimation import least_squares_wind_calc
import matplotlib.pyplot as plt


def smooth_heading_to_line_with_wind(position, line_point, line_direction, lookahead_distance, wind_vector, airspeed):
    """
    Compute the heading required to move toward the lookahead point, considering wind.
    """
    p = np.array(position)
    a = np.array(line_point)
    d = np.array(line_direction)
    d = d / np.linalg.norm(d)

    # Project position onto the line to find the closest point
    ap = p - a
    t = np.dot(ap, d)
    closest_point = a + t * d

    # Lookahead point on the line
    lookahead_point = closest_point + d * lookahead_distance

    # Desired ground track vector
    desired_track = lookahead_point - p
    desired_track /= np.linalg.norm(desired_track)

    # Solve for air vector that, when combined with wind, gives the desired track
    air_vector = desired_track * airspeed - wind_vector

    # Compute heading from air vector
    required_heading = np.arctan2(air_vector[1], air_vector[0])
    return required_heading

def wrap_angle(angle):
    return (angle + 2 * np.pi) % (2 * np.pi)

def guidance_update(params, state):
    """
    Update the state of the system based on the current state and time.
    This function is called at each time step of the simulation.
    """
    flare_magnitude = 0
    # Get the current position from the simulator
    position, current_height, current_heading = update_kinematics(params, state)
    
    # check that we will make it to the IPI
    if params["mode"] != "Final Approach":
        vertical_time_to_IPI = (current_height - params["IPI"][2]) / params["sink_velocity"]
        horizontal_time_to_IPI = np.linalg.norm(params["IPI"][:2] - position[:2])/params["horizontal_velocity"]
        if vertical_time_to_IPI < horizontal_time_to_IPI:
            print("heading to IPI prematurely")
            vector_to_IPI = params["IPI"][:2] - position
            # perpendicular vector to the wind direction
            params["desired_heading"] = smooth_heading_to_line_with_wind(position, params["IPI"][:2], vector_to_IPI, 
                                                                    10, params["wind_unit_vector"] *params["wind_magnitude"], params["horizontal_velocity"])
            return params["desired_heading"], flare_magnitude

    # get estimate of time until FTP height reached
    time_to_FTP = (current_height - params["final_approach_height"]) / params["sink_velocity"]
    if time_to_FTP  < 0:
        # we have hit FTP
        params["mode"] = "Final Approach"
    
    if params["mode"] == "Final Approach":
        print("Final Approach")
        # line up with the wind...
        params["desired_heading"] = smooth_heading_to_line_with_wind(position, params["FTP_centre"], - params["wind_unit_vector"], 
                                                                10, params["wind_unit_vector"] *params["wind_magnitude"], params["horizontal_velocity"])
        # flare at 10m
        if(current_height < params["flare_height"]):
            flare_magnitude = 1

    elif params["mode"] == "initialising":
        if params["initialised"] == False:
            # set the start heading for the wind estimation
            print("SETTING STARTING HEADING")
            params["start_heading"] = current_heading
            params["initialised"] = True
        # Generate critical points for the T-approach algorithm
        if current_heading - params["start_heading"] > np.deg2rad(360):
            print("Generating critical points")
            # get the wind estimate
            wind_estimate = least_squares_wind_calc(params["wind_v_list"])
            update_wind(params,wind_estimate)
            # update the FTP centre
            params["mode"] = "homing"
            update_kinematics(params, state)
            
        else:
            print("Initialising")
            params["wind_v_list"].append(state[1][:2])
            # keep going in a circle
            angular_vel = params["horizontal_velocity"] / params["spirialing_radius"]
            delta_heading = angular_vel * params["update_rate"]
            # go clockwise, add onto desired heading
            params["desired_heading"] += delta_heading


    # initialising mode could've been set to homing mode in the last update, better check
    if params["mode"] == "homing":
        print("Homing")
        # calculate where the spiral centre is
        spiral_centre_current = params["FTP_centre"] - time_to_FTP * params["wind_unit_vector"] * params["wind_magnitude"]
        # work out the distance and heading to get to this point
        vector_to_centre = spiral_centre_current - position
        # perpendicular vector to the wind direction
        perp_vector = np.array([vector_to_centre[1], -vector_to_centre[0]])
        distance_to_tangent = np.linalg.norm(vector_to_centre)
        # add perp vector so we line up with the circle
        vector_to_tangent = spiral_centre_current + perp_vector / np.linalg.norm(perp_vector) * params["spirialing_radius"] * 0.9 - position
        # calculate the desired heading
        params["desired_heading"],_ = compute_required_heading(params["wind_unit_vector"] * params["wind_magnitude"], params["horizontal_velocity"], vector_to_tangent)
        # calculate the heading
        params["desired_heading"] = wrap_angle(params["desired_heading"])

        # Extract needed values
        spiral_centre_current = spiral_centre_current
        ipi_point = params["IPI"][:2]  # Assuming IPI is a 3D point and we only need the 2D position

        """        # Plot setup
        plt.figure(figsize=(8, 8))
        plt.axis('equal')
        plt.grid(True)
        plt.title("Parafoil Homing Guidance")

        # Plot parafoil position
        plt.plot(position[0], position[1], 'bo', label='Parafoil Position')

        # Plot current heading vector
        current_heading_vector = np.array([
            np.cos(current_heading),
            np.sin(current_heading)
        ]) * 10  # scale for visibility
        plt.arrow(position[0], position[1],
                current_heading_vector[0], current_heading_vector[1],
                head_width=1, color='blue', label='Current Heading')

        # Plot desired heading vector
        desired_heading_vector = np.array([
            np.cos(params["desired_heading"]),
            np.sin(params["desired_heading"])
        ]) * 10  # scale for visibility
        plt.arrow(position[0], position[1],
                desired_heading_vector[0], desired_heading_vector[1],
                head_width=1, color='green', label='Desired Heading')

        # plot wind
        wind_vector = params["wind_unit_vector"] * 10
        plt.arrow(position[0], position[1],
                wind_vector[0], wind_vector[1],
                head_width=1, color='yellow', label='wind vector')
        # plot actual velocity
        plt.arrow(position[0], position[0], 
                state[1][0], state[1][1],
                head_width=1, color='orange', label='actual velocity vector')
        # Plot spiral centre
        plt.plot(spiral_centre_current[0], spiral_centre_current[1], 'rx', label='Spiral Centre')

        # Plot IPI point (assumed FTP_centre)
        plt.plot(ipi_point[0], ipi_point[1], 'ms', label='IPI (FTP Centre)')

        # Add legend and labels
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        # plt.show()"""

        # Check if we need to start turning into the final approach
        if distance_to_tangent < 1.2 * params["spirialing_radius"]:
            print("Entering final approach mode")
            params["mode"] = "energy_management"

    # homing mode could've been set to energy management mode in the last update, better check
    if params["mode"] == "energy_management":
            print("Energy Management")
            # keep going in a circle
            angular_vel = params["horizontal_velocity"] / params["spirialing_radius"]
            delta_heading = angular_vel * params["update_rate"]
            # go clockwise, add onto desired heading
            params["desired_heading"] += delta_heading
    return params["desired_heading"], flare_magnitude

def update_wind(params, wind_vector):
    """
    Update the wind vector based on the current state.
    This function is called at each time step of the simulation.
    """
    # Update the wind vector based on the current state
    wind_2d = np.array([wind_vector[0], wind_vector[1]])
    # Update the wind vector based on the current state
    params["wind_magnitude"] = np.linalg.norm(wind_2d)
    params["wind_unit_vector"] = wind_2d / params["wind_magnitude"]
    params["wind_heading"] = np.arctan2(wind_2d[1], wind_2d[0])
    if params["wind_magnitude"] < 0.5:
        params["wind_magnitude"] = 0.0
        params["wind_unit_vector"] = np.array([1, 0])
        params["wind_heading"] = 0.0

def update_kinematics(params, state):
    """
    Update the kinematics of the system based on the current state.
    """
    position = state[0][:2]
    current_height = state[0][2]
    current_heading = state[2][2]
    if params["mode"] != "initialising":
        time = params["final_approach_height"] / params["sink_velocity"]
        params["FTP_centre"] = params["IPI"][:2] + (params["horizontal_velocity"] - params["wind_magnitude"])  * time * params["wind_unit_vector"] - params["wind_unit_vector"] * 2 * params["spirialing_radius"]
    return position, current_height, current_heading

def compute_required_heading(wind_vector, airspeed, target_vector):
    """
    Calculate the required heading angle to follow the desired vector in wind.

    Args:
        wind_vector (tuple or list): (w_x, w_y), wind vector in m/s.
        airspeed (float): Airspeed of the parafoil in m/s.
        target_vector (tuple or list): (d_x, d_y), vector pointing to the desired target.

    Returns:
        float: Required heading angle in radians (0 is east, pi/2 is north).
        np.ndarray: Air velocity vector (v_a_x, v_a_y).
    """
    w = np.array(wind_vector)
    d = np.array(target_vector)
    d_hat = d / np.linalg.norm(d)

    # Solve for Vg using quadratic equation
    a = 1
    b = -2 * np.dot(d_hat, w)
    c = np.dot(w, w) - airspeed**2

    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        raise ValueError("No solution: desired vector cannot be achieved with given airspeed and wind.")

    Vg = (-b + np.sqrt(discriminant)) / (2 * a)

    # Compute air velocity vector
    v_g = Vg * d_hat
    v_a = v_g - w

    # Compute heading angle
    heading = np.arctan2(v_a[1], v_a[0])

    return heading, v_a

def get_estimated_path(params, current_heading, actual_wind_vector, ideal_positions, dt):
    vel_x = params['horizontal_velocity'] * np.cos(current_heading) + actual_wind_vector[0]
    vel_y = params['horizontal_velocity'] * np.sin(current_heading) + actual_wind_vector[1]
    vel_z = params['sink_velocity']  # Assuming constant sink velocity
    new_x = vel_x * dt + ideal_positions[-1][0]  # Update x position
    new_y = vel_y * dt + ideal_positions[-1][1]  # Update y position
    new_z = ideal_positions[0][2] - vel_z * dt  # Update z position
    print(f"New Position: {new_x}, {new_y}, {new_z}")
    print(f"New Velocity: {vel_x}, {vel_y}, {vel_z}")
    ideal_positions.append(np.array([new_x,new_y,new_z]))  # Store position for plotting


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def plot_3D_position(inertial_positions,guidance_params,estimated_path = None):
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
    wind_line = guidance_params['IPI'] + np.array([guidance_params['wind_unit_vector'][0] * 100,guidance_params['wind_unit_vector'][1] * 100, 0])
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