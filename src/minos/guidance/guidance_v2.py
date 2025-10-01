import numpy as np

import matplotlib.pyplot as plt

def smooth_heading_to_line_with_wind(position, line_point, 
                                     line_direction, lookahead_distance, wind_vector, airspeed):
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
    """
    Returns angle within the bounds of 0 and 2 PI
    """
    return (angle + 2 * np.pi) % (2 * np.pi)

def wrap_angle_pi_to_pi(angle):
    """
    Returns angle within the bounds of -PI and PI
    """
    return (wrap_angle(angle) + np.pi) % (2 * np.pi) - np.pi

def guidance_update(params, state):
    """
    Update the state of the system based on the current state and time.
    This function is called at each time step of the simulation.
    """
    flare_magnitude = 0
    # Get the current position from the simulator
    position, current_velocity, current_height, current_heading = update_kinematics(params, state)

    #params["wind_v_list"].append(current_velocity)
    #wind_estimate = least_squares_wind_calc(params["wind_v_list"])

    # check that we will make it to the IPI
    if params["mode"] != "Final Approach":
        # easy
        vertical_time_to_IPI = (current_height - params["IPI"][2]) / params["sink_velocity"]
        # first we need to get the effective vel (including wind)
        vector_to_IPI = params["IPI"][:2] - position[:2]
        direction_to_IPI = vector_to_IPI / np.linalg.norm(vector_to_IPI)
        effective_vel = params["horizontal_velocity"] * direction_to_IPI + params["wind_magnitude"] * params["wind_unit_vector"]
        # get the effective vel componenet along the direction to the IPI
        groundspeed_along_path = np.dot(effective_vel, direction_to_IPI)
        if groundspeed_along_path <= 0:
            # we are never going to make it
            horizontal_time_to_IPI = vertical_time_to_IPI + 1
        else:
            horizontal_time_to_IPI =  np.linalg.norm(vector_to_IPI) / groundspeed_along_path

        if vertical_time_to_IPI < horizontal_time_to_IPI:
            print("heading to IPI prematurely")
            vector_to_IPI = params["IPI"][:2] - position
            # perpendicular vector to the wind direction
            params["desired_heading"] = smooth_heading_to_line_with_wind(position, params["IPI"][:2], vector_to_IPI, 
                                                                    10, params["wind_unit_vector"] *params["wind_magnitude"], params["horizontal_velocity"])
            return params["desired_heading"], flare_magnitude

    # get estimate of time until FTP height reached
    time_to_FTP = (current_height - params["final_approach_height"]) / params["sink_velocity"]
    spiral_centre_current = params["FTP_centre"] - time_to_FTP * params["wind_unit_vector"] * params["wind_magnitude"]
    
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

        params["RLS"].update(current_velocity[0],current_velocity[1])
        wind_estimate = params["RLS"].get_wind_estimate()
        update_wind(params, wind_estimate)

       # params["wind_v_list"].append(current_velocity)
       # wind_estimate = least_squares_wind_calc(params["wind_v_list"])
       # update_wind(params,wind_estimate)   
        if params["initialised"] == False:
            # set the start heading for the wind estimation
            print("SETTING STARTING HEADING")
            params["start_heading"] = current_heading
            params["initialised"] = True
        # Generate critical points for the T-approach algorithm
        # we need raw state to check if the thing has rotated over 360 degrees
        if state[2][2] - params["start_heading"] > np.deg2rad(360):
            print("Generating critical points")
            # get the wind estimate
            #wind_estimate = least_squares_wind_calc(params["wind_v_list"])
            #update_wind(params,wind_estimate)
            # update the FTP centre
            params["mode"] = "homing"
            position, current_velocity, current_height, current_heading = update_kinematics(params, state)
        else:
            print("Initialising")
            # params["wind_v_list"].append(current_velocity)
            # keep going in a circle
            angular_vel = params["horizontal_velocity"] / params["spirialing_radius"]
            delta_heading = angular_vel * params["update_rate"]
            # go clockwise, add onto desired heading
            params["desired_heading"] += delta_heading


    # initialising mode could've been set to homing mode in the last update, better check
    if params["mode"] == "homing":
        print("Homing")

        params["RLS"].update(current_velocity[0],current_velocity[1])
        wind_estimate = params["RLS"].get_wind_estimate()
        update_wind(params, wind_estimate)

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
            params["mode"] = "energy_management"

    # homing mode could've been set to energy management mode in the last update, better check
    if params["mode"] == "energy_management":
            
            print("Energy Management")
            dist_to_CTP = np.linalg.norm(np.subtract(position, spiral_centre_current))
            if dist_to_CTP > 5 * params["spirialing_radius"]:
                params["mode"] = "homing"
            #params["wind_v_list"].append(current_velocity)
           # wind_estimate = least_squares_wind_calc(params["wind_v_list"])
            #update_wind(params,wind_estimate)
            # keep going in a circle
            angular_vel = params["horizontal_velocity"] / params["spirialing_radius"]
            delta_heading = angular_vel * params["update_rate"]
            # go clockwise, add onto desired heading
            params["desired_heading"] += delta_heading
    params["desired_heading"] = wrap_angle(params["desired_heading"])
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
    if params["wind_magnitude"] < 0.3:
        params["wind_magnitude"] = 0.0
        params["wind_unit_vector"] = np.array([1, 0])
        params["wind_heading"] = 0.0

def update_kinematics(params, state):
    """
    Update the kinematics of the system based on the current state.
    """
    position = state[0][:2]
    current_height = state[0][2]
    current_velocity = state[1][:2]
    current_heading = wrap_angle(state[2][2])
    if params["mode"] != "initialising":
        time = params["final_approach_height"] / params["sink_velocity"]
        params["FTP_centre"] = params["IPI"][:2] + (params["horizontal_velocity"] - params["wind_magnitude"])  * time * params["wind_unit_vector"]
    return position, current_velocity, current_height, current_heading

def compute_required_heading(wind_vector, airspeed, target_vector):
    """
    Calculate the required heading angle to follow the desired vector in wind.
    Uses wind triangle, taking norms and solving quadratic for ground speed along 
    desired vector. 
    Args:
        wind_vector (list): (w_x, w_y), wind vector in m/s.
        airspeed (float): Airspeed of the parafoil  in m/s.
        target_vector (list): (d_x, d_y), vector pointing to the desired target.

    Returns
    -------
        float: Required heading angle in radians.
        np.ndarray: Air velocity vector (v_a_x, v_a_y).
    """
    # format and normalise
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

def simple_control(current_heading, desired_heading, heading_rate):
    """
    Takes both headings in radians, returns flap deflections
    """
    # convert and calcuate requred headings
    current_heading = wrap_angle_pi_to_pi(current_heading)
    desired_heading = wrap_angle_pi_to_pi(desired_heading)
    heading_error = wrap_angle_pi_to_pi(desired_heading - current_heading)
    print(f"current heading: {np.rad2deg(current_heading)}, desired heading: {np.rad2deg(desired_heading)}, heading error: {np.rad2deg(heading_error)}")
    # double check that heading error is correct
    # convert to degrees for easibility
    heading_error = np.rad2deg(heading_error)
    deflections = [0,0]
    magnitude = 0.5
    if heading_error < 0:
        deflections = [1,0]
    else:
        deflections = [0,1]
    if abs(heading_error) < 1:
        magnitude = 0
    elif abs(heading_error) < 3:
        magnitude = 0.1
    elif abs(heading_error) < 5:
        magnitude = 0.2
    elif abs(heading_error) < 10:
        magnitude = 0.3
    elif abs(heading_error) < 30:
        magnitude = 0.4
    return [magnitude * deflections[0], magnitude * deflections[1]]
        
def PID_control(current_heading, desired_heading, heading_rate):
    # Normalize angles
    current_heading = wrap_angle_pi_to_pi(current_heading)
    desired_heading = wrap_angle_pi_to_pi(desired_heading)
    heading_error = wrap_angle_pi_to_pi(desired_heading - current_heading)

    print(f"current heading: {np.rad2deg(current_heading):.2f}°, desired heading: {np.rad2deg(desired_heading):.2f}°, heading error: {np.rad2deg(heading_error):.2f}°, heading rate: {heading_rate}")

    # PD control
    Kp = 3
    Kd = 4
    control_effort = Kp * heading_error - Kd * heading_rate
    print(f"control effort: {control_effort}")
    # Convert control effort to flap values
    max_deflection = 0.6  # full range
    control_effort = np.clip(control_effort, -max_deflection, max_deflection)

    if control_effort > 0:
        # turn right
        left_flap = 0.0
        right_flap = control_effort
    else:
        # Turn left
        left_flap = -control_effort
        right_flap = 0.0
    print(left_flap,right_flap)
    return [left_flap, right_flap]

def ideal_guidance(params, actual_wind, inital_state, max_turn_rate, max_steps): 
    print("ideal guidance calculating.....")
    print(params['mode'])
    # for plotting
    positions = [inital_state[0]]
    print("inital pos:",inital_state[0])
    velocities = [inital_state[1]]
    # Simulate path
    max_turn_rate = np.deg2rad(max_turn_rate)
    num_steps = max_steps
    dt = params['update_rate']  # time step
    state = inital_state
    for _ in range(num_steps):
        # stop sim if we have landed on ground
        if state[0][2] < 0:
            print("IDEAL SIM COMPLETE")
            print(f"Final Position: {[f'{coord:.3g}' for coord in state[0]]}")
            IPI_error = params['IPI'][:2] - state[0][:2]
            print(f"Final IPI Error: {[f'{e:.3g}' for e in IPI_error]}")
            estimated_wind_vector = params['wind_unit_vector'] * params['wind_magnitude']
            print(f"Estimated Wind: {[f'{w:.3g}' for w in estimated_wind_vector]}")
            print(f"Actual Wind: {[f'{w:.3g}' for w in actual_wind]}")
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
       #print(f"Heading Error: {np.degrees(heading_error)}, Current Heading: {np.degrees(wrap_angle(current_heading))}, Desired Heading: {np.degrees(wrap_angle(desired_heading))}, New Heading: {np.degrees(wrap_angle(new_heading))}")
        
        # ================= Motion update ===================
        # Move in the direction of the current heading + wind
        vel_x = params['horizontal_velocity'] * np.cos(new_heading) + actual_wind[0]
        vel_y = params['horizontal_velocity'] * np.sin(new_heading) + actual_wind[1]
        vel_z = params['sink_velocity']  # Assuming constant sink velocity
        new_x = vel_x * dt + state[0][0]  # Update x position
        new_y = vel_y * dt + state[0][1]  # Update y position
        new_z = state[0][2] - vel_z * dt  # Update z position
        #print(f"New Position: {new_x}, {new_y}, {new_z}")
        #print(f"New Velocity: {vel_x}, {vel_y}, {vel_z}")
        state = [np.array([new_x,new_y,new_z]), 
                np.array([vel_x,vel_y,vel_z]), 
                np.array([0,0,new_heading])
                ]  # Update state
        positions.append(state[0])  # Store position for plotting
    return np.array(positions)

def plot_3D_position(inertial_positions, guidance_params, ideal_positions = None, ideal_params = None):
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # plotting
    ax.plot(inertial_positions[:, 0], inertial_positions[:, 1], inertial_positions[:, 2], label="actual Parafoil Path", color='gray', linewidth=2)
    if ideal_positions is not None:
        ax.plot(ideal_positions[:, 0], ideal_positions[:, 1], ideal_positions[:, 2], label="ideal Parafoil Path", color='brown', linewidth=2)
    
    ax.scatter(guidance_params["deployment_pos"][0], guidance_params["deployment_pos"][1], guidance_params["deployment_pos"][2], color='red', label="Start Position")
    ax.scatter(guidance_params['IPI'][0], guidance_params['IPI'][1], guidance_params['IPI'][2], color='green', label="Impact Point Indicator (IPI)")
    ax.scatter(guidance_params['FTP_centre'][0], guidance_params['FTP_centre'][1], guidance_params['final_approach_height'], color='blue', label="Final Target Point (FTP)")
    wind_line = guidance_params['IPI'] + np.array([guidance_params['wind_unit_vector'][0] * 100,guidance_params['wind_unit_vector'][1] * 100, 0])
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