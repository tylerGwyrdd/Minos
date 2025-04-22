import six_DoF_simulator as sim
from wind_estimation import least_squares_wind_calc
import numpy as np
import logging

class T_approach:
    def __init__(self, params):
        # geometry ideal path
        IPI = params["IPI"] # meters
        self.IPI = np.array([IPI[0],IPI[1]])
        self.IPI_height = IPI[2]
        self.spirialing_radius = params["spiral_radius"] # meters
        self.final_approach_height = params["final_approach_height"]  # meters
        self.FTP_centre = []
        self.flare_height = params["flare_height"] # meters
        self.update_rate = params["update_rate"] # seconds

        # estimated from body velocities in sim - for ALEXPADS model
        self.horizontal_velocity = params["horizontal_velocity"]
        self.sink_velocity = params["sink_velocity"]

        # wind stuff
        self.wind_heading = 0.0
        self.wind_unit_vector = params["wind_unit_vector"]
        self.wind_magnitude = params["wind_magnitude"]
        self.wind_v_list = params["wind_v_list"]

        # general locals
        self.initialised = params["initialised"]
        self.start_heading = params["start_heading"]
        self.mode = params["mode"] # initalising, homing, final approach, energy management
        self.flare = False
        self.position = np.zeros(2)
        self.current_height = 0
        self.init_pos = params["deployment_pos"]

        # outputs
        self.desired_heading = 0
        self.flare_magnitude = 0.0

    def get_params(self):
        """
        Get the parameters of the T-approach algorithm.
        This function is called at each time step of the simulation.
        """
        return {
            'deployment_pos': self.init_pos,
            'final_approach_height': self.final_approach_height,
            'spiral_radius': self.spirialing_radius,
            'update_rate': self.update_rate,
            'wind_unit_vector': self.wind_unit_vector,
            'wind_magnitude': self.wind_magnitude,
            'wind_v_list': self.wind_v_list,
            'horizontal_velocity': self.horizontal_velocity,
            'sink_velocity': self.sink_velocity,
            'IPI': [self.IPI[0], self.IPI[1], self.IPI_height],
            'flare_height': self.flare_height,
            'initialised': self.initialised,
            'mode': self.mode,  
            'start_heading': self.start_heading,  
        }
    
    def generate_ciritical_point(self):
        """
        Generate critical points for the T-approach algorithm.
        This function is called once at the start of the simulation.
        """
        # time required to land
        time_to_land = self.final_approach_height / self.sink_velocity

        # calculate FTP centre
        self.FTP_centre = self.position + time_to_land * self.wind_magnitude * self.wind_unit_vector
        
    def update_wind(self, wind_vector):
        """
        Update the wind vector based on the current state.
        This function is called at each time step of the simulation.
        """
        # Update the wind vector based on the current state
        wind_2d = np.array([wind_vector[0], wind_vector[1]])
        # Update the wind vector based on the current state
        self.wind_magnitude = np.linalg.norm(wind_2d)
        self.wind_unit_vector = wind_2d / self.wind_magnitude
        self.wind_heading = np.arctan2(wind_2d[1], wind_2d[0])
        if self.wind_magnitude < 0.5:
            self.wind_magnitude = 0.0
            self.wind_unit_vector = np.array([1, 0])
            self.wind_heading = 0.0

    def update_kinematics(self,state):
        """
        Update the kinematics of the system based on the current state.
        This function is called at each time step of the simulation.
        """
        # Get the current position and velocity from the simulator

        self.position = state[0][:2]
        self.current_height = state[0][2]
        self.current_heading = state[2][2]
        if self.mode != "initialising":
            time = self.final_approach_height / self.sink_velocity
            self.FTP_centre = self.IPI + (self.horizontal_velocity - self.wind_magnitude) * time * self.wind_unit_vector
    
    def update(self, state):
        """
        Update the state of the system based on the current state and time.
        This function is called at each time step of the simulation.
        """
        # Get the current position from the simulator
        self.update_kinematics(state)
        # get estimate of time until FTP height reached
        time_to_FTP = (self.current_height - self.final_approach_height) / self.sink_velocity
        if time_to_FTP < 0:
            logging.info(f"entering final approach mode")
            # we have hit FTP
            self.mode = "Final Approach"
        
        if self.mode == "Final Approach":
            # line up with the wind...
            self.desired_heading = smooth_heading_to_line_with_wind(self.position, self.FTP_centre, self.wind_unit_vector, 
                                                                    10, self.wind_unit_vector *self.wind_magnitude, self.horizontal_velocity)
            # flare at 10m
            if(self.current_height < self.flare_height):
                self.flare = True

        elif self.mode == "initialising":
            if self.initialised == False:
                # set the start heading for the wind estimation
                print("SETTING STARTING HEADING")
                self.start_heading = self.current_heading
                self.initialised = True
            # Generate critical points for the T-approach algorithm
            if self.current_heading - self.start_heading > np.deg2rad(360):
                print("Generating critical points")
                # get the wind estimate
                wind_estimate = least_squares_wind_calc(self.wind_v_list)
                self.update_wind(wind_estimate)
                # update the FTP centre
                self.mode = "homing"
                self.update_kinematics(state)
                
            else:
                self.wind_v_list.append(state[1][:2])
                # keep going in a circle
                angular_vel = self.horizontal_velocity / self.spirialing_radius
                delta_heading = angular_vel * self.update_rate
                # go clockwise, add onto desired heading
                self.desired_heading += delta_heading
                print(f"desired heading: {np.degrees(self.desired_heading)}")

        # initialising mode could've been set to homing mode in the last update, better check
        if self.mode == "homing":
            # calculate where the spiral centre is
            spiral_centre_current = self.FTP_centre - time_to_FTP * self.wind_unit_vector * self.wind_magnitude
            # work out the distance and heading to get to this point
            vector_to_centre = spiral_centre_current - self.position
            # perpendicular vector to the wind direction
            perp_vector = np.array([-vector_to_centre[1], vector_to_centre[0]])
            # add perp vector so we line up with the circle
            vector_to_tangent = spiral_centre_current + perp_vector / np.linalg.norm(perp_vector) * self.spirialing_radius * 0.9 - self.position
            distance_to_target = np.linalg.norm(vector_to_centre)
            # calculate the heading to get to this point
            self.desired_heading = self.wind_heading + np.arctan2(vector_to_tangent[1], vector_to_tangent[0])
            # Check if we need to start turning into the final approach
            if distance_to_target < 1.2 * self.spirialing_radius:
                logging.info("Distance to target is within the final approach radius.")
                self.mode = "energy_management"

        # homing mode could've been set to energy management mode in the last update, better check
        if self.mode == "energy_management":
            # update heading set amount depending on frequency
            angular_vel = self.horizontal_velocity / self.spirialing_radius
            delta_heading = angular_vel * self.update_rate
            # go clockwise, add onto desired heading
            self.desired_heading += delta_heading
        return self.desired_heading, self.flare_magnitude
class Control:
    def __init__(self):
        pass
    def simple_heading(self, current_flaps, current_heading, desired_heading, dt):
        kp = 0.05

        heading_error = (desired_heading - current_heading + 180) % 360 - 180
        self.prev_heading_error = heading_error
        if abs(heading_error) < 2:
            heading_error = 0
        control_input = kp * heading_error
        left_flap = current_flaps[0]
        right_flap = current_flaps[1]
        if heading_error > 0:
            if left_flap > 0:
                left_flap -= control_input
            else:
                right_flap += control_input
        else:
            if right_flap > 0:
                right_flap -= control_input
            else:
                left_flap += control_input
        # Limit the flap deflection to a maximum value
        left_flap = np.clip(left_flap, 0, 0.6)
        right_flap = np.clip(right_flap, 0, 0.6)
        return [left_flap, right_flap]

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
    return (angle + np.pi) % (2 * np.pi) - np.pi