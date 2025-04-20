import six_DoF_simulator as sim
import wind_estimation as wind_estimator
import numpy as np

class T_approach:
    def __init__(self, IPI,update_rate):
        # geometry ideal path
        self.IPI = IPI
        self.spirialing_radius = 20 # meters
        self.final_approach_height = 50 # meters
        self.FTP_centre = []
        self.flare_height = 10 # meters
        self.update_rate = update_rate # seconds

        # estimated from body velocities in sim - for ALEXPADS model
        self.horizontal_velocity = 6
        self.sink_velocity = 5.5

        # wind stuff
        self.wind = wind_estimator.wind_estimation(self.spirialing_radius)
        self.wind_heading = 0.0
        self.wind_unit_vector = np.zeros(3)
        self.wind_magnitude = 0.0

        # general locals
        self.mode = "homing"
        self.flare = False
        self.position = np.zeros(3)
        self.current_height = 0
        
        # outputs
        self.desired_heading = 0
        self.flare_magnitude = 0.0


    def generate_ciritical_point(self):
        """
        Generate critical points for the T-approach algorithm.
        This function is called once at the start of the simulation.
        """
        # time required to land
        time_to_land = self.final_approach_height / self.sink_velocity

        # calculate FTP centre
        self.FTP_centre = self.position + time_to_land * self.wind_magnitude * self.wind_vector
        
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

    def update_kinematics(self,state):
        """
        Update the kinematics of the system based on the current state.
        This function is called at each time step of the simulation.
        """
        # Get the current position and velocity from the simulator
        self.position = state[0]
        self.current_height = state[2]
        
    def update(self, state):
        """
        Update the state of the system based on the current state and time.
        This function is called at each time step of the simulation.
        """
        # Get the current position from the simulator
        self.update_kinematics(state)

        # get estimate of time until FTP height reached
        time_to_FTP = (self.position[2] - self.final_approach_height)/self.sink_velocity
        
        if time_to_FTP < 0:
            # we have hit FTP
            self.mode == "Final Approach"
        
        if self.mode == "Final Approach":
            # line up with the wind...
            self.desired_heading = 0
            # flare at 10m
            if(self.position[2] < self.flare_height):
                self.flare = True
        elif self.mode == "initialising": 
            # Generate critical points for the T-approach algorithm
            if self.wind.update(state):
                # get the wind estimate
                wind_estimate = self.wind.least_squares_wind_calc()
                self.update_wind(wind_estimate)
                # now generate wind frame of reference
                self.mode = "homing"
            else:
                # keep going in a circle
                angular_vel = self.horizontal_velocity / self.spirialing_radius
                delta_heading = angular_vel * self.update_rate
                # go clockwise, add onto desired heading
                self.desired_heading += delta_heading

        # initialising mode could've been set to homing mode in the last update, better check
        if self.mode == "homing":
            # calculate where the spiral centre is
            spiral_centre_current = self.position + time_to_FTP * self.wind_unit_vector * self.wind_magnitude
            # work out the distance and heading to get to this point
            distance_to_target = np.linalg.norm(spiral_centre_current - self.position)
            # calculate the heading to get to this point
            self.desired_heading = self.wind_heading + np.arctan2(spiral_centre_current - self.position)
            # Check if the distance is within the final approach radius
            if distance_to_target < self.spirialing_radius:
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
    def simple_heading(current_flaps, current_heading, desired_heading, dt):
        gain = 10
        # calculate the heading error
        heading_error = desired_heading - current_heading
        # calculate the control input
        control_input = 0.5 * gain * heading_error * dt
        # calculate control output
        left_flap = np.clip( current_flaps[0] - control_input, 0, 1)
        right_flap = np.clip( current_flaps[1] + control_input, 0, 1)
        return [left_flap, right_flap]

if __name__ == "__main__":
    # Example usage
    sim = None  # Replace with actual simulator instance
    IPI = None  # Replace with actual IPI instance
    start = 0.0  # Replace with actual start time

    guidance = T_approach(sim, [0,20,0], start)

    # Call methods on guidance_instance as needed