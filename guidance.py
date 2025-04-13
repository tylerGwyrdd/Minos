import six_DoF_simulator as sim
import wind_estimation as wind
import numpy as np

class T_approach:
    def __init__(self, sim, IPI, start):
        self.sim = sim
        self.IPI = IPI
        self.start = start
        self.spirialing_radius = 20
        self.spirialing_offset = [1,2,3]
        self.spirialing_centre = []
        self.mode = "homing"
        self.wp = []
        self.horizontal_velocity = 0.0
        self.downward_velocity = 0.0
        self.wind_magnitude = 0.0
        """
        uses t approach algorithm
        """
        def generate_ciritical_points():
            """
            Generate critical points for the T-approach algorithm.
            This function is called once at the start of the simulation.
            """
            # first lets decide at what point we need to start final approach
            length = 10
            self.start_final_approach = [(self.horizontal_velocity + self.wind_magnitude) * length, 0, self.downward_velocity * length]
        def path():
            """
            Calculate the heading to the target point.
            This function is called at each time step of the simulation.
            """
            # Get the current position from the simulator
            position = self.sim.get_position()
            # Calculate the heading to the target point
            heading = np.arctan2(target[1] - position[1], target[0] - position[0])
            return heading
        def update_downward_velocity():
            """
            updates dawnward velocity based on the current position and velocity.
            """

            return downward_velocity
        def wp_update(self, state):
            """
            Update the state of the system based on the current state and time.
            This function is called at each time step of the simulation.
            """
            # Get the current position and velocity from the simulator
            position = self.sim.get_position()
            velocity = self.sim.get_velocity()
            if self.mode == "initialising": 
                # Generate critical points for the T-approach algorithm
                initalised = generate_ciritical_points()
                if generate_ciritical_points() == 1:
                    self.mode = "homing"
                elif initalised == 0:
                    self.mode = "initialising"
                
            if self.mode == "homing":
                            # Calculate the distance to the target
                distance_to_target = np.linalg.norm(position - self.spirialing_centre)
                # Check if the distance is within the final approach radius
                if distance_to_target < self.final_approach_radius:
                    self.mode = "energy_management"
            if self.mode == "energy_management":
                # Calculate the distance to the target
                distance_to_target = np.linalg.norm(position - self.spirialing_centre)
                # Check if the distance is within the final approach radius
                if distance_to_target < self.final_approach_radius:
                    self.mode = "homing"

        return self.wp
                
                
            



if __name__ == "__main__":
    # Example usage
    sim = None  # Replace with actual simulator instance
    IPI = None  # Replace with actual IPI instance
    start = 0.0  # Replace with actual start time

    guidance_instance = guidance(sim, IPI, start)
    # Call methods on guidance_instance as needed