"""
We need to estimate wind from 
position vs simulated"""
import numpy as np
import logging
class wind_estimation:
    def __init__(self, radius):
        # geometry of the wind estimation path
        self.start_heading = None
        self.radius = radius
        self.initialised = False
        self.v_inertial = []

    def set_start(self, start):
        self.start_heading = start
        self.initialised = True
        return True

    def update(self, current_heading, state):
        # add to array ready for calculation
        self.v_inertial.append(state[1][:2])
        # have we done a full circle?
        if current_heading - self.start_heading > 360:
            return True
        return False

    def simple_wind_calc(self):
        """
        This function estimates the wind speed and direction based on the inertial velocities and headings.
        args:
            v_inertial: np array.  inertial velocity vector (m/s)
            heading: np array. heading vector (degrees)"""

        index = np.argmax(self.v_inertial)
        max_v = self.v_inertial[index]
        max_heading = self.heading[index]
        index = np.argmin(self.v_inertial)
        min_v = self.v_inertial[index]
        min_heading = self.v_inertial[index]
        # calculate the average wind speed and direction
        horizontal_v = 0.5*(max_v + min_v)
        wind = 0.5* (max_v - min_v)
        wind_heading = 0.5*(max_heading + min_heading + 180) % 360
        return horizontal_v, wind, wind_heading

def least_squares_wind_calc(v_inertial):
    """
    This function estimates the wind speed and direction based on the inertial velocities and headings.
    args:
        v_inertial: np array.  inertial velocity vector (m/s)
        heading: np array. heading vector (degrees)"""
    
    # calculate the average wind speed and direction
    v_inertial = np.array(v_inertial)
    v_x = v_inertial[:, 0]
    v_y = v_inertial[:, 1]
    mu_Vx = np.mean(v_x)
    mu_Vy = np.mean(v_y)
    V2 = v_x**2 + v_y**2
    mu_V2 = np.mean(V2)

    # Construct A and b
    A = np.column_stack((v_x - mu_Vx, v_y - mu_Vy))
    b = 0.5 * (V2 - mu_V2)

    # Least squares solution
    w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    logging.info(f"wind estimate: {w}")
    return w

