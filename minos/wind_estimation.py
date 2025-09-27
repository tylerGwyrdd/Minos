"""
We need to estimate wind from 
position vs simulated"""
import numpy as np
import logging
import matplotlib.pyplot as plt
from scipy.signal import bilinear, lfilter

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


class WindRLS:
    """
    Follows derivation from:
    Luo, S., Tan, P., Sun, Q., Wu, W., Luo, H., & Chen, Z. (2018). 
    In-flight wind identification and soft landing control for autonomous unmanned powered parafoils. 
    International Journal of Systems Science, 49(5), 929–946. https://doi.org/10.1080/00207721.2018.1433245

    """
    def __init__(self, lambda_=0.99, delta=1000.0):
        self.lambda_ = lambda_
        self.theta = np.zeros((2, 1))  # [V_Wx, V_Wy]
        self.P = delta * np.eye(2)
        self.Vx_prev = 0
        self.Vy_prev = 0

    def update(self, Vx_k, Vy_k):
        # Compute y(k)
        V_k = np.sqrt(Vx_k**2 + Vy_k**2)
        V_k_1 = np.sqrt(self.Vx_prev**2 + self.Vy_prev**2)
        y_k = 0.5 * (V_k**2 - V_k_1**2)

        # Compute phi(k)
        phi_k = np.array([[Vx_k - self.Vx_prev],
                          [Vy_k - self.Vy_prev]])

        # Kalman gain K(k)
        numerator = self.P @ phi_k
        denominator = self.lambda_ + (phi_k.T @ self.P @ phi_k)
        K_k = numerator / denominator

        # Update theta
        error = y_k - (phi_k.T @ self.theta)
        self.theta = self.theta + K_k * error

        # update velocity
        self.Vx_prev = Vx_k
        self.Vy_prev = Vy_k

        # Update P
        self.P = (np.eye(2) - K_k @ phi_k.T) @ self.P / self.lambda_

    def get_wind_estimate(self):
        return self.theta.flatten()




class HorizontalWindModel:
    def __init__(self, dt=0.1, U_ref=5.0, z_ref=100.0, alpha=0.15):
        self.dt = dt
        self.U_ref = U_ref
        self.z_ref = z_ref
        self.alpha = alpha

        # Pre-generate white noise
        self.n_samples = 100000
        self.noise_u = np.random.randn(self.n_samples)
        self.noise_v = np.random.randn(self.n_samples)
        self.index = 0  # step index

    def wind_at_altitude(self, z):
        # --- Mean wind using power law ---
        if z < 0:
            z = 0
        mean_speed = 0.1 + self.U_ref * (z / self.z_ref)**self.alpha
        wind_dir_rad = np.pi / 4  # 45° for example
        u_mean = mean_speed * np.cos(wind_dir_rad)
        v_mean = mean_speed * np.sin(wind_dir_rad)

        # --- set to average wind --
        self.V = mean_speed
        # --- Dryden turbulence (altitude-scaled or fixed) ---
        Lu = Lv = max(50, 0.2 * z)  # scale length increases with altitude
        sigma_u = sigma_v = min(5.0, 0.1 * z)  # intensity increases with altitude

        # Dryden transfer functions (discretized)
        num_u = [sigma_u * np.sqrt(2 * Lu / (np.pi * self.V))]
        den_u = [Lu / self.V, 1]
        bu, au = bilinear(num_u, den_u, fs=1/self.dt)

        num_v = [sigma_v * np.sqrt(3 * Lv / (np.pi * self.V)), 0]
        den_v = [(Lv / self.V)**2, 2 * (Lv / self.V), 1]
        bv, av = bilinear(num_v, den_v, fs=1/self.dt)

        # Get turbulence using current noise window
        window = slice(self.index, self.index + 100)
        u_turb = lfilter(bu, au, self.noise_u[window])[-1]
        v_turb = lfilter(bv, av, self.noise_v[window])[-1]

        # Increment index (wrap around)
        self.index = (self.index + 1) % (self.n_samples - 100)

        return np.array([u_mean + u_turb, v_mean + v_turb])


if __name__ == '__main__':
    # Create wind model instance
    wind_model = HorizontalWindModel()

    # Define altitudes to sample
    altitudes = np.linspace(5, 800, 200)
    u_values = []
    v_values = []

    # Sample wind at each altitude
    for z in altitudes:
        wind = wind_model.wind_at_altitude(z)
        u_values.append(wind[0])
        v_values.append(wind[1])

    # Convert to numpy arrays
    u_values = np.array(u_values)
    v_values = np.array(v_values)

    # Plotting
    plt.figure(figsize=(10, 5))

    # u component
    plt.subplot(1, 2, 1)
    plt.plot(u_values, altitudes, label="u (longitudinal)")
    plt.xlabel("Wind Component (m/s)")
    plt.ylabel("Altitude (m)")
    plt.title("Longitudinal Wind vs Altitude")
    plt.grid()
    plt.legend()

    # v component
    plt.subplot(1, 2, 2)
    plt.plot(v_values, altitudes, label="v (lateral)", color='orange')
    plt.xlabel("Wind Component (m/s)")
    plt.title("Lateral Wind vs Altitude")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()