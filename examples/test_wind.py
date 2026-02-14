""" import numpy as np
from scipy.signal import bilinear, lfilter
import matplotlib.pyplot as plt

# --- Configuration ---
U_ref = 5.0       # wind speed at reference height (m/s)
z_ref = 200      # reference altitude (m)
alpha = 0.15      # wind shear exponent (flat terrain ~0.15)

altitudes = np.linspace(5, 800, 100)  # Altitude range (m)
dt = 0.01         # Time step for Dryden filter
n_turb_samples = 100  # How many time steps to simulate turbulence
wind_dir_deg = 45     # Mean wind direction in degrees

# Direction unit vector
theta = np.radians(wind_dir_deg)
wind_unit = np.array([np.cos(theta), np.sin(theta)])

# Store wind vectors
u_all = []
v_all = []

for z in altitudes:
    # --- Mean wind from power law ---
    mean_speed = U_ref * (z / z_ref)**alpha
    u_mean = mean_speed * wind_unit[0]
    v_mean = mean_speed * wind_unit[1]

    # --- Altitude-dependent Dryden parameters ---
    V = mean_speed if mean_speed > 0.1 else 1.0
    Lu = Lv = max(50.0, 0.2 * z)
    sigma_u = sigma_v = min(5.0, 0.1 * z)

    # Dryden filters (discrete)
    num_u = [sigma_u * np.sqrt(2 * Lu / (np.pi * V))]
    den_u = [Lu / V, 1]
    bu, au = bilinear(num_u, den_u, fs=1/dt)

    num_v = [sigma_v * np.sqrt(3 * Lv / (np.pi * V)), 0]
    den_v = [(Lv / V)**2, 2 * (Lv / V), 1]
    bv, av = bilinear(num_v, den_v, fs=1/dt)

    # Generate turbulence (last sample is used)
    noise_u = np.random.randn(n_turb_samples)
    noise_v = np.random.randn(n_turb_samples)
    u_turb = lfilter(bu, au, noise_u)[-1]
    v_turb = lfilter(bv, av, noise_v)[-1]

    u_all.append(u_mean + u_turb)
    v_all.append(v_mean + v_turb)

u_all = np.array(u_all)
v_all = np.array(v_all)
wind_speed = np.sqrt(u_all**2 + v_all**2)
wind_direction = np.degrees(np.arctan2(v_all, u_all))

# --- Plot wind speed vs altitude ---
plt.figure()
plt.plot(wind_speed, altitudes)
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Altitude (m)")
plt.title("Wind Speed vs Altitude (with Dryden Turbulence)")
plt.grid()

# --- Plot wind direction vs altitude ---
plt.figure()
plt.plot(wind_direction, altitudes)
plt.xlabel("Wind Direction (deg from X)")
plt.ylabel("Altitude (m)")
plt.title("Wind Direction vs Altitude (with Dryden Turbulence)")
plt.grid()

plt.show() """