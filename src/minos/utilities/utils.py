import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button

def safe_clamp_vector(self, vec, max_abs=1e3):
    """
    Clamp a 3D vector to avoid NaN, inf, and excessively large values.

    Parameters
    ----------
        vec (np.ndarray): Input 3D vector.
        max_abs (float): Maximum allowed absolute value for each component.

    Returns
    -------
        np.ndarray: A safe, clamped 3D vector.
    """
    safe_vec = np.zeros(3)

    for i in range(3):
        val = vec[i]
        if not np.isfinite(val):
            #print("clamping")
            self.error = True
            safe_vec[i] = 0.0
        elif abs(val) > max_abs:
            #print()
            self.error = True
            safe_vec[i] = np.clip(val, -max_abs, max_abs)
        else:
            safe_vec[i] = val

    return safe_vec