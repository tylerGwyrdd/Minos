import guidance
import numpy as np

def guidance_update(params,state):
    """
    Update the guidance system with the current state and parameters.
    Args:
        params (dict): A dictionary containing the parameters for the guidance system.
        state (list): The current state of the system, including position, velocity, euler angles, and angular velocity.
    
    returns:
        desired heading (float): The desired heading angle for the parafoil IN DEGREES.
        flare magnitude (float): The magnitude of the flare for the parafoil. 1 = flare, 0 = normal
    """
    # create instance of guidance class
    guide = guidance.T_approach(params)
    params = guide.get_params()
    # update it with the 
    return params, guide.update(state)