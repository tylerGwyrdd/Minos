
# ODE solver, simplest one, error prone
def forward_euler(state, derivatives, dt):
    """
    'OLD'

    Simplist ODE solver. Not recomended to be used. Introduces alot of error.
    Use RK4 Method.

    Parameters
    ----------
    state : list
        state vector    [position, velocity_body, eulers, angular_velocity].
    
    Derivites : list
        The derivities of the state vector. 
        [velocity_body, acceleration_body, euler_rates, angular_acceleration].
    dt : float
        the time period to use. e.g 0.01 - 0.1
    
    Returns
    -------
    new_state : list
        The updated state with the added effects of the derivities applied over dt.
    """
    dx = [dt * der for der in derivatives]
    new_state = [s + a for s, a in zip(state, dx)]
    return new_state

# RK4 is a more accurate ODE solver
def rk4(state, derivative_func, dt):
    """
    Runge-Kutta 4th order integrator. 
    
    A good, accurate ODE solver.

    Parameters
    ----------
    state : list
        state vector    [position, velocity_body, eulers, angular_velocity].
    
    derivative_func : function
        function that returns derivatives given a state.  
    
    dt : float
        the time period to use. e.g 0.01 - 0.1
    
    Returns
    -------
    new_state : list
        The updated state with the added effects of the derivities applied over dt.
    """

    def add_scaled(state, derivative, scale):
        return [s + scale * d for s, d in zip(state, derivative)]

    # k1
    k1 = derivative_func(state)

    # k2
    state_k2 = add_scaled(state, k1, dt / 2)
    k2 = derivative_func(state_k2)

    # k3
    state_k3 = add_scaled(state, k2, dt / 2)
    k3 = derivative_func(state_k3)

    # k4
    state_k4 = add_scaled(state, k3, dt)
    k4 = derivative_func(state_k4)

    # Weighted average
    new_state = [
        s + (dt / 6) * (d1 + 2 * d2 + 2 * d3 + d4)
        for s, d1, d2, d3, d4 in zip(state, k1, k2, k3, k4)
    ]

    return new_state

