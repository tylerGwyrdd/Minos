import numpy as np
import matplotlib.pyplot as plt
import minos.physics.six_DoF_simulator as simulator
import logging
from utils import visualize_parafoil_pose
from utils import rk4
import guidance_v2
import copy

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(message)s')

def simple_time_control(time):
    # Simple control logic for demonstration purposes
    time_interval = 20
    if time < 1 * time_interval:
        return [0, 0]  # Flap deflections
    elif time < 2* time_interval:
        return [0.0, 0.3]
    elif time < 3 * time_interval:
        return [0.0, 0.4]
    elif time < 4 * time_interval:
        return [0.0, 0.5]
    elif time < 5 * time_interval:
        return [0.0, 0.6]
    elif time < 6 * time_interval:
        return [0.0, 0.7]
    elif time < 7 * time_interval:
        return [0.0, 0.8]
    elif time < 8 * time_interval:
        return [0.0, 0.9]
    elif time < 9 * time_interval:
        return [0.0, 1]
    else:
        return None
    
def run_simulation(sim, steps, dt):
    data = []
    t = 0
    for i in range(steps): 
        state = sim.get_state()
        new_inputs = simple_time_control(t)
        if new_inputs is None:
            break
        sim.set_actual_flaps(new_inputs)
        # uses rk4 to work out how much derivititve to add - its is pretty good
        new_state = rk4(state, sim.get_solver_derivatives, dt)
        # update sim
        sim.set_state(new_state)

        # save the current data
        data.append([
            t, state, sim.angle_of_attack, sim.sideslip_angle,
            sim.angular_acc, sim.acc, sim.CL, sim.CD, sim.Cl, sim.Cn, sim.Cm,
            sim.F_aero, sim.F_g, sim.F_fictious, sim.M_aero, sim.M_f_aero, 
            sim.M_fictious, sim.va, sim.w, [sim.flap_l, sim.flap_r],
            [0, 0], sim.get_inertial_position(), 
            sim.get_euler_rates()
        ])
        # update the time
        t += dt
    return data

def run_simulation_with_guidance(sim, steps, dt, guidance_params):
    data = []
    t = 0
    for i in range(steps):
        # get state
        state = sim.get_state()
        inertial_state = sim.get_inertial_state()
        # check if we have hit the ground
        if(inertial_state[0][2] < guidance_params["IPI"][2]):
            logging.info("Parafoil has hit the ground, stopping simulation.")
            logging.info(f"Final Position: {[f'{coord:.3g}' for coord in state[0]]}")
            IPI_error = guidance_params['IPI'][:2] - state[0][:2]
            logging.info(f"Final IPI Error: {[f'{e:.3g}' for e in IPI_error]}")
            estimated_wind_vector = guidance_params['wind_unit_vector'] * guidance_params['wind_magnitude']
            logging.info(f"Estimated Wind: {[f'{w:.3g}' for w in estimated_wind_vector]}")
            logging.info(f"Actual Wind: {[f'{w:.3g}' for w in sim.w]}")
            break

        # ============ path planning coding ============================
        current_heading = state[2][2]
        current_euler_rate = inertial_state[3][2]
        desired_heading,_ = guidance_v2.guidance_update(guidance_params, inertial_state)
        
        new_inputs = guidance_v2.PID_control(current_heading, desired_heading,current_euler_rate)
        # logging.info(f"Current heading: {current_heading:.2f}, Desired heading: {desired_heading:.2f}, new inputs: {new_inputs}")
        
        # ================ simple physics sim =======================
        #new_inputs = simple_control(t)
        #desired_heading, current_heading = 0, 0

        # update the control inputs
        sim.set_desired_flaps(new_inputs)
        sim.update_flaps(dt)
        # uses rk4 to work out how much derivititve to add - its is pretty good
        new_state = rk4(state, sim.get_solver_derivatives, dt)
        # update sim
        sim.set_state(new_state)

        # save the current data
        data.append([
            t, state, sim.angle_of_attack, sim.sideslip_angle,
            sim.angular_acc, sim.acc, sim.CL, sim.CD, sim.Cl, sim.Cn, sim.Cm,
            sim.F_aero, sim.F_g, sim.F_fictious, sim.M_aero, sim.M_f_aero, 
            sim.M_fictious, sim.va, sim.w, [sim.flap_l, sim.flap_r],
            [current_heading, desired_heading], sim.get_inertial_position(), 
            sim.get_euler_rates()
        ])
        # update the time
        t += dt
    return data


def sim_with_guidance():
    # sim params
    # teperal resolution of the sim
    dt = 0.1
    # number of steps to run the sim for
    steps = 5000 

    # -------------- sim init -----------------------
    # Note sim uses NED Coordinate system, so Z is up
    init_body_pos=np.array([0, 0, 0])
    init_body_vel=np.array([10, 0, 3]) # this is stable
    init_eulers=np.array([0, 0, 0])
    init_omega=np.array([0, 0, 0])
    init_state = [init_body_pos,init_body_vel,init_eulers,init_omega]

    deployment_pos_inertial = np.array([0, 50, 500])

    wind = np.array([1, 1, 0])

    initial_inputs = [[0.0, 0.0], wind]

    # params: you can chase the specifics using this. its a dict
    sim_params = {
        'initial_pos': deployment_pos_inertial ,
    }
    # guidance
    guidance_params = {
    'deployment_pos': deployment_pos_inertial,
    'final_approach_height': 100,
    'spirialing_radius': 20,
    'update_rate': dt,
    'wind_unit_vector': np.array([1, 0]),
    'wind_magnitude': 0.0,
    'wind_v_list': [],
    'horizontal_velocity': 5.9,
    'sink_velocity': 4.9,
    'IPI': np.array([0, 0, 0]),
    'flare_height': 20,
    'initialised': False,
    'mode': 'initialising',  # initalising, homing, final approach, energy management
    'start_heading': 0,  # radians
    'desired_heading': np.deg2rad(0),  # radians
    'FTP_centre': np.array([0,0]),  # Final target point
    }
    # save for later
    ideal_guidance_params = copy.deepcopy(guidance_params)
    ideal_guidance_path = []
    # ----------------------- running --------------------------
    # lets make the objects
    sim = simulator.ParafoilSimulation_6Dof(sim_params, init_state, initial_inputs)

    # run the sim
    data = run_simulation_with_guidance(sim,guidance_params, steps, dt)
    guidance_init_state = init_state
    guidance_init_state[0] = deployment_pos_inertial
    ideal_guidance_path = guidance_v2.ideal_guidance(ideal_guidance_params, wind, init_state, 10, steps)
    
    # ------------------------ post ---------------------------
    plots_to_show = {
        'Position': True,
        'Velocity': False,
        'Acceleration': False,
        'Euler Angles': True,
        'Angular Velocity': False,
        'Angular Acceleration': False,
        'Angle of Attack': False,
        'Sideslip Angle': True,
        'Force Coefficients': False,
        'Moment Coefficients': False,
        'Forces': False,
        'Moments': False,
        'Airspeed Vector': True,
        'Wind Vector': True,
        'Deflection': True,
        'Euler Rates': True,
        'headings': True
    }

    plot_selected_parameters(data, plots_to_show)
    eulers = np.degrees(np.array([entry[1][2] for entry in data]))
    positions = np.array([entry[1][0] for entry in data])
    inertial_positions = np.array([entry[21] for entry in data])
    # Visualize parafoil pose
    visualize_parafoil_pose(
        euler_series=eulers,
        position_series=positions,
        interval=100,
        slowmo_factor=1.0,
        save_path=False
    )
    # ------------------- ideal guidance ----------------------
    # lets generate ideal guidance path

    guidance_v2.plot_3D_position(inertial_positions,guidance_params, ideal_guidance_path)

def just_physics_sim():
       # sim params
    # teperal resolution of the sim
    dt = 0.1
    # number of steps to run the sim for
    steps = 2000 

    # -------------- sim init -----------------------
    # Note sim uses NED Coordinate system, so Z is up
    init_body_pos=np.array([0, 0, 0])
    init_body_vel=np.array([10, 0, 3]) # this is stable
    init_eulers=np.array([0, 0, 0])
    init_omega=np.array([0, 0, 0])
    init_state = [init_body_pos,init_body_vel,init_eulers,init_omega]

    deployment_pos_inertial = np.array([0, 0, 500])

    wind = np.array([0, 0, 0])

    initial_inputs = [[0.0, 0.0], wind]

    # params: you can chase the specifics using this. its a dict
    sim_params = {
        'initial_pos': deployment_pos_inertial ,
    }
       # ----------------------- running --------------------------
    # lets make the objects
    sim = simulator.ParafoilSimulation_6Dof(sim_params, init_state, initial_inputs)

    # run the sim
    data = run_simulation(sim, steps, dt)
    
    # ------------------------ post ---------------------------
    plots_to_show = {
        'Position': True,
        'Velocity': False,
        'Acceleration': False,
        'Euler Angles': True,
        'Angular Velocity': False,
        'Angular Acceleration': False,
        'Angle of Attack': False,
        'Sideslip Angle': True,
        'Force Coefficients': False,
        'Moment Coefficients': False,
        'Forces': False,
        'Moments': False,
        'Airspeed Vector': True,
        'Wind Vector': True,
        'Deflection': True,
        'Euler Rates': True,
        'headings': True
    }

    plot_selected_parameters(data, plots_to_show)
    eulers = np.degrees(np.array([entry[1][2] for entry in data]))
    positions = np.array([entry[1][0] for entry in data])
    inertial_positions = np.array([entry[21] for entry in data])
    # Visualize parafoil pose
    visualize_parafoil_pose(
        euler_series=eulers,
        position_series=positions,
        interval=100,
        slowmo_factor=1.0,
        save_path=False
    )
