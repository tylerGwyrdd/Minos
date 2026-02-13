from minos.sim import runners as run_sim
import TSGA_utils as TSGA_utils
import numpy as np

params = {'initial_pos': [0, 0, 500]}

initial_conditions = np.array([
    np.array([0, 0, 0]),
    np.array([10, 0, 3]),
    np.array([0, 0, 0]),
    np.array([0, 0, 0])
])
t_end = 30
time_vector = np.arange(0, t_end + 0.1, 0.1)

new_coeffs = {
        "CDo": 0.32762986012330625,
        "CDa": 0.0026101301156606508,
        "CLo": 0.013817576109538515,
        "CLa": 1.7855420535732465,
        "Cmo": 0.5007028466665423,
        "Cma": -1.4465184567757894,
        "Cmq": -1.107077640407827,
        "CL_sym": 0.3654945354871382,
        "CD_sym": 0.49985844024563386 
 }
wind_vect = np.array([0, 0, 0])
wind_list = [wind_vect.copy() for _ in range(time_vector.size)]
left, right = TSGA_utils.generate_complete_flight(time_vector)

inputs = [left, right, wind_list]

a_data_set,_ = run_sim.bare_simulate_model(time_vector,initial_conditions,inputs,params,True,new_coeffs,False)
print(a_data_set)
TSGA_utils.plot_3D_position(a_data_set)

