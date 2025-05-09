import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import multiprocessing
import run_sim
import json
import os
import numbers
# Define coefficient names and bounds
coeff_names = [
    "CDo", "CDa", "CD_sym", "CYB", "CLo", "CLa", "CL_sym", "Cmo", "Cma", "Cmq",
    "ClB", "Clp", "Clr", "Cl_asym", "CnB", "Cn_p", "Cn_r", "Cn_asym"
]

bounds_dict = {
    "CDo": (0, 0.5),
    "CDa": (0, 0.25),
    "CD_sym": (0, 0.5),
    "CYB": (-0.5, 0),
    "CLo": (0, 0.2),
    "CLa": (0, 1.8),
    "CL_sym": (0, 0.5),
    "Cmo": (0, 0.7),
    "Cma": (-1.5, 0),
    "Cmq": (-3.0, 0),
    "ClB": (-0.1, 0),
    "Clp": (-1.7, 0),
    "Clr": (-0.2, 0),
    "Cl_asym": (-0.01, 0),
    "CnB": (-0.01, 0),
    "Cn_p": (-0.2, 0),
    "Cn_r": (-0.6, 0),
    "Cn_asym": (0, 0.03)
}

aero_coeffs = {
    "CDo": 0.25,
    "CDa": 0.12,
    "CD_sym": 0.2,
    "CYB": -0.23,
    "CLo": 0.091,
    "CLa": 0.90,
    "CL_sym": 0.2,
    "Cmo": 0.35,
    "Cma": -0.72,
    "Cmq": -1.49,
    "ClB": -0.036,
    "Clp": -0.84,
    "Clr": -0.082,
    "Cl_asym": -0.0035,
    "CnB": -0.0015,
    "Cn_p": -0.082,
    "Cn_r": -0.27,
    "Cn_asym": 0.0115,
}

# the desired areodynamic coeffs
ideal_coeffs = {
    "CDo": 0.3,
    "CDa": 0.05,
    "CD_sym": 0.25,
    "CYB": -0.15,
    "CLo": 0.12,
    "CLa": 0.9,
    "CL_sym": 0.3,
    "Cmo": 0.3,
    "Cma": -0.5,
    "Cmq": -1.7,
    "ClB": -0.07,
    "Clp": -0.8,
    "Clr": -0.082,
    "Cl_asym": -0.004,
    "CnB": -0.002,
    "Cn_p": -0.082,
    "Cn_r": -0.27,
    "Cn_asym": 0.02
}

def generate_complete_flight(time_vector):
    left_input = []
    right_input = []
    for t in time_vector:
        if t < 10:
            left_input.append(0)
            right_input.append(0)
        elif t < 20:
            left_input.append(0.2)
            right_input.append(0)
        elif t < 30:
            left_input.append(0)
            right_input.append(0)
        elif t < 40:
            left_input.append(0)
            right_input.append(0.2)           
        else:
            left_input.append(0.2)
            right_input.append(0.2)
    return left_input,right_input

def generate_straight_flight(time_vector):
    left_input = []
    right_input = []
    for t in time_vector:
        if t < 1:
            left_input.append(0)
            right_input.append(0)
        elif t < 2:
            left_input.append(0.1)
            right_input.append(0.1)
        elif t < 3:
            left_input.append(0.5)
            right_input.append(0.5)
        elif t < 6:
            left_input.append(0)
            right_input.append(0)
        elif t < 9:
            left_input.append(1)
            right_input.append(1)
        else:
            left_input.append(0)
            right_input.append(0)
    return left_input,right_input

def genertate_spirial_flight(time_vector):
    left_input = []
    right_input = []
    for t in time_vector:
        if t < 5:
            left_input.append(0)
            right_input.append(0)
        elif t < 10:
            left_input.append(0.1)
            right_input.append(0)
        elif t < 15:
            left_input.append(0)
            right_input.append(0)
        elif t < 20:
            left_input.append(0.2)
            right_input.append(0)
        elif t < 30:
            left_input.append(0.2)
            right_input.append(0)
        else:
            left_input.append(0)
            right_input.append(0)
    return left_input,right_input

def generate_real_data(time_vector, l_input, r_input, wind_list , params, initial_conditions):
    coeffcients = ideal_coeffs
    inputs = [l_input, r_input, wind_list]
    result,_ = run_sim.bare_simulate_model(time_vector, initial_conditions, inputs, params, True, coeffcients)
    return result

def evaluate_coeffs_error(coeff_dict):
    percent_errors = [
        100 - abs((val - ideal_coeffs[name]) / ideal_coeffs[name]) * 100
        if ideal_coeffs[name] != 0 else 0
        for name, val in coeff_dict.items()
    ]
    return sum(percent_errors) / len(percent_errors) if percent_errors else 0

def evaluate_partial_coeffs_error(values, names):
    errors = {}
    for val, name in zip(values, names):
        ideal = ideal_coeffs[name]
        error = abs((val - ideal) / ideal) * 100 if ideal != 0 else 0
        errors[name] = error
    return errors

def evaluate_ind_coeffs_error(value, name):
    ideal = ideal_coeffs[name]
    return abs((value - ideal) / ideal) * 100 if ideal != 0 else 0

def plot_metrics_individually(file_path):
    # Load the JSON data
    with open(file_path, "r") as file:
        data = json.load(file)

    # Generate a separate plot for each coefficient set
    for key, value in data.items():
        generations = [entry["gen"] for entry in value["fitness_over_time"]]
        avg_fitness = [entry["avg"] for entry in value["fitness_over_time"]]
        
        # Create a new figure for each coefficient set
        plt.figure(figsize=(10, 5))
        plt.plot(generations, avg_fitness, label='Average Fitness')
        plt.title(f"Average Fitness Over Generations\n({key})")
        plt.xlabel("Generation")
        plt.ylabel("Average Fitness")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
    
    plt.show()

def plot_individual_errors(file_path):
    """Plot all coefficient errors on a single graph per coefficient set."""
    # Load the JSON data
    with open(file_path, "r") as file:
        data = json.load(file)

    for key, value in data.items():
            if not isinstance(value, dict):
                    print("skipped", value)
                    continue
            errors = value.get("actual_error", [])
            if not isinstance(errors, list) or not errors:
                print("skipped h", value)
                continue

            generations = list(range(len(errors)))
            coeffs = errors[0].keys()

            plt.figure(figsize=(10, 5))
            for coeff in coeffs:
                values = [entry[coeff] for entry in errors]
                plt.plot(generations, values, label=coeff)

            plt.title(f"Coefficient Errors Over Generations\n({key})")
            plt.xlabel("Generation")
            plt.ylabel("Error")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

def plot_3D_geometry_comparision(time_vector, real_data, initial_conditions, inputs, params, coefficients):
    sim_pos = run_sim.bare_simulate_model(time_vector, initial_conditions, inputs, params, True, coefficients)
    sim_pos = np.array([np.asarray(s) for s in sim_pos])
    real_data = np.array([np.asarray(s) for s in real_data])
    fig_best = plt.figure()
    ax_best = fig_best.add_subplot(111, projection='3d')
    ax_best.plot(sim_pos[:, 0], sim_pos[:, 1], sim_pos[:, 2], label=f'GA Best')
    ax_best.plot(real_data[:, 0], real_data[:, 1], real_data[:, 2], label=f'Ideal')
    ax_best.set_xlabel('X')
    ax_best.set_ylabel('Y')
    ax_best.set_zlabel('Z')
    ax_best.set_title(f'3D Trajectory - best one')
    ax_best.legend()
    plt.show()

def plot_3D_position(real_data):
    real_data = np.array([np.asarray(s) for s in real_data])
    fig_best = plt.figure()
    ax_best = fig_best.add_subplot(111, projection='3d')
    ax_best.plot(real_data[:, 0], real_data[:, 1], real_data[:, 2], label=f'Ideal')
    ax_best.set_xlabel('X')
    ax_best.set_ylabel('Y')
    ax_best.set_zlabel('Z')
    ax_best.set_title(f'Ideal trajectory')
    ax_best.legend()
    plt.show()

def plot_fitness(file_path):
    # Load the JSON data
    with open(file_path, "r") as file:
        data = json.load(file)

#plot_metrics_individually("modified_tsga_metrics.json")

#plot_positions_individually("modified_tsga_metrics.json")
"""
if __name__ == '__main__':    
    params = {'initial_pos': [0, 0, 500]}
    initial_conditions = np.array([
        np.array([0, 0, 0]),
        np.array([10, 0, 3]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0])
    ])
    t_end = 20
    time_vector = np.arange(0, t_end + 0.1, 0.1)

    wind_vect = np.array([0, 0, 0])
    wind_list = [wind_vect.copy() for _ in range(time_vector.size)]

    left, right = generate_complete_flight(time_vector)

    real_data = generate_real_data(time_vector, left, right, wind_list, params, initial_conditions)
    plot_3D_position(real_data)

"""
