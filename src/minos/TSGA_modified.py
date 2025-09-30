import random
import numpy as np
from deap import base, creator, tools, algorithms
import multiprocessing
import run_sim
import json
import os

import TSGA_utils as TSGA_utils
from TSGA_utils import ideal_coeffs, aero_coeffs, bounds_dict, coeff_names

params = {'initial_pos': [0, 0, 0]}

initial_conditions = np.array([
    np.array([0, 0, 0]),
    np.array([10, 0, 3]),
    np.array([0, 0, 0]),
    np.array([0, 0, 0])
])

if os.path.exists("shared_data.npz"):
    data = np.load("shared_data.npz", allow_pickle=True)
    time_vector = data['time_vector']
    inputs = data['inputs'].tolist()
    real_data = data['real_data']
else:
    raise FileNotFoundError("shared_data.npz not found.")


# Step 1: Independent optimization of each coefficient
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

NGENS = 100
NPOP = 20

def make_toolbox(names, lows, highs):
    n_params = len(names)

    toolbox = base.Toolbox()

    toolbox.register("individual", 
                     lambda: creator.Individual([random.uniform(l, h) for l, h in zip(lows, highs)]))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=10.0, low=lows, up=highs)
    toolbox.register("mutate", tools.mutPolynomialBounded, eta=0.5, low=lows, up=highs, indpb=1.0)
    toolbox.register("select", tools.selTournament, tournsize=3)

    def evaluate(individual):
        coeffs = aero_coeffs.copy()
        for i, name in enumerate(names):
            coeffs[name] = individual[i]
        ordered_coeffs = [coeffs[k] for k in coeff_names]
        simulated = run_sim.bare_simulate_model(time_vector, initial_conditions, inputs, params, True, ordered_coeffs)
        error = np.linalg.norm(simulated - np.array(real_data)) / len(simulated)
        return (error,) if np.isfinite(error) else (1e6,)

    toolbox.register("evaluate", evaluate)
    return toolbox

def optimize_coefficients(names):
    import time

    # Get lower and upper bounds per coefficient
    lows = [bounds_dict[name][0] for name in names]
    highs = [bounds_dict[name][1] for name in names]

    toolbox = make_toolbox(names, lows, highs)
    pop = toolbox.population(n=NPOP)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals", "avg", "min", "max"]

    start_time = time.time()
    actual_errors = []

    for gen in range(NGENS):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.3)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        pop = toolbox.select(offspring, k=len(pop))
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(pop), **record)

        # Track individual coefficient errors
        best_ind = tools.selBest(pop, 1)[0]
        actual_errors.append(TSGA_utils.evaluate_partial_coeffs_error(best_ind, names))

    elapsed_time = time.time() - start_time
    best_vals = hof[0]
    print("completed estimations for the set: ", names)
    for i, name in enumerate(names):
        print(f"Best for {name}: {best_vals[i]:.6f} (ideal: {ideal_coeffs[name]:.6f})")

    return {
        "best_values": dict(zip(names, best_vals)),
        "actual_errors": actual_errors,
        "fitness_over_time": logbook,
        "time_seconds": elapsed_time
    }

def wrapped_optimize(names):
    result = optimize_coefficients(names)
    return (tuple(names), result)

if __name__ == '__main__':
    t_end = 50
    time_vector = np.linspace(0, t_end, t_end * 10)
    l_input, r_input = TSGA_utils.generate_inputs(time_vector)
    wind_vect = np.array([0, 0, 0])
    wind_list = [wind_vect.copy() for _ in range(time_vector.size)]
    inputs = [l_input, r_input, wind_list]
    real_data = run_sim.sim_with_noise(time_vector, initial_conditions, inputs, params, True, [ideal_coeffs[k] for k in coeff_names])
    np.savez("shared_data.npz", time_vector=time_vector, inputs=np.array(inputs, dtype=object), real_data=np.array(real_data))
    print("__________________STARTING_________________")
    multiprocessing.freeze_support()

    # Define groups of coefficient names to optimize together
    grouped_names = [
        ["CDo", "CDa", "CYB", "CLo", "CLa"],
        ["CD_sym", "CL_sym", "Cl_asym", "Cn_asym"],
        ["Cmo", "Cma", "Cmq"],
        ["ClB", "Clp", "Clr"],
        ["CnB", "Cn_p", "Cn_r"]
    ]

    with multiprocessing.Pool() as pool:
        results = pool.map(wrapped_optimize, grouped_names)

    # Collect results into dictionaries
    best_coeffs = {group: res["best_values"] for group, res in results}
    metrics = {
        group: {
            "fitness_over_time": [dict(rec) for rec in res["fitness_over_time"]],
            "time_seconds": res["time_seconds"],
            "actual_error": res["actual_errors"]
        }
        for group, res in results
    }
    with open("tsga_step1_best.json", "w") as f:
        json.dump(best_coeffs, f, indent=4)

    with open("tsga_step1_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Step 1 complete. Best coefficients written to tsga_step1_best.json")
    TSGA_utils.plot_metrics_individually()
