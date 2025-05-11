import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import multiprocessing
import run_sim
import json
import os
import numbers
from deap import tools

import TSGA_utils
from TSGA_utils import ideal_coeffs, aero_coeffs, bounds_dict, coeff_names

params = {'initial_pos': [0, 0, 500]}

initial_conditions = np.array([
    np.array([0, 0, 0]),
    np.array([10, 0, 3]),
    np.array([0, 0, 0]),
    np.array([0, 0, 0])
])
t_end = 10
time_vector = np.arange(0, t_end + 0.1, 0.1)


# Step 1: Independent optimization of each coefficient
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

NGENS = 100
NPOP = 20

def unique_individuals(pop):
    """ work out how close genes are between individuals and return the number of different individuals"""
    return len(set(tuple(round(g, 5) for g in ind) for ind in pop))

def compute_genome_variance(population):
    """Computes variance of each gene across the population and returns mean variance."""
    genomes = np.array([ind[:] for ind in population])
    if len(genomes) == 0:
        return 0.0
    return np.mean(np.var(genomes, axis=0))

# Custom individual generator based on bounds
def generate_individual_from_set(names):
    """creates from set. Generates individual with just the named coeffs as genes"""
    std_dev = 0.01
    base_point = [aero_coeffs[name] for name in names]
    return creator.Individual([
        min(max(np.random.normal(loc=base, scale=max(abs(base * std_dev), 1e-3)),
                bounds_dict[name][0]), bounds_dict[name][1])
        for name, base in zip(names, base_point)
    ])

def make_toolbox(names, lows, highs, real_data, inputs):
    toolbox = base.Toolbox()
    # random setting from of the specific coefficients
    toolbox.register("individual_uniform", 
                     lambda: creator.Individual([random.uniform(l, h) for l, h in zip(lows, highs)]))
    toolbox.register("individual_gauss", lambda: generate_individual_from_set(names))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual_gauss)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=10, low=lows, up=highs)
    toolbox.register("mutate", tools.mutPolynomialBounded, eta=10, low=lows, up=highs, indpb=1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    def evaluate(individual):
        coeffs = ideal_coeffs.copy()
        for i, name in enumerate(names):
           coeffs[name] = individual[i]
        ordered_coeffs = [coeffs[k] for k in coeff_names]

        simulated, broke  = run_sim.bare_simulate_model(time_vector, initial_conditions, inputs, params, True, ordered_coeffs)
        error = np.linalg.norm(np.square(simulated) - np.square(np.array(real_data))) / (broke * 10)
        return (error,) if np.isfinite(error) else (1e6,)

    toolbox.register("evaluate", evaluate)
    return toolbox

def optimize_coefficients(names, real_data, inputs):
    import time
    lows = [bounds_dict[name][0] for name in names]
    highs = [bounds_dict[name][1] for name in names]

    toolbox = make_toolbox(names, lows, highs, real_data, inputs)
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
    # for elitism
    elite_size = 3
    mutp_change = 0.4
    mutp_base = 0.4

    # for stats
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    for gen in range(NGENS):
        mutation_chance = mutp_base + mutp_change * ( 1 - gen / NGENS)
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb = 1)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
                # sort out the population for elitism
        if gen > 0:
            pop = toolbox.select(offspring, k=len(pop) - elite_size) + elites
        else:
            pop = toolbox.select(offspring, k=len(pop))
        
        # get the elites of this population
        elites = tools.selBest(pop, elite_size)

        # --------- Stats -----------------
        hof.update(pop)
        record = stats.compile(pop)
        if gen % 10 == 0:
            best = elites[0].fitness.values[0]
            genome_var = compute_genome_variance(pop)
            avg = record["avg"]
            std = record["std"]
            uniq = unique_individuals(pop)
            print(f"Run: {names} | Gen: {gen} | Best: {best:.6f} | Avg: {avg:.6f} | Std: {std:.6f} | GenomeVar: {genome_var:.6f} | Unique individuals: {uniq} / {len(pop)}")
        if gen % 100 == 0 and gen != 0:
            for i, name in enumerate(names):
                print(f"Best for {name}: {hof[0][i]:.6f} ideal: {ideal_coeffs[name]:.6f} ")

        logbook.record(gen=gen, nevals=len(pop), **record)
        actual_errors.append(TSGA_utils.evaluate_partial_coeffs_error(elites[0], names))

    # ----------------------------------------------------------------------

    elapsed_time = time.time() - start_time
    best_vals = hof[0]
    print("completed estimations for the set: ", names)
    for i, name in enumerate(names):

        print(f"Best for {name}: {best_vals[i]:.6f} (ideal: {ideal_coeffs[name]:.6f})")
    
    coeffs = aero_coeffs.copy()
    for i, name in enumerate(names):
        coeffs[name] = best_vals[i]
    ordered_coeffs = [coeffs[k] for k in coeff_names]
    # Convert to ordered list of values matching the order of `names`
    sim_pos, died = run_sim.bare_simulate_model(time_vector, initial_conditions, inputs, params, True, ordered_coeffs, True)

    print(died)
    TSGA_utils.plot_3D_position_scatter(sim_pos)
    #TSGA_utils.plot_3D_geometry_comparision(time_vector,real_data, initial_conditions, inputs, params, ordered_coeffs)
    TSGA_utils.save_population_to_json(pop, names, f"population_gen_{gen}.json")
    return {
        "best_values": dict(zip(names, best_vals)),
        "actual_errors": actual_errors,
        "fitness_over_time": logbook,
        "time_seconds": elapsed_time
    }

def wrapped_optimize(args):
    names, real_data, inputs = args
    result = optimize_coefficients(names, real_data, inputs)
    return (" + ".join(names), result)

if __name__ == '__main__':
    wind_vect = np.array([0, 0, 0])
    wind_list = [wind_vect.copy() for _ in range(time_vector.size)]
    grouped_names = [
        ["CDo", "CDa", "CLo", "CLa", "Cmo", "Cma", "Cmq", "CL_sym", "CD_sym"]
    ]
    """        ["CDo"],
        ["CDa"],
        ["CLo"],
        ["CLa"],
        ["Cma"],
        ["Cmq"],
        ["CL_sym"],
        ["CD_sym"]
    ]
    """
    real_datasets = []
    inputs_list = []

    left, right = TSGA_utils.generate_straight_flight(time_vector)
    inputs_list.append([left, right, wind_list])

    """    inputs_list.append([left, right, wind_list])
    inputs_list.append([left, right, wind_list])
    inputs_list.append([left, right, wind_list])
    inputs_list.append([left, right, wind_list])
    inputs_list.append([left, right, wind_list])
    inputs_list.append([left, right, wind_list])
    inputs_list.append([left, right, wind_list])
    inputs_list.append([left, right, wind_list])"""
    coeffs = ideal_coeffs.copy()
    for i, name in enumerate(grouped_names[0]):
        print("setting", name , " to ", ideal_coeffs[name])
        coeffs[name] = ideal_coeffs[name]
    ordered_coeffs = [coeffs[k] for k in coeff_names]
    a_data_set = TSGA_utils.generate_real_data(time_vector, left, right, wind_list, params, initial_conditions)
    TSGA_utils.plot_3D_position_scatter(a_data_set)
    real_datasets.append(a_data_set)

    """    real_datasets.append(a_data_set)
    real_datasets.append(a_data_set)
    real_datasets.append(a_data_set)
    real_datasets.append(a_data_set)
    real_datasets.append(a_data_set)
    real_datasets.append(a_data_set)
    real_datasets.append(a_data_set)
    real_datasets.append(a_data_set)"""


    """    for _ in range(4):
        left, right = TSGA_utils.genertate_spirial_flight(time_vector)
        left = match_length(left, len(time_vector))
        right = match_length(right, len(time_vector))
        inputs_list.append([left, right, wind_list])
        real_datasets.append(TSGA_utils.generate_real_data(time_vector, left, right, wind_vect, params, initial_conditions))"""

    tasks = list(zip(grouped_names, real_datasets, inputs_list))
    print("RUNNIN")
    with multiprocessing.Pool() as pool:
        results = pool.map(wrapped_optimize, tasks)
        
    best_coeffs = {group: res["best_values"] for group, res in results}
    metrics = {
        group: {
            "fitness_over_time": [dict(rec) for rec in res["fitness_over_time"]],
            "time_seconds": res["time_seconds"],
            "actual_error": res["actual_errors"]
        }
        for group, res in results
    }
    filename = "hello "
    with open("modified_tsga_best_coeffs.json", "w") as f:
        json.dump(best_coeffs, f, indent=4)

    with open("modified_tsga_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    

    print("Step 1 complete. Best coefficients written to tsga_step1_best.json")
    TSGA_utils.plot_metrics_individually("modified_tsga_metrics.json")
    TSGA_utils.plot_individual_errors("modified_tsga_metrics.json")



