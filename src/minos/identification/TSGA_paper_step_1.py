import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import multiprocessing
from minos.sim import runners as run_sim
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

def make_toolbox(name, low, high):
    toolbox = base.Toolbox()

    toolbox.register("attr_float", random.uniform, low, high)
    toolbox.register("individual", tools.initRepeat, creator.Individual[0], toolbox.attr_float, 1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=10.0, low=low, up=high)
    toolbox.register("mutate", tools.mutPolynomialBounded, eta=0.5, low=low, up=high, indpb=1.0)
    toolbox.register("select", tools.selTournament, tournsize=3)

    def evaluate_param(name, individual):
        coeffs = aero_coeffs.copy()
        coeffs[name] = individual[0]
        ordered_coeffs = [coeffs[k] for k in coeff_names]
        simulated = run_sim.bare_simulate_model(time_vector, initial_conditions, inputs, params, True, ordered_coeffs)
        error = np.linalg.norm(simulated - np.array(real_data)) / len(simulated)
        return (error,) if np.isfinite(error) else (1e6,)

    def evaluate_wrapper(name):
        def evaluate(ind):
            return evaluate_param(name, ind)
        return evaluate

    toolbox.register("evaluate", evaluate_wrapper(name))
    return toolbox

def optimize_coefficient(name):
    import time
    low, high = bounds_dict[name]
    toolbox = make_toolbox(name, low, high)
    pop = toolbox.population(n=NPOP)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals", "avg", "min", "max"]

    start_time = time.time()
    actual_er = []

    for gen in range(NGENS):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.3)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        pop = toolbox.select(offspring, k = len(pop))
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(pop), **record)
        actual_er.append(TSGA_utils.evaluate_ind_coeffs_error(tools.selBest(pop, 1)[0][0],name))

    elapsed_time = time.time() - start_time
    best_val = hof[0][0]
    print(f"Best for {name}: {best_val:.6f} (ideal: {ideal_coeffs[name]:.6f})")
    return name, {
        "best_value": best_val,
        "actual_error": actual_er,
        "fitness_over_time": logbook,
        "time_seconds": elapsed_time
    }

def plot_metrics_one_graph(metrics_file="tsga_step1_metrics.json"):
    # Load metrics from JSON
    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    fig, ax1 = plt.subplots(figsize=(12, 7))

    ax2 = ax1.twinx()  # Secondary y-axis for percent error
    # Plot for each coefficient (e.g., "ClB", "CmA", etc.)
    for name, data in metrics.items():
        fitness = data["fitness_over_time"]
        generations = [entry["gen"] for entry in fitness]
        avg_fitness = [entry["avg"] for entry in fitness]
        actual_error = data["actual_error"]

        # Primary axis: average fitness
        ax1.plot(generations, avg_fitness, label=f"{name} Avg Fitness")

        # Secondary axis: percent error (plotted as a constant line)
        ax2.plot(generations, actual_error, '--', label=f"{name} % Error")

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Average Fitness")
    ax2.set_ylabel("Percent Error")

    ax1.set_title("Fitness and Percent Error over Generations")
    ax1.grid(True)

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    plt.tight_layout()
    plt.show()

def plot_metrics_individually(metrics_file="tsga_step1_metrics.json"):
    import matplotlib.pyplot as plt
    import json
    import math

    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    num_coeffs = len(metrics)
    ncols = 2
    nrows = math.ceil(num_coeffs / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4 * nrows), sharex=True)

    # Flatten axes array for easy indexing
    axes = axes.flatten()

    for i, (name, data) in enumerate(metrics.items()):
        ax = axes[i]
        fitness = data["fitness_over_time"]
        generations = [entry["gen"] for entry in fitness]
        avg_fitness = [entry["avg"] for entry in fitness]
        error_vals = [entry for entry in data["actual_error"]]

        ax2 = ax.twinx()

        # Left y-axis: fitness
        ax.plot(generations, avg_fitness, label="Avg Fitness", color="tab:blue")
        ax.set_ylabel("Fitness", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")

        # Right y-axis: percent error
        ax2.plot(generations, error_vals, label="Percent Error", color="tab:red", linestyle='--')
        ax2.set_ylabel("Percent Error", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        ax.set_title(f"Coefficient: {name}")
        ax.set_xlabel("Generation")
        ax.grid(True)

    # Hide any unused subplots
    for j in range(len(metrics), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    t_end = 50
    time_vector = np.linspace(0, t_end, t_end * 10)
    l_input, r_input = TSGA_utils.generate_complete_flight(time_vector)
    wind_vect = np.array([0, 0, 0])
    wind_list = [wind_vect.copy() for _ in range(time_vector.size)]
    inputs = [l_input, r_input, wind_list]
    real_data = run_sim.sim_with_noise(time_vector, initial_conditions, inputs, params, True, [ideal_coeffs[k] for k in coeff_names])
    np.savez("shared_data.npz", time_vector=time_vector, inputs=np.array(inputs, dtype=object), real_data=np.array(real_data))
    print("__________________STARTING_________________")

    multiprocessing.freeze_support()
    with multiprocessing.Pool() as pool:
        results = pool.map(optimize_coefficient, coeff_names)

    best_coeffs = {name: res["best_value"] for name, res in results}
    metrics = {name: {"fitness_over_time": [dict(rec) for rec in res["fitness_over_time"]], "time_seconds": res["time_seconds"],"actual_error": res["actual_error"]} for name, res in results}
    with open("tsga_step1_best.json", "w") as f:
        json.dump(best_coeffs, f, indent=4)

    with open("tsga_step1_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Step 1 complete. Best coefficients written to tsga_step1_best.json")
    plot_metrics_individually()
