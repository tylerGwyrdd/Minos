import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import multiprocessing
import run_sim
import json
import os
import numbers
import time
import TSGA_utils as TSGA_utils

# ------------- Initalising Vars ------------

snowflake_coefficients = [
        0.25,
        0.12,
        0.2,
        0.091,
        0.90,
        0.2,
        -0.23,
        -0.036, # coefficient due to sideslip angle
        -0.84,
        -0.082,
        -0.0035,
        0.35, # coefficient at zero lift
        -0.72, # coefficient due to angle of incidance
        -1.49,
        -0.0015, # coefficient due to sideslip angle
        -0.082, # coefficient due to roll rate
        -0.27,
        0.0115 
]

bounds = [
    (0, 0.5),     # 0.25 → rounded to 0.5
    (0, 0.25),    # 0.12 → 0.24 → rounded to 0.25
    (0, 0.4),     # 0.2
    (0, 0.2),     # 0.091 → 0.182 → 0.2
    (0, 1.8),     # 0.9
    (0, 0.4),     # 0.2
    (-0.5, 0),    # -0.23 → -0.46 → -0.5
    (-0.1, 0),    # -0.036 → -0.072 → -0.1
    (-1.7, 0),    # -0.84 → -1.68 → -1.7
    (-0.2, 0),    # -0.082 → -0.164 → -0.2
    (-0.01, 0),# -0.0035 → ±0.007 → rounded
    (0, 0.7),     # 0.35
    (-1.5, 0),    # -0.72 → -1.44 → -1.5
    (-3.0, 0),    # -1.49 → -2.98 → -3.0
    (-0.01, 0),# -0.0015 → ±0.003 → rounded
    (-0.2, 0),    # -0.082 → -0.164 → -0.2
    (-0.6, 0),    # -0.27 → -0.54 → -0.6
    (0, 0.03)     # 0.0115 → 0.023 → rounded to 0.02
]

ideal_coefficients = [
    0.3, # 0
    0.05, # 1
    0.3, # 2
    0.12, # 3
    0.9, # 4
    0.24, #5
    -0.15, # 6
    -0.07, # 7
    -0.8, # 8
    -0.082, # 9
    -0.004, # 10
    0.3, # 11
    -0.5, # 12
    -1.7, # 13
    -0.002, # 14
    -0.082, # 15
    -0.27, # 16
    0.02 # 17
]

params = {'initial_pos': [0, 0, 0]}
initial_conditions = np.array([np.array([0,0,0]), np.array([10,0,3]), np.array([0,0,0]), np.array([0,0,0])])

try:
    if os.path.exists("shared_data.npz"):
        data = np.load("shared_data.npz", allow_pickle=True)
        time_vector = data['time_vector']
        inputs = data['inputs'].tolist()
        real_data = data['real_data']
    else:
        raise FileNotFoundError("shared_data.npz not found.")
except Exception as e:
    time_vector = None
    inputs = None
    real_data = None
    raise RuntimeError(f"Failed to load shared data: {e}") from e



# --- Fitness function ---
def evaluate(coefficients):
    simulated = run_sim.bare_simulate_model(time_vector, initial_conditions, inputs, params, True, coefficients)
    error = np.linalg.norm(simulated - np.array(real_data)) / len(simulated)
    # print("evaluated. Error = ", error)
    if not np.isfinite(error):
        return (1e6,)
    return (error,)

# Custom individual generator based on bounds
def generate_individual():
    return creator.Individual([random.uniform(low, high) for low, high in bounds])

# Custom individual generator based on bounds
def generate_individual_from_set():
    base_point = snowflake_coefficients  # can be replaced with any starting set
    std_dev = 0.5  # amount of variation around each coefficient
    return creator.Individual([
        min(max(np.random.normal(loc=base, scale= abs(base * std_dev)), low), high)
        for (low, high), base in zip(bounds, base_point)
    ])
# Custom individual generator based on bounds
def generate_individual():
    base_point = ideal_coefficients  # can be replaced with any starting set
    std_dev = 0.05  # amount of variation around each coefficient
    return creator.Individual([
        min(max(np.random.normal(loc=base, scale=std_dev), low), high)
        for (low, high), base in zip(bounds, base_point)
    ])

# Custom mutation respecting bounds
def mutate_individual(individual, eta=0.5, indpb=0.2):
    individual = clamp_individual(individual)
    # Apply polynomial mutation with fallback for complex values
    for i, (low, up) in enumerate(bounds):
        if random.random() < indpb:
            try:
                mutated = tools.mutPolynomialBounded(
                    [individual[i]], eta=eta, low=low, up=up, indpb=1.0
                )[0][0]
                if isinstance(mutated, numbers.Real) and not isinstance(mutated, complex):
                    individual[i] = mutated
                else:
                    raise ValueError("Mutation resulted in complex or invalid number")
            except Exception:
                individual[i] = random.uniform(low, up)
    return individual,


# Clamp individual after mating to keep within bounds
def clamp_individual(individual):
    for i, (low, up) in enumerate(bounds):
        individual[i] = max(min(individual[i], up), low)
    return individual





# --- DEAP setup ---
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("individual", generate_individual_from_set)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", mutate_individual)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Register statistics
stats = tools.Statistics(lambda ind: ind.fitness.values[0])
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

logbook = tools.Logbook()
logbook.header = ["gen", "evals", "avg", "std", "min", "max"]

if __name__ == '__main__':
    # ------ generate the real data ----------------
    t_end = 50
    time_vector = np.linspace(0, t_end, t_end * 10)
    l_input, r_input = TSGA_utils.generate_complete_flight(time_vector)
    wind_vect = np.array([0, 0, 0])
    wind_list = [wind_vect.copy() for _ in range(time_vector.size)]
    inputs = [l_input, r_input, wind_list]
    real_data = run_sim.sim_with_noise(time_vector, initial_conditions, inputs, params, True, ideal_coefficients)
    np.savez("shared_data.npz", time_vector=time_vector, inputs=np.array(inputs, dtype=object), real_data=np.array(real_data))

    print("__________________STARTING_________________")
    multiprocessing.freeze_support()
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    population = toolbox.population(n=20)
    NGEN = 1000
    actual_error = []
    elite_size = 2
    mutp_change = 0.4
    mutp_base = 0.2
    num_to_regenerate = 1  # number of worst individuals to replace
    run_start_time = time.time()
    for gen in range(NGEN):

        # generate and mutate the population to generate offspring
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb= mutp_base + mutp_change * ( 1 - gen / NGEN))
        # calculate fitness of the offspring
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        
        # sort out the population for elitism
        if gen > 0:
            population = toolbox.select(offspring, k=len(population) - elite_size) + elites
        else:
            population = toolbox.select(offspring, k=len(population))
        
        # get the elites of this population
        elites = tools.selBest(population, elite_size)

        # --------- stats -------------------
        start_time = time.time()
        best = elites[0]
        actual_error.append(TSGA_utils.evaluate_coeffs_error(best))
        # Record and print statistics
        record = stats.compile(population)
        logbook.record(gen=gen, evals=len(population), **record)
        print(logbook.stream)
        
        #-------------- Ensure diversity ----------
        # replace the worst individuals
        worst = tools.selWorst(population, num_to_regenerate)
        regenerated = [toolbox.individual() for _ in range(num_to_regenerate)]
        # replace worst individuals in offspring
        for w, r in zip(worst, regenerated):
            population[population.index(w)] = r
    




    # ------------------------------------------------------------------

    best_ind = tools.selBest(population, 1)[0]
    print('Best Coefficients:')
    for i, coeff in enumerate(best_ind):
        print(f"Coefficient {i}: {coeff:.6f}, actual required value: {ideal_coefficients[i]:.6f}")

    with open("best_coefficients.json", "w") as f:
        json.dump(list(best_ind), f, indent=4)

    pool.close()
    pool.join()

    # Plot statistics summary
    gen = logbook.select("gen")
    min_fits = logbook.select("min")
    avg_fits = logbook.select("avg")

    sim_pos = run_sim.bare_simulate_model(time_vector, initial_conditions, inputs, params, True, best_ind)
    sim_pos = np.array([np.asarray(s) for s in sim_pos])
    real_data = np.array([np.asarray(s) for s in real_data])
    fig_best = plt.figure()
    ax_best = fig_best.add_subplot(111, projection='3d')
    ax_best.plot(sim_pos[:, 0], sim_pos[:, 1], sim_pos[:, 2], label='GA Best')
    ax_best.plot(real_data[:, 0], real_data[:, 1], real_data[:, 2], label='Ideal')
    ax_best.set_xlabel('X')
    ax_best.set_ylabel('Y')
    ax_best.set_zlabel('Z')
    ax_best.set_title('3D Trajectory - best one')
    ax_best.legend()


    # Combined plot: Coefficient error and Fitness statistics
    plt.figure()
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(gen, min_fits, label="Min Fitness", color='tab:blue')
    ax1.plot(gen, avg_fits, label="Avg Fitness", color='tab:orange')
    ax2.plot(gen, actual_error, label="Coeff Error from Ideal", color='tab:red', linestyle='--')

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax2.set_ylabel("Coefficient Error")
    ax1.set_title("Fitness & Coefficient Error over Generations")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    ax1.grid(True)
    plt.show()