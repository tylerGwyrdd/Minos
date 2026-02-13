import random
import numpy as np
from deap import base, creator, tools, algorithms
import multiprocessing
from minos.sim import runners as run_sim
import json

import TSGA_utils as TSGA_utils
from TSGA_utils import ideal_coeffs, aero_coeffs, bounds_dict

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

NGENS = 500
NPOP = 30

def unique_individuals(pop):
    """Work out how close genes are between individuals and return the number of different individuals"""
    return len(set(tuple(round(g, 5) for g in ind) for ind in pop))

def compute_genome_variance(population):
    """Computes variance of each gene across the population and returns mean variance."""
    genomes = np.array([ind[:] for ind in population])
    if len(genomes) == 0:
        return 0.0
    return np.mean(np.var(genomes, axis=0))

def generate_ideal_individual_from_set(names):
    """
    Generates an individual with genes equal to the corresponding ideal coefficients.
    Only includes coefficients specified in `names`.
    """
    return creator.Individual([ideal_coeffs[name] for name in names])

# Custom individual generator based on bounds

def generate_individual_from_set(names):
    """Creates from set. Generates individual with just the named coeffs as genes"""
    std_dev = 0.1
    base_point = [aero_coeffs[name] for name in names]
    return creator.Individual([
        min(max(np.random.normal(loc=base, scale=max(std_dev, 1e-6)),
                bounds_dict[name][0]), bounds_dict[name][1])
        for name, base in zip(names, base_point)
    ])

def adaptive_gaussian_mutation(individual, population, low, high, indpb=1.0):
    """
    Mutate an individual using Gaussian noise scaled by population standard deviation.
    
    Args:
        individual: the DEAP individual to mutate
        population: list of all individuals in the current population
        low, high: lists of bounds for each gene
        indpb: probability of mutating each gene
    
    Returns
    -------
        tuple containing the mutated individual
    """
    genomes = np.array([ind[:] for ind in population])
    std_devs = np.std(genomes, axis=0)

    for i in range(len(individual)):
        if random.random() < indpb:
            std = max(std_devs[i], 1e-6)  # prevent zero std
            individual[i] += random.gauss(0, std)
            individual[i] = min(max(individual[i], low[i]), high[i])  # clamp to bounds

    return (individual,)

def make_toolbox(names, lows, highs, real_data, inputs, sets):
    toolbox = base.Toolbox()
    # random setting from of the specific coefficients
    toolbox.register("individual_uniform", 
                     lambda: creator.Individual([random.uniform(l, h) for l, h in zip(lows, highs)]))
    toolbox.register("individual_gauss", lambda: generate_individual_from_set(names))
    toolbox.register(
    "individual_from_partial",
    lambda: creator.Individual(
        TSGA_utils.generate_individual_from_partials_dict(sets, names, aero_coeffs, bounds_dict, 0.1)))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual_from_partial)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=5, low=lows, up=highs)
    toolbox.register("mutate", lambda ind: adaptive_gaussian_mutation(ind, toolbox.population_ref, lows, highs, indpb=1.0))
    # toolbox.register("mutate", tools.mutPolynomialBounded, eta=5, low=lows, up=highs, indpb=1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    def evaluate(individual):
        coeffs = aero_coeffs.copy()
        for i, name in enumerate(names):
           coeffs[name] = individual[i]
        # get the states
        simulated_states, broken_index  = run_sim.multi_obj_sim(time_vector, initial_conditions, inputs, params, True, coeffs)
        state_array = np.array(simulated_states)
        real_array = np.array(real_data[:broken_index+1])
        # position
        pos = state_array[:, 0, :]
        ideal_pos = real_array[:, 0, :]
        positional_error = np.sqrt(np.mean(np.square(np.linalg.norm(pos - ideal_pos, axis=1))))
        
        combinational_error = positional_error
        error =  combinational_error + combinational_error * (len(time_vector) - broken_index)
        return (error,) if np.isfinite(error) else (1e6,)

    toolbox.register("evaluate", evaluate)
    return toolbox

def optimize_coefficients(names, real_data, inputs, sets):
    print("starting to optimise:", names)
    import time
    lows = [bounds_dict[name][0] for name in names]
    highs = [bounds_dict[name][1] for name in names]

    toolbox = make_toolbox(names, lows, highs, real_data, inputs, sets)
    pop = toolbox.population(n=NPOP)
    hof = tools.HallOfFame(1)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals", "avg", "std", "act", "uniq", "gvar"]
 
    start_time = time.time()
    actual_errors = []

    # for elitism
    elite_size = 1
    mutp_change = 0.4
    mutp_base = 0.2

    # for stats
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    for gen in range(NGENS):
        # lets not mutate the first generation, see where we are at. 
        toolbox.population_ref = pop  # update for mutation
        if gen > 0:
            mutation_chance = mutp_base + mutp_change * ( 1 - gen / NGENS)
            offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb = mutation_chance)
        else:
            offspring = toolbox.select(pop, k=len(pop))

        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
                # sort out the population for elitism
        uniq = unique_individuals(pop)
        if gen > 0 and uniq > 4:
            pop = toolbox.select(offspring, k=len(pop) - elite_size) + elites
        else:
            pop = toolbox.select(offspring, k=len(pop))
        
        # get the elites of this population
        elites = tools.selBest(pop, elite_size)

        # --------- Stats -----------------
        hof.update(pop)
        record = stats.compile(pop)

        av_act_error = 0
        for p in pop:
            av_act_error += TSGA_utils.average_actual_error(p,names)
        av_act_error  = av_act_error / NPOP
        record["act"] =av_act_error
        record["uniq"] = unique_individuals(pop)
        record["gvar"] = compute_genome_variance(pop)
        logbook.record(gen=gen, nevals=len(pop), **record)

        #actual_error = TSGA_utils.evaluate_partial_coeffs_error(elites[0], names)
        #actual_errors.append(actual_error)
        #mean_act_error = np.mean(actual_error)
        if gen % 10 == 0:
            best = elites[0].fitness.values[0]
            avg = record["avg"]
            std = record["std"]
            
            print(f"Run: {names} | Gen: {gen} | Best: {best:.6f} | Avg: {avg:.6f} | Std: {std:.6f} | mean_error:{av_act_error:.6f} | Uniq : {uniq} / {len(pop)}")
        
        if gen % 20 == 0 and gen != 0:
            for i, name in enumerate(names):
                print(f"Best for {name}: {hof[0][i]:.6f} ideal: {ideal_coeffs[name]:.6f} ")
        


    # ----------------------------------------------------------------------

    elapsed_time = time.time() - start_time
    best_vals = hof[0]
    print("completed estimations for the set: ", names)
    """for i, name in enumerate(names):

        print(f"Best for {name}: {best_vals[i]:.6f} (ideal: {ideal_coeffs[name]:.6f})")
    """
    #coeffs = aero_coeffs.copy()
    #for i, name in enumerate(names):
    #    coeffs[name] = best_vals[i]
    #real_pos = [state[0] for state in real_data]
    #estimated = run_sim.bare_simulate_model(time_vector,real_pos, initial_conditions, inputs, params, coeffs,False)
    #TSGA_utils.plot_3D_position(estimated)
    return {
        "best_values": dict(zip(names, best_vals)),
        "fitness_over_time": logbook,
        "time_seconds": elapsed_time
    }

def wrapped_optimize(args):
    names, real_data, inputs, sets = args
    result = optimize_coefficients(names, real_data, inputs, sets)
    return (" + ".join(names), result)

if __name__ == '__main__':
    wind_vect = np.array([0, 0, 0])
    wind_list = [wind_vect.copy() for _ in range(time_vector.size)]
    grouped_names = [
              #  ["CDo", "CDa", "CLo", "CLa", "Cmo", "Cma", "Cmq", "CL_sym", "CD_sym"],
            #["CYB", "ClB", "Clp", "CLa", "Clr", "Cl_asym", "CnB", "Cn_p", "Cn_r","Cn_asym"],
            #["CDo", "CDa", "CLo", "CLa"],
            ["CDo", "CDa", "CD_sym", "CYB", "CLo", "CLa", "CL_sym", "Cmo", "Cma", "Cmq","ClB", "Clp", "Clr", "Cl_asym", "CnB", "Cn_p", "Cn_r", "Cn_asym"]
    ]
    """        ["CDo", "CDa", "CLo", "CLa", "Cmo", "Cma", "Cmq", "CL_sym", "CD_sym"],
            ["CYB", "ClB", "Clp", "CLa", "Clr", "Cl_asym", "CnB", "Cn_p", "Cn_r","Cn_asym"],
            ["CDo", "CDa", "CLo", "CLa"],
            }
            ["CDo"],
            ["CDa"],
            ["CD_sym"],
            ["CYB"],
            ["CLo"],
            ["CLa"],
            ["CL_sym"],
            ["Cmo"],
            ["Cma"],
            ["Cmq"],
            ["ClB"],
            ["Clp"],
            ["Clr"],
            ["Cl_asym"],
            ["CnB"],
            ["Cn_p"],
            ["Cn_r"],
            ["Cn_asym"]
            ]
        """
    Testing2 = {
        "CDo": 0.27901115628038753,
        "CDa": 0.06072191338987484,
        "CLo": 7.421838385622049e-05,
        "CLa": 0.6099110995940484,
        "Cmo": 0.6814064658988337,
        "Cma": -0.6994146874974438,
        "Cmq": -2.4174060506386117,
        "CL_sym": 0.12469958262887637,
        "CD_sym": 0.06246955246041637,
        "CYB": -0.23723366643056895,
        "ClB": -0.08725961866797849,
        "Clp": -0.5578617654322627,
        "CLa": 1.2045475362244498,
        "Clr": -0.07091641614766138,
        "Cl_asym": -0.0009115793536172734,
        "CnB": -0.0006920595289966945,
        "Cn_p": -0.13225474760260356,
        "Cn_r": -0.3270419622562816,
        "Cn_asym": 0.026246073978055666
    }
    normal_coeffs = {
        "CDo": 0.2874163139948833,
        "CDa": 0.08444219629801189,
        "CD_sym": 0.12846942144459808,
        "CYB": -0.3766334167093541,
        "CLo": 0.16996745187581644,
        "CLa": 0.6746912561504149,
        "CL_sym": 0.11308127007839822,
        "Cmo": 0.45316266636409985,
        "Cma": -0.625506728752408,
        "Cmq": -0.831110991198116,
        "ClB": -0.059459704510160444,
        "Clp": -0.5289276957537661,
        "Clr": -0.07740528453133262,
        "Cl_asym": -0.006369705992741746,
        "CnB": -9.135062815230092e-05,
        "Cn_p": -0.11044012915060586,
        "Cn_r": -0.326482921930228,
        "Cn_asym": 0.022727309145881603

    }
    single_coeffs = {
        "CDo": 0.3734264119264236,
        "CDa": 0.2499999999999994,
        "CD_sym": 0.10808686701245557,
        "CYB": -0.05896802530093083,
        "CLo": 0.1999999999999989,
        "CLa": 1.180636419463905,
        "CL_sym": 0.49999999999982536,
        "Cmo": 0.5494638311218906,
        "Cma": -0.4465929853972019,
        "Cmq": -4.254182602949606e-14,
        "ClB": -0.03241167573360374,
        "Clp": -0.8639237943586331,
        "Clr": -7.453282743840536e-16,
        "Cl_asym": -1.0854087793537988e-17,
        "Cn_p": -4.079250027162165e-17,
        "Cn_r": -0.15655578729558184,
        "Cn_asym": 0.019630257041709634,
    }
    targeted_1 = {
        "CYB": -0.23723366643056895,
        "ClB": -0.08725961866797849,
        "Clp": -0.5578617654322627,
        "CLa": 1.2045475362244498,
        "Clr": -0.07091641614766138,
        "Cl_asym": -0.0009115793536172734,
        "CnB": -0.0006920595289966945,
        "Cn_p": -0.13225474760260356,
        "Cn_r": -0.3270419622562816,
        "Cn_asym": 0.026246073978055666        
    }
    targeted_2 = {
        "CDo": 0.27901115628038753,
        "CDa": 0.06072191338987484,
        "CLo": 7.421838385622049e-05,
        "CLa": 0.6099110995940484,
        "Cmo": 0.6814064658988337,
        "Cma": -0.6994146874974438,
        "Cmq": -2.4174060506386117,
        "CL_sym": 0.12469958262887637,
        "CD_sym": 0.06246955246041637        
    }
    targeted_3 = {
        "CDo": 0.3586867541489667,
        "CDa": 0.130285817511622,
        "CLo": 0.13068959458392845,
        "CLa": 1.155143210723681        
    }
    
    testing = {
        "CDo": 0.16818496690257798,
        "CDa": 0.21900169724977633,
        "CD_sym": 4.421982178681186e-06,
        "CYB": -6.412949814679987e-05,
        "CLo": 0.0020474173733351516,
        "CLa": 0.8674121186480263,
        "CL_sym": 0.18403387779097813,
        "Cmo": 0.5928306348367021,
        "Cma": -0.7483580859373181,
        "Cmq": -2.6370738419968744,
        "ClB": -0.09049594292385367,
        "Clp": -0.9969575569659275,
        "Clr": -0.17333913098107334,
        "Cl_asym": -0.007608922257938582,
        "CnB": -0.00540268419011263,
        "Cn_p": -0.11452530422548163,
        "Cn_r": -0.3441113873193863,
        "Cn_asym": 0.022457115249920397
    }
    partial_dicts = [targeted_1, targeted_2, targeted_3]

    real_datasets = []
    sets_list = []
    inputs_list = []

    left, right = TSGA_utils.genertate_spirial_flight(time_vector)
    inputs_list.append([left, right, wind_list])

    a_data_set = TSGA_utils.generate_real_state_data(time_vector, left, right, wind_list, params, initial_conditions)
    starting,broke = run_sim.bare_simulate_model(time_vector, initial_conditions, inputs_list[0], params,True,testing, False)
    pos = np.array([np.array(p[0]) for p in a_data_set])
    positional_error = np.sqrt(np.mean(np.square(np.linalg.norm(pos - np.array(starting), axis=1))))
    print(positional_error)
    error =  positional_error + positional_error * (len(time_vector) - broke)
    print(error)
    TSGA_utils.plot_3D_geometry_comparision(time_vector,pos, initial_conditions, inputs_list[0], params,testing)
    TSGA_utils.plot_3D_position(starting)
    
    for i in grouped_names:
        inputs_list.append([left, right, wind_list])
        real_datasets.append(a_data_set)
        sets_list.append(partial_dicts)
    tasks = list(zip(grouped_names, real_datasets, inputs_list, sets_list))

    print("RUNNIN")


    with multiprocessing.Pool() as pool:
        results = pool.map(wrapped_optimize, tasks)

    

    best_coeffs = {group: res["best_values"] for group, res in results}
    metrics = {
        group: {
            "fitness_over_time": [dict(rec) for rec in res["fitness_over_time"]],
            "time_seconds": res["time_seconds"],
        }
        for group, res in results
    }

    filename = "step2_500gen_30pop"
    with open(filename + "_best_coeffs.json", "w") as f:
        json.dump(best_coeffs, f, indent=4)

    with open(filename + "_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Step 1 complete. Best coefficients written to tsga_step1_best.json")
    TSGA_utils.plot_metrics_individually("modified_tsga_metrics.json")
    TSGA_utils.plot_individual_errors("modified_tsga_metrics.json")
    



