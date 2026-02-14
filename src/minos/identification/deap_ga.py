"""DEAP-based optimizer for aerodynamic coefficient identification.

This wrapper intentionally centralizes GA behavior so experiment scripts only
declare datasets, coefficient groups, and output handling.
"""

from __future__ import annotations

import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
from deap import algorithms, base, creator, tools

from minos.identification.core import CoefficientBounds, CoefficientCodec
from minos.identification.evaluator import EvaluationResult
from minos.physics.types import AeroCoefficients


@dataclass(frozen=True)
class GAConfig:
    """Genetic algorithm configuration.

    Performance-related fields (`n_jobs`, multi-fidelity options) are here so
    experiments can trade runtime vs. fidelity without code changes.
    """

    population_size: int = 40
    generations: int = 120
    cxpb: float = 0.5
    mutpb: float = 0.3
    mutation_eta: float = 10.0
    mutation_indpb: float = 0.25
    tournament_size: int = 3
    elite_size: int = 2
    n_jobs: int = 1
    enable_multi_fidelity: bool = False
    coarse_stride: int = 3
    coarse_until_fraction: float = 0.6
    coarse_top_k_full: int = 2
    seed: int | None = None


@dataclass(frozen=True)
class GAResult:
    """Optimization result bundle."""

    best_coefficients: dict[str, float]
    best_cost: float
    history: list[dict[str, float]]


def _ensure_deap_types() -> None:
    # DEAP stores created classes globally; guarded creation prevents errors
    # when multiple scripts import and execute in the same interpreter.
    if not hasattr(creator, "FitnessMinIdent"):
        creator.create("FitnessMinIdent", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "IndividualIdent"):
        creator.create("IndividualIdent", list, fitness=creator.FitnessMinIdent)


def optimize_coefficients_ga(
    evaluate_coefficients: Callable[[Mapping[str, float], int], EvaluationResult],
    bounds: CoefficientBounds,
    *,
    optimize_names: Sequence[str] | None = None,
    base_coefficients: Mapping[str, float] | None = None,
    config: GAConfig = GAConfig(),
) -> GAResult:
    """Optimize aerodynamic coefficients with a DEAP single-objective GA.

    Design choices
    --------------
    1. Partial optimization support is first-class for grouped coefficient runs.
    2. Threaded evaluation avoids process-spawn overhead and sandbox issues.
    3. Optional coarse/fine evaluation reduces compute in early generations.
    """
    _ensure_deap_types()
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)

    codec = CoefficientCodec()
    all_names = list(AeroCoefficients.ORDER)
    names = list(optimize_names) if optimize_names is not None else all_names
    if base_coefficients is not None:
        base_coeffs = dict(base_coefficients)
    else:
        base_coeffs = codec.to_dict(codec.to_vector(AeroCoefficients()))
    lows, highs = bounds.for_names(names)

    toolbox = base.Toolbox()
    toolbox.register(
        "individual",
        lambda: creator.IndividualIdent([random.uniform(lo, hi) for lo, hi in zip(lows, highs)]),
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=10.0, low=lows, up=highs)
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        eta=config.mutation_eta,
        low=lows,
        up=highs,
        indpb=config.mutation_indpb,
    )
    toolbox.register("select", tools.selTournament, tournsize=config.tournament_size)

    def evaluate(individual: Sequence[float], sample_stride: int = 1) -> tuple[float]:
        coeff_dict = codec.merge_partial(base_coeffs, names, individual)
        result = evaluate_coefficients(coeff_dict, sample_stride=sample_stride)
        return (float(result.cost),)

    toolbox.register("evaluate", evaluate)

    population = toolbox.population(n=config.population_size)
    history: list[dict[str, float]] = []
    resolved_jobs = int(config.n_jobs)
    if resolved_jobs == 0:
        # `0` means "auto" to keep user scripts concise.
        resolved_jobs = max(1, (os.cpu_count() or 1) - 1)
    resolved_jobs = max(1, resolved_jobs)

    # Thread backend keeps callable wiring simple (no pickling of evaluator
    # closures). This is robust in notebook/sandbox contexts.
    # TODO: Intent unclear — appears to assume objective runtime is dominated by NumPy/SciPy calls that release the GIL. Confirm.
    executor = ThreadPoolExecutor(max_workers=resolved_jobs) if resolved_jobs > 1 else None
    eval_map = executor.map if executor is not None else map

    try:
        for gen in range(config.generations):
            use_coarse = (
                config.enable_multi_fidelity
                and gen < int(config.generations * float(config.coarse_until_fraction))
                and int(config.coarse_stride) > 1
            )
            sample_stride = int(config.coarse_stride) if use_coarse else 1

            offspring = algorithms.varAnd(population, toolbox, cxpb=config.cxpb, mutpb=config.mutpb)
            # `eval_map` preserves input order, which is required because fitness
            # values are zipped back onto `offspring` by position.
            fits = list(eval_map(lambda ind: toolbox.evaluate(ind, sample_stride=sample_stride), offspring))
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit

            if use_coarse and config.coarse_top_k_full > 0:
                # Re-score a few best coarse candidates at full fidelity to
                # reduce ranking noise from downsampling.
                top = tools.selBest(offspring, k=min(int(config.coarse_top_k_full), len(offspring)))
                # Full-fidelity rescoring only on top coarse candidates limits
                # compute while protecting selection quality.
                top_fits = list(eval_map(lambda ind: toolbox.evaluate(ind, sample_stride=1), top))
                for fit, ind in zip(top_fits, top):
                    ind.fitness.values = fit

            elites = tools.selBest(population, k=min(config.elite_size, len(population)))
            survivors = toolbox.select(offspring, k=max(0, len(population) - len(elites)))
            population = survivors + elites

            costs = [ind.fitness.values[0] for ind in population if ind.fitness.valid]
            if costs:
                history.append(
                    {
                        "gen": float(gen),
                        "min": float(np.min(costs)),
                        "avg": float(np.mean(costs)),
                        "max": float(np.max(costs)),
                        "stride": float(sample_stride),
                    }
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    best = tools.selBest(population, k=1)[0]
    best_coeffs = codec.merge_partial(base_coeffs, names, best)
    best_cost = float(best.fitness.values[0])
    return GAResult(best_coefficients=best_coeffs, best_cost=best_cost, history=history)
