"""Scenario-based benchmarking helpers for GnC method comparison."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from minos.physics.model import ParafoilModel6DOF
from minos.physics.types import Inputs, State
from minos.utilities.snapshots import SimulationSnapshot

from .interfaces import MissionContext
from .metrics import GncAggregateMetrics, GncRunMetrics, aggregate_metrics, compute_run_metrics
from .stack import GncStack


@dataclass(frozen=True)
class ScenarioConfig:
    """Deterministic scenario definition for fair method comparison."""

    name: str
    dt: float
    steps: int
    params: dict[str, object]
    initial_state: list[np.ndarray]
    wind_inertial: np.ndarray
    mission_factory: Callable[[], MissionContext]
    ipi: np.ndarray | None = None
    seed: int | None = None


@dataclass(frozen=True)
class BenchmarkRunResult:
    """Result of running one method on one scenario."""

    scenario_name: str
    method_name: str
    snapshots: list[SimulationSnapshot]
    metrics: GncRunMetrics


def run_benchmark_scenario(
    scenario: ScenarioConfig,
    *,
    method_name: str,
    stack_factory: Callable[[MissionContext], GncStack],
) -> BenchmarkRunResult:
    """Execute one closed-loop benchmark scenario and return snapshots + metrics."""
    from minos.sim.runners import run_simulation_with_gnc

    if scenario.seed is not None:
        np.random.seed(int(scenario.seed))

    mission = scenario.mission_factory()
    stack = stack_factory(mission)
    sim = ParafoilModel6DOF(
        params=scenario.params,
        initial_state=State.from_sequence(scenario.initial_state),
        initial_inputs=Inputs(0.0, 0.0, np.asarray(scenario.wind_inertial, dtype=float)),
    )
    snapshots = run_simulation_with_gnc(sim=sim, steps=int(scenario.steps), dt=float(scenario.dt), gnc_stack=stack)
    metrics = compute_run_metrics(
        snapshots,
        scenario_name=scenario.name,
        method_name=method_name,
        ipi=scenario.ipi,
        max_flap_abs=stack.config.max_flap_abs,
        wind_truth=np.asarray(scenario.wind_inertial, dtype=float),
    )
    return BenchmarkRunResult(
        scenario_name=scenario.name,
        method_name=method_name,
        snapshots=snapshots,
        metrics=metrics,
    )


def run_benchmark_suite(
    scenarios: Sequence[ScenarioConfig],
    *,
    methods: Sequence[tuple[str, Callable[[MissionContext], GncStack]]],
) -> tuple[list[BenchmarkRunResult], dict[str, GncAggregateMetrics]]:
    """Run all method/scenario combinations and aggregate by method."""
    results: list[BenchmarkRunResult] = []
    by_method: dict[str, list[GncRunMetrics]] = {}
    for scenario in scenarios:
        for method_name, factory in methods:
            out = run_benchmark_scenario(scenario, method_name=method_name, stack_factory=factory)
            results.append(out)
            by_method.setdefault(method_name, []).append(out.metrics)
    agg = {method_name: aggregate_metrics(metrics) for method_name, metrics in by_method.items()}
    return results, agg
