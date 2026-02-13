# Physics Model identification

## Introduction

The refactored identification flow provides a typed, modular stack in
`minos.identification`:

- `FlightDataset`: validated measured data container.
- `TrajectoryEvaluator`: computes trajectory objective cost using the refactored physics simulator.
- `optimize_coefficients_ga`: DEAP-based optimizer decoupled from simulation internals.
- `run_position_identification_ga`: high-level runner for position-based fitting.

This lets old script workflows remain in place while enabling incremental migration to modular code.

## Design Rationale

The code is structured this way to keep runtime-heavy experiments manageable and
to avoid silent mismatches between optimization workflows and the simulation
plant.

1. Single coefficient order source:
   `CoefficientCodec` uses `AeroCoefficients.ORDER` directly, so coefficient
   vectors used by GA always map to the same names used by physics.
2. Typed dataset boundary:
   `FlightDataset` validates shapes once and exposes a simulation-input adapter.
   This prevents repeated shape checks and reduces per-evaluation overhead.
3. Separation of responsibilities:
   `TrajectoryEvaluator` owns objective math and simulation calls.
   `deap_ga` owns search logic.
   Experiment scripts only define scenario-specific data and outputs.
4. Compute-aware defaults:
   GA supports parallel evaluation and multi-fidelity (coarse-to-fine) scoring.
   This keeps long runs practical without changing physics equations.

## Runtime Reduction Features

The optimizer includes two built-in runtime controls:

1. Parallel fitness evaluation:
   Set `GAConfig.n_jobs`.
   `1` means serial, `0` means auto (`cpu_count - 1`), and values `>1` set
   explicit worker count.
2. Multi-fidelity evaluation:
   Enable with `GAConfig.enable_multi_fidelity=True` and tune:
   `coarse_stride`, `coarse_until_fraction`, and `coarse_top_k_full`.
   Early generations run on downsampled trajectories and top candidates are
   re-scored at full fidelity to keep ranking quality.

Example:

```python
import numpy as np
from minos.identification import FlightDataset, GAConfig, run_position_identification_ga

dataset = FlightDataset(
    time_s=time_s,
    flap_left=flap_left,
    flap_right=flap_right,
    wind_inertial=wind,
    measured_positions=measured_positions,
)

result = run_position_identification_ga(
    dataset,
    params={"initial_pos": [0.0, 0.0, 500.0]},
    initial_conditions=initial_conditions,
    optimize_names=["CDo", "CDa", "CLo", "CLa"],
    ga_config=GAConfig(population_size=30, generations=100, seed=7),
)
```

## Experiment Scripts

Three scripts are maintained as concrete experiment entry points:

1. `src/minos/identification/Simple_GA.py`
   Full coefficient vector optimization in one run.
2. `src/minos/identification/mod_v2.py`
   Group-focused optimization for a selected coefficient subset.
3. `src/minos/identification/TSGA_modified.py`
   Sequential grouped optimization where later groups start from earlier updates.

All three scripts share the same evaluator and GA core, which keeps behavior
consistent and easier to benchmark.

## TSGA API reference

::: minos.identification.TSGA_modified
