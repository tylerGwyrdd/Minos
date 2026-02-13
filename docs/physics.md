# Physics Simulator

## Introduction

Parafoil dynamics have been modeled at multiple levels of fidelity in the
literature, from low-order 3-DoF and 4-DoF approximations to full 6-DoF and
higher-order coupled canopy-payload models.

Lower-order models are often useful for early guidance and control design
because they are computationally light and easier to analyze, but they can miss
important coupled rotational and aerodynamic effects needed for accurate
closed-loop behavior prediction.

Higher-fidelity models (for example 6-DoF rigid-body models, and 8/9-DoF models
that include additional canopy-payload coupling) improve physical realism, but
increase nonlinearity, parameter count, and computational cost. In practice,
parafoil simulation frameworks must balance:

- model fidelity,
- identification effort (many aerodynamic coefficients),
- numerical robustness,
- and runtime cost for repeated guidance/control experiments.

This project uses a 6-DoF physics core as the main fidelity level for
closed-loop evaluation. The intent is to preserve enough dynamic realism for
meaningful GnC comparisons while keeping simulation throughput high enough for
strategy iteration and coefficient identification workflows.

## Purpose

The physics layer is the plant model for the project.  
It provides a deterministic 6-DoF parafoil simulation that can be driven by:

- open-loop input schedules (for baseline analysis and identification), or
- closed-loop GnC logic (for strategy comparison).

The key design goal is separation of concerns:

- `minos.physics` owns vehicle dynamics.
- `minos.gnc` owns navigation, guidance, and control decisions.
- `minos.sim.runners` owns simulation loop orchestration.

## System Layout

### 1) Physics model (`minos.physics`)

The physics package is split by responsibility:

- `types.py`
  - typed containers for state, derivatives, inputs, coefficients, and parameters.
  - input validation and shape checks.
- `frames.py`
  - coordinate transforms and rotation helpers.
- `aero.py`
  - aerodynamic coefficient and load calculations.
- `dynamics.py`
  - pure derivative evaluation (`state_dot`) and diagnostics.
- `model.py`
  - stateful plant wrapper (`ParafoilModel6DOF`) that manages:
    - current state and inputs,
    - actuator lag for flap deflections,
    - time stepping (RK4),
    - cached diagnostics for logging/analysis.

This keeps the math testable (pure functions) while still providing a convenient simulation object.

### 2) Simulation runners (`minos.sim.runners`)

`minos.sim.runners` is the canonical place to run simulations.

- Open-loop runners:
  - `simulate_model`
  - `bare_simulate_model`
  - `multi_obj_sim`
  - `sim_with_noise`
  - `sim_state_with_noise`
- Closed-loop runner:
  - `run_simulation_with_gnc`

All runners use the same underlying physics model and return typed `SimulationSnapshot` logs for downstream plotting and analysis.

### 3) GnC stack (`minos.gnc`)

The GnC package uses composable interfaces:

- Navigator -> estimates quantities (for example wind).
- Guidance law -> computes high-level commands (for example desired heading).
- Controller -> maps guidance commands to flap deflections.
- `GncStack` -> orchestrates the above and outputs `Inputs` for the plant.

The physics model does not implement control policy; it only consumes `Inputs`.

## End-to-End Data Flow

### Open-loop flow

1. Provide time vector + flap/wind input series.
2. Runner applies those inputs each step.
3. Physics model advances state.
4. Snapshots are logged.

### Closed-loop GnC flow

1. Runner builds an `Observation` from the current simulator state/diagnostics.
2. `GncStack.update(observation, dt)` runs:
   - navigation,
   - guidance,
   - control.
3. Stack returns plant `Inputs` (`flap_left`, `flap_right`, wind).
4. Runner applies inputs to `ParafoilModel6DOF` and steps physics.
5. Runner logs `SimulationSnapshot` with dynamics and command context.

This architecture enables fair strategy comparison: swap GnC components while keeping the same physics plant and runner loop.

## What To Change For New Experiments

- Change physics assumptions:
  - edit `minos.physics` (coefficients, forces/moments, parameters).
- Change guidance/control strategy:
  - add or swap components in `minos.gnc`.
- Change how scenarios are executed or logged:
  - edit `minos.sim.runners`.

Keep those concerns separate to avoid coupling strategy logic into plant dynamics.

## Examples

- Open-loop physics example:
  - `examples/physics_model_3d_example.py`
- Closed-loop modular GnC example:
  - `examples/gnc_3d_example.py`

These are the recommended entry points for understanding and extending the current architecture.
