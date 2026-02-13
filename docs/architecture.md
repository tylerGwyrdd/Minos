# Architecture Overview

## Goal

The repository is structured to compare guidance/control strategies under a
shared physics plant with minimal coupling between layers.

## Core modules

- `minos.physics`
  - 6-DoF parafoil plant model.
- `minos.gnc`
  - modular navigation, guidance, and control interfaces + implementations.
- `minos.sim.runners`
  - canonical simulation orchestration layer.
- `minos.identification`
  - coefficient identification and GA-based fitting workflows.
- `minos.utilities`
  - plotting, visualization, and snapshot helpers.

## Layer responsibilities

1. Physics computes state evolution from inputs.
2. GnC computes inputs from observations.
3. Simulation runners connect physics and GnC at runtime.
4. Utilities consume snapshot outputs for analysis.

## Closed-loop flow

1. Read simulator state and diagnostics.
2. Build `Observation`.
3. Run `GncStack.update(observation, dt)`.
4. Apply returned `Inputs` to the physics model.
5. Step plant and log `SimulationSnapshot`.

## Open-loop flow

1. Provide flap and wind schedules.
2. Runner applies schedule values each time step.
3. Plant advances and snapshots are logged.

## Why this structure

- Strategy comparison remains fair because plant and runner are fixed.
- New GnC methods can be added without physics edits.
- Identification pipelines can reuse the same simulation layer.
