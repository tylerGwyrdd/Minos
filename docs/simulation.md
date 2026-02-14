# Simulation Runners

## Canonical API

`minos.sim.runners` is the single simulation entrypoint module.

## Open-loop runners

- `simulate_model`
  - full typed snapshots plus state history.
- `bare_simulate_model`
  - position trajectory only.
- `multi_obj_sim`
  - state trajectory for optimization workflows.
- `sim_with_noise`
  - noisy position measurements.
- `sim_state_with_noise`
  - noisy state trajectories.

Use open-loop runners when flap/wind inputs are pre-defined.

## Closed-loop runner

- `run_simulation_with_gnc`
  - integrates `ParafoilModel6DOF` with `GncStack` in a feedback loop.

Use this runner for guidance/control strategy testing.

## Inputs and outputs

All runners are built around:

- physics inputs (`Inputs`) and state (`State`) for plant execution,
- `SimulationSnapshot` logs for analysis and plotting.

`SimulationSnapshot` is the common data product across open-loop and
closed-loop runs.

For closed-loop GnC runs, snapshots may additionally include:

- guidance phase label,
- flare magnitude command,
- navigation wind estimate,
- raw and clipped control commands.

## Design guidance

When adding new simulation behavior:

1. Extend `minos.sim.runners` instead of embedding loops into physics or GnC modules.
2. Keep runner responsibilities orchestration-only.
3. Keep strategy logic inside `minos.gnc` and dynamics inside `minos.physics`.
