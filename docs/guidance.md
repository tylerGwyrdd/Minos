# Guidance, Navigation, and Control (GnC)

## Introduction

Parafoil GnC sits at the intersection of path planning, wind-aware guidance,
and low-level actuation under strong model uncertainty. Unlike many fixed-wing
systems, parafoils are highly wind sensitive, have limited control authority
(typically brake/flap asymmetry and symmetry), and often operate in terminal
descent conditions where timing and geometry are tightly coupled.

In the broader field, parafoil guidance strategies commonly combine:

- phase-based mission logic (for example initialization, homing, energy
  management, final approach),
- online wind estimation from inertial motion,
- and heading tracking control mapped into asymmetric flap/brake commands.

The practical challenge is not only designing each algorithm, but integrating
them so strategies can be swapped and compared fairly against the same physics
plant and scenario conditions.

This project’s GnC architecture is designed for that comparison workflow.

## Purpose

The GnC layer produces plant inputs from observed vehicle behavior.

- Input: current simulation observation (`state`, inertial kinematics, wind,
  rates, time).
- Output: flap commands (and selected wind signal usage) packaged as physics
  `Inputs`.

The key design goal is modular strategy composition:

- navigation is replaceable,
- guidance is replaceable,
- control is replaceable,
- orchestration is consistent.

## System Layout

### 1) GnC interfaces (`minos.gnc.interfaces`)

Core contracts define component boundaries:

- `Navigator`
- `GuidanceLaw`
- `Controller`
- optional `PhaseManager`

Shared typed payloads:

- `Observation`
- `NavigationEstimate`
- `GuidanceCommand`
- `ControlCommand`
- `MissionContext`

These contracts are the stable integration surface for new strategies.

### 2) GnC orchestration (`minos.gnc.stack`)

`GncStack` executes one closed-loop cycle:

1. navigation update,
2. optional phase update,
3. guidance update,
4. controller update,
5. command validation/clipping,
6. return plant `Inputs`.

`GncStackConfig` controls cross-cutting behavior such as flap limits and wind
source selection.

### 3) Current strategy set (`minos.gnc.*`)

- Navigation:
  - `navigation/wind_estimator_rls.py` (`RlsWindEstimator`)
- Guidance:
  - `guidance/t_approach.py` (`TApproachGuidance`)
  - `guidance/t_approach_2.py` (`TApproachGuidance2`)
  - `guidance/timed_heading_sequence.py` (`TimedHeadingSequenceGuidance`)
- Control:
  - `control/pid_heading.py` (`PidHeadingController`)
- Adapters:
  - `adapters/from_sim.py` (`observation_from_sim`)

## How GnC Connects to Physics

The physics model (`minos.physics`) remains the plant and does not embed control
policy.

Closed-loop execution is handled in `minos.sim.runners.run_simulation_with_gnc`:

1. Read plant state/diagnostics.
2. Build `Observation`.
3. Run `GncStack.update(observation, dt)`.
4. Apply returned `Inputs` to `ParafoilModel6DOF`.
5. Step physics and log `SimulationSnapshot`.

This separation keeps strategy logic out of plant dynamics and makes A/B testing
of GnC variants straightforward.

## Wind Estimation in the Loop

Wind estimation belongs to the navigation layer.

- `RlsWindEstimator` estimates horizontal wind from observed inertial velocity.
- Guidance consumes the estimate for path geometry decisions.
- Controller consumes guidance commands (and current rates) to generate flaps.

By default, the closed-loop runner keeps plant wind as scenario truth while GnC
uses estimated wind internally. This preserves realistic estimation error during
evaluation.

## Metrics and Benchmarking

Recent additions provide machine-readable GnC metrics and benchmarking helpers:

- `minos.gnc.metrics`
  - `compute_run_metrics`
  - `aggregate_metrics`
- `minos.gnc.benchmark`
  - `ScenarioConfig`
  - `run_benchmark_scenario`
  - `run_benchmark_suite`

These are intended for fair A/B comparison across guidance/controller variants
using the same plant and scenario definitions.

Snapshot logs now also carry GnC telemetry useful for analysis:

- mission phase,
- flare command,
- estimated wind,
- raw vs clipped flap commands.

## What To Change For New GnC Experiments

- New estimator:
  - add a `Navigator` implementation under `minos.gnc.navigation`.
- New guidance law:
  - add a `GuidanceLaw` implementation under `minos.gnc.guidance`.
- New control law:
  - add a `Controller` implementation under `minos.gnc.control`.
- New mission phase logic:
  - add a `PhaseManager` and wire it into `GncStack`.

Use the same runner and physics plant to keep comparisons fair.

## Examples

- Closed-loop modular GnC example:
  - `examples/gnc_3d_example.py`
- Open-loop physics baseline:
  - `examples/physics_model_3d_example.py`

These examples are the recommended entry points for extending GnC behavior.
