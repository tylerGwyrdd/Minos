# Examples

## Recommended examples

### `examples/physics_model_3d_example.py`

Open-loop physics demonstration:

- builds a simple flap/wind schedule,
- runs `simulate_model`,
- plots selected channels,
- renders interactive 3D motion.

### `examples/gnc_3d_example.py`

Closed-loop modular GnC demonstration:

- builds `ParafoilModel6DOF`,
- builds `GncStack` (`Navigator` + `GuidanceLaw` + `Controller`),
- runs `run_simulation_with_gnc`,
- plots diagnostics and headings,
- renders 3D motion.

Headless mode:

```bash
poetry run python examples/gnc_3d_example.py --no-show
```

### `examples/gnc_t_approach_2_example.py`

Closed-loop run using the pseudocode-driven `TApproachGuidance2` mode machine.
Prints run metrics (landing error, heading RMSE, control effort, wind estimate error).

### `examples/gnc_metrics_example.py`

Scenario-based GnC metric evaluation:

- executes multiple method variants on the same scenario,
- computes per-run + aggregate metrics,
- writes `gnc_metrics_report.json`.

### `examples/identification_metrics_example.py`

Shows shared trajectory metrics in the identification evaluator path
(RMSE/P95 + penalty factor).

### `examples/parallel_gnc_pid_step_sweep.py`

Parallel PID gain sweep for steering-only step-response tests:

- no wind,
- left/right 90-degree heading maneuvers,
- prints best gains and plots desired vs actual heading.

## How to use examples for development

1. Start from the closest example.
2. Change only scenario parameters first (wind, initial conditions, target).
3. Then swap GnC components if needed.
4. Keep output snapshots and plot selections consistent for fair comparison.
