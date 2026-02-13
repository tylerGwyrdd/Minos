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

## How to use examples for development

1. Start from the closest example.
2. Change only scenario parameters first (wind, initial conditions, target).
3. Then swap GnC components if needed.
4. Keep output snapshots and plot selections consistent for fair comparison.
