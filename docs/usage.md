# Usage

## Prerequisites

- Python `3.12+`
- Poetry

## Installation

```bash
git clone <repo-url>
cd Level_4_project
poetry install
```

## Recommended execution pattern

Use Poetry without manually activating environments:

```bash
poetry run python <script.py>
```

## First runs

Open-loop physics + plots + 3D animation:

```bash
poetry run python examples/physics_model_3d_example.py
```

Closed-loop modular GnC example:

```bash
poetry run python examples/gnc_3d_example.py
```

Closed-loop T-approach v2 example:

```bash
poetry run python examples/gnc_t_approach_2_example.py --no-show
```

Headless GnC smoke run:

```bash
poetry run python examples/gnc_3d_example.py --no-show
```

Parallel PID heading-step sweep (process backend):

```bash
poetry run python examples/parallel_gnc_pid_step_sweep.py --backend process --workers 8 --no-show
```

If process pools are restricted in your environment, use thread backend:

```bash
poetry run python examples/parallel_gnc_pid_step_sweep.py --backend thread --workers 6 --no-show
```

## Canonical simulation APIs

Use `minos.sim.runners` as the main integration surface:

- `simulate_model` for open-loop schedule-driven runs.
- `run_simulation_with_gnc` for closed-loop GnC runs.
- `bare_simulate_model`, `multi_obj_sim`, and noise variants for identification workflows.

## Typical workflow

1. Build scenario and initial state in physics terms.
2. Choose open-loop or closed-loop runner.
3. Run simulation to produce `SimulationSnapshot` data.
4. Plot/analyze with utilities or custom scripts.
