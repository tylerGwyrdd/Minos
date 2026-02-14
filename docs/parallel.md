# Parallel Execution

## Purpose

`minos.parallel` is a domain-agnostic task execution layer for running
independent jobs concurrently (for example parameter sweeps and tuning runs).

It is intentionally separate from physics/GnC logic:

- simulation code defines *what* to run,
- parallel code defines *how* tasks are scheduled/executed.

## API

- `TaskSpec`
  - serializable task definition (`task_id`, `callable_path`, args/kwargs).
- `TaskOutcome`
  - stable result envelope (`status`, `result`, `error`, runtime).
- `ParallelRunner`
  - backend selector and execution facade.

Backends:

- `sequential`
- `thread`
- `process`

## Backend Selection

Use:

- `process` for CPU-heavy simulation batches.
- `thread` when process creation is constrained or tasks are mostly I/O/C-extension heavy.
- `sequential` for debugging and deterministic step-through.

## Safety and Assumptions

1. Process backend requires importable top-level callables.
2. Results are returned in input task order, even if completion order differs.
3. `fail_fast` cancels pending work but cannot force-stop already running tasks.

## Example

Parallel PID step-response sweep:

```bash
poetry run python examples/parallel_gnc_pid_step_sweep.py --backend process --workers 8
```

Thread fallback:

```bash
poetry run python examples/parallel_gnc_pid_step_sweep.py --backend thread --workers 6
```
