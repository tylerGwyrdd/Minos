# Guided Parafoil Simulation

Python package for simulating and analyzing a guided parafoil with a 6-DoF physics model.

## Prerequisites

- Python `3.12+`
- [Poetry](https://python-poetry.org/)

## Setup

```bash
git clone <repo-url>
cd Level_4_project
poetry install
```

Recommended (no activation needed):

```bash
poetry run python examples/physics_model_3d_example.py
```

Optional (activate the Poetry environment manually):

PowerShell:

```powershell
Invoke-Expression (poetry env activate)
```

Bash/Zsh:

```bash
eval "$(poetry env activate)"
```

## Run The Physics + 3D Example

This runs the generated example script that uses the physics model, plotting utilities, and 3D visualizer:

```bash
poetry run python examples/physics_model_3d_example.py
```

Expected terminal output:

```text
Example completed successfully.
```

## Documentation

- Project docs: https://tylergwyrdd.github.io/Minos/
