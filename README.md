# Guided Parafoil Simulation

A Python-based simulation environment for modeling and testing the dynamics of guided parafoils. 

The project is structured as a Python package and uses [Poetry](https://python-poetry.org/) for dependency management and packaging.

## Features

- **6-DOF Dynamics:** Nonlinear six degrees-of-freedom parafoil simulation.
- **Aerodynamic Modeling:** Parameterized with 16 aerodynamic coefficients for lift, drag, and moments.
- **Control Surfaces:** Simulate flap deflections and other control inputs.
- **Wind Modeling:** Incorporates time-varying wind vectors.
- **Extensible Architecture:** Easy to modify for new control laws or environmental conditions.
- **Genetic Algorithm Integration:** Supports optimization of aerodynamic coefficients using DEAP-based GA.

## Installation

Make sure you have [Poetry](https://python-poetry.org/) installed. Then install the package in editable mode:

```bash
git clone <repo-url>
cd guided-parafoil-sim
poetry install
poetry shell
```

## reference

check out the [documentation](https://tylergwyrdd.github.io/Minos/)
