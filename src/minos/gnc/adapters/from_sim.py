"""Build GnC observations from the simulator state."""

from __future__ import annotations

from minos.physics.model import ParafoilModel6DOF
from minos.physics.types import State

from ..interfaces import Observation


def observation_from_sim(sim: ParafoilModel6DOF, time_s: float) -> Observation:
    """Convert the current simulator state into a typed GnC observation."""
    if sim.last_diagnostics is None:
        sim.evaluate()
    return Observation(
        time_s=float(time_s),
        state=State.from_sequence(sim.state.as_sequence()),
        inertial_position=sim.inertial_position.copy(),
        inertial_velocity=sim.inertial_velocity.copy(),
        wind_inertial=sim.inputs.wind_inertial.copy(),
        euler_rates=sim.euler_rates.copy(),
    )
