"""Typed simulation snapshot model and row export helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from minos.physics.types import State

LegacySchema = Literal["run_sim", "main"]


@dataclass
class SimulationSnapshot:
    """One typed simulation log sample."""

    time_s: float
    state: State
    inertial_position: np.ndarray
    inertial_velocity: np.ndarray
    euler_rates: np.ndarray
    angle_of_attack: float
    sideslip_angle: float
    angular_acc: np.ndarray
    acc: np.ndarray
    CL: float
    CD: float
    Cl: float
    Cn: float
    Cm: float
    F_aero: np.ndarray
    F_g: np.ndarray
    F_fictious: np.ndarray
    M_aero: np.ndarray
    M_f_aero: np.ndarray
    M_fictious: np.ndarray
    M_total: np.ndarray
    va: np.ndarray
    wind_inertial: np.ndarray
    flap_left: float
    flap_right: float
    heading_current: float | None = None
    heading_desired: float | None = None

    def __post_init__(self) -> None:
        self.time_s = float(self.time_s)
        self.inertial_position = np.asarray(self.inertial_position, dtype=float).reshape(3)
        self.inertial_velocity = np.asarray(self.inertial_velocity, dtype=float).reshape(3)
        self.euler_rates = np.asarray(self.euler_rates, dtype=float).reshape(3)
        self.angular_acc = np.asarray(self.angular_acc, dtype=float).reshape(3)
        self.acc = np.asarray(self.acc, dtype=float).reshape(3)
        self.F_aero = np.asarray(self.F_aero, dtype=float).reshape(3)
        self.F_g = np.asarray(self.F_g, dtype=float).reshape(3)
        self.F_fictious = np.asarray(self.F_fictious, dtype=float).reshape(3)
        self.M_aero = np.asarray(self.M_aero, dtype=float).reshape(3)
        self.M_f_aero = np.asarray(self.M_f_aero, dtype=float).reshape(3)
        self.M_fictious = np.asarray(self.M_fictious, dtype=float).reshape(3)
        self.M_total = np.asarray(self.M_total, dtype=float).reshape(3)
        self.va = np.asarray(self.va, dtype=float).reshape(3)
        self.wind_inertial = np.asarray(self.wind_inertial, dtype=float).reshape(3)
        self.flap_left = float(self.flap_left)
        self.flap_right = float(self.flap_right)
        self.angle_of_attack = float(self.angle_of_attack)
        self.sideslip_angle = float(self.sideslip_angle)
        self.CL = float(self.CL)
        self.CD = float(self.CD)
        self.Cl = float(self.Cl)
        self.Cn = float(self.Cn)
        self.Cm = float(self.Cm)

    @property
    def flap_pair(self) -> np.ndarray:
        """Return flap deflections as ``[left, right]``."""
        return np.array([self.flap_left, self.flap_right], dtype=float)


def to_legacy_row(snapshot: SimulationSnapshot, schema: LegacySchema = "run_sim") -> list[object]:
    """Convert a typed snapshot into a historical list-row format."""
    state = snapshot.state.as_sequence()
    if schema == "main":
        return [
            snapshot.time_s,
            state,
            snapshot.angle_of_attack,
            snapshot.sideslip_angle,
            snapshot.angular_acc.copy(),
            snapshot.acc.copy(),
            snapshot.CL,
            snapshot.CD,
            snapshot.Cl,
            snapshot.Cn,
            snapshot.Cm,
            snapshot.F_aero.copy(),
            snapshot.F_g.copy(),
            snapshot.F_fictious.copy(),
            snapshot.M_aero.copy(),
            snapshot.M_f_aero.copy(),
            snapshot.M_fictious.copy(),
            snapshot.va.copy(),
            snapshot.wind_inertial.copy(),
            snapshot.flap_pair.copy(),
            [
                float(snapshot.heading_current or 0.0),
                float(snapshot.heading_desired or 0.0),
            ],
            snapshot.inertial_position.copy(),
            snapshot.euler_rates.copy(),
        ]

    return [
        snapshot.time_s,
        state,
        snapshot.angle_of_attack,
        snapshot.sideslip_angle,
        snapshot.angular_acc.copy(),
        snapshot.acc.copy(),
        snapshot.CL,
        snapshot.CD,
        snapshot.Cl,
        snapshot.Cn,
        snapshot.Cm,
        snapshot.F_aero.copy(),
        snapshot.F_g.copy(),
        snapshot.F_fictious.copy(),
        snapshot.M_aero.copy(),
        snapshot.M_f_aero.copy(),
        snapshot.M_fictious.copy(),
        snapshot.va.copy(),
        snapshot.wind_inertial.copy(),
        snapshot.flap_pair.copy(),
        snapshot.euler_rates.copy(),
        snapshot.inertial_position.copy(),
        snapshot.M_total.copy(),
        [snapshot.angle_of_attack, snapshot.sideslip_angle],
    ]

