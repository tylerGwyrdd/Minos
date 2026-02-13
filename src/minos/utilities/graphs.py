"""Plotting helpers for typed simulation snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .snapshots import SimulationSnapshot


class PlotKey(str, Enum):
    """Supported plot channels."""

    POSITION = "position"
    VELOCITY = "velocity"
    ACCELERATION = "acceleration"
    EULER_ANGLES = "euler_angles"
    ANGULAR_VELOCITY = "angular_velocity"
    ANGULAR_ACCELERATION = "angular_acceleration"
    ANGLE_OF_ATTACK = "angle_of_attack"
    SIDESLIP_ANGLE = "sideslip_angle"
    FORCE_COEFFICIENTS = "force_coefficients"
    MOMENT_COEFFICIENTS = "moment_coefficients"
    FORCES_MAGNITUDE = "forces_magnitude"
    MOMENTS_MAGNITUDE = "moments_magnitude"
    FORCE_COMPONENTS = "force_components"
    MOMENT_COMPONENTS = "moment_components"
    AIRSPEED_VECTOR = "airspeed_vector"
    WIND_VECTOR = "wind_vector"
    DEFLECTION = "deflection"
    HEADINGS = "headings"
    EULER_RATES = "euler_rates"
    INERTIAL_POSITION_3D = "inertial_position_3d"


@dataclass
class PlotConfig:
    """Typed plot selection configuration."""

    enabled: set[PlotKey] = field(default_factory=set)

    def is_enabled(self, key: PlotKey) -> bool:
        """Return ``True`` when a plot is enabled."""
        return key in self.enabled


_LEGACY_KEY_MAP = {
    "Position": PlotKey.POSITION,
    "Velocity": PlotKey.VELOCITY,
    "Acceleration": PlotKey.ACCELERATION,
    "Euler Angles": PlotKey.EULER_ANGLES,
    "Angular Velocity": PlotKey.ANGULAR_VELOCITY,
    "Angular Acceleration": PlotKey.ANGULAR_ACCELERATION,
    "Angle of Attack": PlotKey.ANGLE_OF_ATTACK,
    "Sideslip Angle": PlotKey.SIDESLIP_ANGLE,
    "Force Coefficients": PlotKey.FORCE_COEFFICIENTS,
    "Moment Coefficients": PlotKey.MOMENT_COEFFICIENTS,
    "Forces": PlotKey.FORCES_MAGNITUDE,
    "Moments": PlotKey.MOMENTS_MAGNITUDE,
    "Forces_components": PlotKey.FORCE_COMPONENTS,
    "Moments_components": PlotKey.MOMENT_COMPONENTS,
    "Airspeed Vector": PlotKey.AIRSPEED_VECTOR,
    "Wind Vector": PlotKey.WIND_VECTOR,
    "Deflection": PlotKey.DEFLECTION,
    "headings": PlotKey.HEADINGS,
    "Euler Rates": PlotKey.EULER_RATES,
    "inertial Position": PlotKey.INERTIAL_POSITION_3D,
    "Inertial Position": PlotKey.INERTIAL_POSITION_3D,
}


def _ensure_snapshots(data: Sequence[SimulationSnapshot]) -> list[SimulationSnapshot]:
    if not data:
        return []
    if not isinstance(data[0], SimulationSnapshot):
        raise TypeError("plot_selected_parameters expects Sequence[SimulationSnapshot].")
    return list(data)


def _normalize_config(plots_to_show: PlotConfig | dict[str, bool] | Iterable[PlotKey]) -> PlotConfig:
    if isinstance(plots_to_show, PlotConfig):
        return plots_to_show
    if isinstance(plots_to_show, dict):
        enabled = {
            _LEGACY_KEY_MAP[k]
            for k, enabled_flag in plots_to_show.items()
            if enabled_flag and k in _LEGACY_KEY_MAP
        }
        return PlotConfig(enabled=enabled)
    return PlotConfig(enabled=set(plots_to_show))


def plot_state_over_time(
    data: np.ndarray,
    labels: list[str],
    title: str,
    ylabel: str,
    times: np.ndarray,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot one or more channels over time."""
    fig, ax = plt.subplots()
    if data.ndim == 1:
        data = data[:, np.newaxis]
    for idx in range(data.shape[1]):
        ax.plot(times, data[:, idx], label=labels[idx])
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig, ax


def plot_force_moment_components_shared(
    times: np.ndarray,
    vectors: list[tuple[str, np.ndarray]],
    title_prefix: str,
    ylabel: str,
) -> tuple[plt.Figure, plt.Axes]:
    """Overlay X/Y/Z components for multiple vector fields."""
    colors = ["tab:red", "tab:blue", "tab:green"]
    linestyles = ["-", "--", ":"]
    components = ["X", "Y", "Z"]
    fig, ax = plt.subplots()
    for vec_idx, (name, vec_series) in enumerate(vectors):
        for comp_idx, comp_label in enumerate(components):
            ax.plot(
                times,
                vec_series[:, comp_idx],
                label=f"{name} {comp_label}",
                linestyle=linestyles[comp_idx],
                color=colors[vec_idx % len(colors)],
            )
    ax.set_title(f"{title_prefix} Components")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig, ax


def plot_3d_position(inertial_positions: np.ndarray, title: str) -> tuple[plt.Figure, plt.Axes]:
    """Plot a 3D inertial trajectory and return figure + axis."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        inertial_positions[:, 0],
        inertial_positions[:, 1],
        inertial_positions[:, 2],
        label="Parafoil Path",
        color="tab:blue",
        linewidth=2,
    )
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Z Position (m)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_selected_parameters(
    data: Sequence[SimulationSnapshot],
    plots_to_show: PlotConfig | dict[str, bool] | Iterable[PlotKey],
) -> list[plt.Figure]:
    """Render selected plots from typed simulation snapshots."""
    snapshots = _ensure_snapshots(data)
    if not snapshots:
        return []
    cfg = _normalize_config(plots_to_show)
    times = np.array([s.time_s for s in snapshots], dtype=float)
    figures: list[plt.Figure] = []

    def add_plot(enabled_key: PlotKey, series: np.ndarray, labels: list[str], title: str, ylabel: str, deg: bool = False) -> None:
        if not cfg.is_enabled(enabled_key):
            return
        plot_data = np.degrees(series) if deg else series
        fig, _ = plot_state_over_time(plot_data, labels, title, ylabel, times)
        figures.append(fig)

    state_pos = np.array([s.state.position for s in snapshots])
    state_vel = np.array([s.state.velocity_body for s in snapshots])
    eulers = np.array([s.state.eulers for s in snapshots])
    omega = np.array([s.state.angular_velocity for s in snapshots])
    acc = np.array([s.acc for s in snapshots])
    ang_acc = np.array([s.angular_acc for s in snapshots])
    aoa = np.array([s.angle_of_attack for s in snapshots])
    beta = np.array([s.sideslip_angle for s in snapshots])
    euler_rates = np.array([s.euler_rates for s in snapshots])
    va = np.array([s.va for s in snapshots])
    wind = np.array([s.wind_inertial for s in snapshots])
    deflections = np.array([[s.flap_left, s.flap_right] for s in snapshots])
    coeff_force = np.array([[s.CL, s.CD] for s in snapshots])
    coeff_moment = np.array([[s.Cl, s.Cm, s.Cn] for s in snapshots])
    force_mags = np.array(
        [
            [np.linalg.norm(s.F_aero), np.linalg.norm(s.F_g), np.linalg.norm(s.F_fictious)]
            for s in snapshots
        ]
    )
    moment_mags = np.array(
        [
            [np.linalg.norm(s.M_aero), np.linalg.norm(s.M_f_aero), np.linalg.norm(s.M_fictious)]
            for s in snapshots
        ]
    )

    add_plot(PlotKey.POSITION, state_pos, ["X", "Y", "Z"], "Position vs Time", "Position (m)")
    add_plot(PlotKey.VELOCITY, state_vel, ["u", "v", "w"], "Velocity vs Time", "Velocity (m/s)")
    add_plot(PlotKey.ACCELERATION, acc, ["ax", "ay", "az"], "Acceleration vs Time", "Acceleration (m/s^2)")
    add_plot(PlotKey.EULER_ANGLES, eulers, ["Roll", "Pitch", "Yaw"], "Euler Angles vs Time", "Angle (deg)", deg=True)
    add_plot(PlotKey.ANGULAR_VELOCITY, omega, ["p", "q", "r"], "Angular Velocity vs Time", "Angular rate (deg/s)", deg=True)
    add_plot(PlotKey.ANGULAR_ACCELERATION, ang_acc, ["pdot", "qdot", "rdot"], "Angular Acceleration vs Time", "Angular rate (deg/s^2)", deg=True)
    add_plot(PlotKey.ANGLE_OF_ATTACK, aoa, ["AoA"], "Angle of Attack vs Time", "Angle (deg)", deg=True)
    add_plot(PlotKey.SIDESLIP_ANGLE, beta, ["Beta"], "Sideslip vs Time", "Angle (deg)", deg=True)
    add_plot(PlotKey.FORCE_COEFFICIENTS, coeff_force, ["CL", "CD"], "Force Coefficients vs Time", "Coefficient")
    add_plot(PlotKey.MOMENT_COEFFICIENTS, coeff_moment, ["Cl", "Cm", "Cn"], "Moment Coefficients vs Time", "Coefficient")
    add_plot(PlotKey.FORCES_MAGNITUDE, force_mags, ["F_aero", "F_g", "F_fictious"], "Force Magnitudes vs Time", "Force (N)")
    add_plot(PlotKey.MOMENTS_MAGNITUDE, moment_mags, ["M_aero", "M_f_aero", "M_fictious"], "Moment Magnitudes vs Time", "Moment (Nm)")
    add_plot(PlotKey.AIRSPEED_VECTOR, va, ["Vx", "Vy", "Vz"], "Airspeed Vector vs Time", "Airspeed (m/s)")
    add_plot(PlotKey.WIND_VECTOR, wind, ["Wx", "Wy", "Wz"], "Wind Vector vs Time", "Wind (m/s)")
    add_plot(PlotKey.DEFLECTION, deflections, ["Flap Left", "Flap Right"], "Flap Deflection vs Time", "Deflection (rad)")
    add_plot(PlotKey.EULER_RATES, euler_rates, ["Roll rate", "Pitch rate", "Yaw rate"], "Euler Rates vs Time", "Angular rate (deg/s)", deg=True)

    if cfg.is_enabled(PlotKey.HEADINGS):
        heading = np.array(
            [
                [
                    0.0 if s.heading_current is None else s.heading_current,
                    0.0 if s.heading_desired is None else s.heading_desired,
                ]
                for s in snapshots
            ]
        )
        fig, _ = plot_state_over_time(np.degrees(heading), ["Current", "Desired"], "Heading vs Time", "Heading (deg)", times)
        figures.append(fig)

    if cfg.is_enabled(PlotKey.FORCE_COMPONENTS):
        vectors = [
            ("F_aero", np.array([s.F_aero for s in snapshots])),
            ("F_g", np.array([s.F_g for s in snapshots])),
            ("F_fictious", np.array([s.F_fictious for s in snapshots])),
        ]
        fig, _ = plot_force_moment_components_shared(times, vectors, "Force", "Force (N)")
        figures.append(fig)

    if cfg.is_enabled(PlotKey.MOMENT_COMPONENTS):
        vectors = [
            ("M_aero", np.array([s.M_aero for s in snapshots])),
            ("M_f_aero", np.array([s.M_f_aero for s in snapshots])),
            ("M_fictious", np.array([s.M_fictious for s in snapshots])),
        ]
        fig, _ = plot_force_moment_components_shared(times, vectors, "Moment", "Moment (Nm)")
        figures.append(fig)

    if cfg.is_enabled(PlotKey.INERTIAL_POSITION_3D):
        inertial_positions = np.array([s.inertial_position for s in snapshots])
        fig, _ = plot_3d_position(inertial_positions, "Inertial Position")
        figures.append(fig)

    return figures
