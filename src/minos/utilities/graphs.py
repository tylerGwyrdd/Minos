"""Plotting helpers for typed simulation snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .mode_colors import mode_color_lookup, normalize_mode_series
from .snapshots import SimulationSnapshot


class PlotKey(str, Enum):
    """Enumerate supported plot channels for snapshot visualization.

    ### Notes
    - Keys are stable string values used by both typed and legacy plot selection paths.
    - `INERTIAL_POSITION_3D` and `INERTIAL_POSITION_XY` can optionally include
      guidance-owned annotations when enabled in `PlotConfig`.
    """

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
    INERTIAL_POSITION_XY = "inertial_position_xy"


@dataclass
class PlotConfig:
    """Configure which plots are rendered and optional overlays.

    ### Parameters
    - `enabled` (`set[PlotKey]`): Channels to render.
    - `include_guidance_annotations` (`bool`): If `True`, render typed guidance
      markers/style hints when present in snapshots.

    ### Returns
    - `PlotConfig`: Plot configuration container.

    ### Raises
    - None directly.

    ### Notes
    - Annotation rendering is opt-in to keep default plots lightweight.
    """

    enabled: set[PlotKey] = field(default_factory=set)
    include_guidance_annotations: bool = False

    def is_enabled(self, key: PlotKey) -> bool:
        """Check whether a specific plot channel is enabled.

        ### Parameters
        - `key` (`PlotKey`): Channel to query.

        ### Returns
        - `bool`: `True` if enabled; otherwise `False`.

        ### Raises
        - None.
        """
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
    "Inertial Position XY": PlotKey.INERTIAL_POSITION_XY,
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
        include_annotations = bool(plots_to_show.get("include_guidance_annotations", False))
        enabled = {
            _LEGACY_KEY_MAP[k]
            for k, enabled_flag in plots_to_show.items()
            if enabled_flag and k in _LEGACY_KEY_MAP
        }
        return PlotConfig(enabled=enabled, include_guidance_annotations=include_annotations)
    return PlotConfig(enabled=set(plots_to_show))


def plot_state_over_time(
    data: np.ndarray,
    labels: list[str],
    title: str,
    ylabel: str,
    times: np.ndarray,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot one or more scalar/vector channels against time.

    ### Parameters
    - `data` (`np.ndarray`): Time-aligned series with shape `(N,)` or `(N, C)`.
    - `labels` (`list[str]`): Channel labels matching `C`.
    - `title` (`str`): Figure title.
    - `ylabel` (`str`): Y-axis label.
    - `times` (`np.ndarray`): Time vector in seconds, shape `(N,)`.

    ### Returns
    - `tuple[plt.Figure, plt.Axes]`: Created figure and axis.

    ### Raises
    - `IndexError`: If `labels` length is inconsistent with channel count.

    ### Notes
    - One-dimensional input is promoted to `(N, 1)` for a uniform plotting loop.
    """
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
    """Overlay X/Y/Z components for multiple vector fields on one axis.

    ### Parameters
    - `times` (`np.ndarray`): Time vector in seconds.
    - `vectors` (`list[tuple[str, np.ndarray]]`): Named vector series, each shape `(N, 3)`.
    - `title_prefix` (`str`): Prefix used to build chart title.
    - `ylabel` (`str`): Y-axis units label.

    ### Returns
    - `tuple[plt.Figure, plt.Axes]`: Created figure and axis.

    ### Raises
    - `IndexError`: If any vector does not have 3 components.

    ### Notes
    - Color encodes vector source; line style encodes axis component.
    """
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


def plot_3d_position(
    inertial_positions: np.ndarray,
    title: str,
    *,
    mode_series: Sequence[str | None] | None = None,
    mode_style_map: dict[str, str] | None = None,
    guidance_markers: Sequence[tuple[str, np.ndarray, dict[str, object]]] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot inertial 3D trajectory with optional mode and marker overlays.

    ### Parameters
    - `inertial_positions` (`np.ndarray`): Position series in inertial frame, shape `(N, 3)`.
    - `title` (`str`): Plot title.
    - `mode_series` (`Sequence[str | None] | None`): Optional phase/mode labels per sample.
    - `mode_style_map` (`dict[str, str] | None`): Optional explicit mode-to-color map.
    - `guidance_markers` (`Sequence[tuple[str, np.ndarray, dict[str, object]]] | None`):
      Optional marker tuples `(label, xyz, style)`.

    ### Returns
    - `tuple[plt.Figure, plt.Axes]`: Created figure and axis.

    ### Raises
    - `ValueError`: If `mode_series` length does not match trajectory length.

    ### Notes
    - Mode colors are deterministic from first appearance order.
    - Marker styling is renderer-driven and independent of guidance internals.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    if mode_series is None:
        ax.plot(
            inertial_positions[:, 0],
            inertial_positions[:, 1],
            inertial_positions[:, 2],
            label="Parafoil Path",
            color="tab:blue",
            linewidth=2,
        )
    else:
        modes = normalize_mode_series(mode_series, inertial_positions.shape[0])
        color_by_mode = mode_color_lookup(modes, preferred_colors=mode_style_map)
        seen: set[str] = set()
        for idx in range(inertial_positions.shape[0] - 1):
            mode = modes[idx]
            segment = inertial_positions[idx : idx + 2]
            label = mode if mode not in seen else "_nolegend_"
            ax.plot(
                segment[:, 0],
                segment[:, 1],
                segment[:, 2],
                color=color_by_mode[mode],
                linewidth=2,
                label=label,
            )
            seen.add(mode)
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Z Position (m)")
    ax.set_title(title)
    if guidance_markers:
        for label, point, style in guidance_markers:
            xyz = np.asarray(point, dtype=float).reshape(3)
            color = str(style.get("color", "black"))
            marker = str(style.get("marker", "x"))
            size = float(style.get("size", 70.0))
            linewidth = float(style.get("linewidth", 2.0))
            ax.scatter(xyz[0], xyz[1], xyz[2], marker=marker, s=size, linewidths=linewidth, color=color, label=label)
    ax.legend(title="Mode" if mode_series is not None else None)
    fig.tight_layout()
    return fig, ax


def plot_xy_position(
    inertial_positions: np.ndarray,
    title: str,
    *,
    mode_series: Sequence[str | None] | None = None,
    mode_style_map: dict[str, str] | None = None,
    guidance_markers: Sequence[tuple[str, np.ndarray, dict[str, object]]] | None = None,
    wind_anchor_xy: np.ndarray | None = None,
    wind_vector_xy: np.ndarray | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot inertial XY trajectory with optional guidance and wind overlays.

    ### Parameters
    - `inertial_positions` (`np.ndarray`): Position series in inertial frame, shape `(N, 3)`.
    - `title` (`str`): Plot title.
    - `mode_series` (`Sequence[str | None] | None`): Optional phase/mode labels per sample.
    - `mode_style_map` (`dict[str, str] | None`): Optional explicit mode-to-color map.
    - `guidance_markers` (`Sequence[tuple[str, np.ndarray, dict[str, object]]] | None`):
      Optional marker tuples `(label, xyz, style)`.
    - `wind_anchor_xy` (`np.ndarray | None`): Optional XY origin for wind arrow.
    - `wind_vector_xy` (`np.ndarray | None`): Optional inertial wind vector in XY plane.

    ### Returns
    - `tuple[plt.Figure, plt.Axes]`: Created figure and axis.

    ### Raises
    - `ValueError`: If mode series length is inconsistent with trajectory length.

    ### Notes
    - Wind arrow encodes direction only; length is normalized for readability.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    xy = np.asarray(inertial_positions[:, :2], dtype=float)
    if mode_series is None:
        ax.plot(xy[:, 0], xy[:, 1], label="Parafoil Path", color="tab:blue", linewidth=2)
    else:
        modes = normalize_mode_series(mode_series, xy.shape[0])
        color_by_mode = mode_color_lookup(modes, preferred_colors=mode_style_map)
        seen: set[str] = set()
        for idx in range(xy.shape[0] - 1):
            mode = modes[idx]
            segment = xy[idx : idx + 2]
            label = mode if mode not in seen else "_nolegend_"
            ax.plot(segment[:, 0], segment[:, 1], color=color_by_mode[mode], linewidth=2, label=label)
            seen.add(mode)

    if guidance_markers:
        for label, point, style in guidance_markers:
            xyz = np.asarray(point, dtype=float).reshape(3)
            color = str(style.get("color", "black"))
            marker = str(style.get("marker", "x"))
            size = float(style.get("size", 70.0))
            linewidth = float(style.get("linewidth", 2.0))
            ax.scatter(xyz[0], xyz[1], marker=marker, s=size, linewidths=linewidth, color=color, label=label)

    if wind_anchor_xy is not None and wind_vector_xy is not None:
        anchor = np.asarray(wind_anchor_xy, dtype=float).reshape(2)
        wind_xy = np.asarray(wind_vector_xy, dtype=float).reshape(2)
        wind_norm = float(np.linalg.norm(wind_xy))
        if wind_norm > 1.0e-9:
            span = float(np.linalg.norm(np.max(xy, axis=0) - np.min(xy, axis=0)))
            # Scale arrow with trajectory extent so wind remains visible in
            # both short and long trajectories.
            arrow_len = max(10.0, 0.15 * span)
            direction = wind_xy / wind_norm
            arrow = direction * arrow_len
            ax.quiver(
                anchor[0],
                anchor[1],
                arrow[0],
                arrow[1],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color="tab:orange",
                width=0.004,
                label="Wind Direction",
            )

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title(title)
    ax.grid(True)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(title="Mode" if mode_series is not None else None)
    fig.tight_layout()
    return fig, ax


def _extract_guidance_annotations(
    snapshots: Sequence[SimulationSnapshot],
) -> tuple[dict[str, str], list[tuple[str, np.ndarray, dict[str, object]]], np.ndarray | None]:
    visualization = None
    for sample in reversed(snapshots):
        if sample.guidance_visualization is not None:
            visualization = sample.guidance_visualization
            break
    if visualization is None:
        return {}, [], None

    mode_style_map: dict[str, str] = {}
    for mode, style in visualization.mode_styles.items():
        color = style.get("color")
        if color is not None:
            mode_style_map[str(mode)] = str(color)

    markers: list[tuple[str, np.ndarray, dict[str, object]]] = []
    wind_anchor_xyz: np.ndarray | None = None
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["tab:blue"])
    marker_cycle = ["x", "^", "s", "D", "P", "v", "*", ">", "<", "h", "8"]
    auto_style_by_id: dict[str, tuple[str, str]] = {}
    next_style_idx = 0
    for marker in visualization.markers:
        style = dict(marker.style_hint)
        marker_key = str(marker.marker_id)
        if marker_key not in auto_style_by_id:
            auto_style_by_id[marker_key] = (
                color_cycle[next_style_idx % len(color_cycle)],
                marker_cycle[next_style_idx % len(marker_cycle)],
            )
            next_style_idx += 1
        auto_color, auto_marker = auto_style_by_id[marker_key]
        if marker.kind == "active":
            style.setdefault("marker", "o")
            style.setdefault("size", 55.0)
        else:
            style.setdefault("marker", auto_marker)
            style.setdefault("size", 70.0)
        style.setdefault("color", auto_color)
        style.setdefault("linewidth", 2.0)
        xyz = np.asarray(marker.xyz, dtype=float).copy()
        if bool(style.get("wind_anchor", False)):
            wind_anchor_xyz = xyz
        markers.append((marker.label, xyz, style))
    return mode_style_map, markers, wind_anchor_xyz


def plot_selected_parameters(
    data: Sequence[SimulationSnapshot],
    plots_to_show: PlotConfig | dict[str, bool] | Iterable[PlotKey],
) -> list[plt.Figure]:
    """Render selected plot channels from typed simulation snapshots.

    ### Parameters
    - `data` (`Sequence[SimulationSnapshot]`): Snapshot sequence from simulation runs.
    - `plots_to_show` (`PlotConfig | dict[str, bool] | Iterable[PlotKey]`):
      Plot selection in typed or legacy form.

    ### Returns
    - `list[plt.Figure]`: Created matplotlib figures in render order.

    ### Raises
    - `TypeError`: If input data is non-empty and not `SimulationSnapshot`.
    - `ValueError`: If mode series lengths are inconsistent in mode-aware plots.

    ### Notes
    - Uses inertial-frame quantities for trajectory plots and wind vectors.
    - `include_guidance_annotations=True` is required to draw guidance markers.
    - For XY wind overlays, estimated wind is preferred when available, with
      fallback to truth wind from snapshots.

    ### Example
    ```python
    cfg = PlotConfig(
        enabled={PlotKey.HEADINGS, PlotKey.INERTIAL_POSITION_XY},
        include_guidance_annotations=True,
    )
    figs = plot_selected_parameters(snapshots, cfg)
    ```
    """
    snapshots = _ensure_snapshots(data)
    if not snapshots:
        return []
    cfg = _normalize_config(plots_to_show)
    times = np.array([s.time_s for s in snapshots], dtype=float)
    figures: list[plt.Figure] = []

    def add_plot(enabled_key: PlotKey, series: np.ndarray, labels: list[str], title: str, ylabel: str, deg: bool = False) -> None:
        if not cfg.is_enabled(enabled_key):
            return
        # Angle-like channels are converted here so upstream extraction can
        # stay in native SI/radian units.
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
        phases = [s.phase for s in snapshots]
        has_mode_data = any(phase is not None and str(phase).strip() for phase in phases)
        mode_style_map: dict[str, str] | None = None
        guidance_markers: list[tuple[str, np.ndarray, dict[str, object]]] | None = None
        wind_anchor_xyz: np.ndarray | None = None
        if cfg.include_guidance_annotations:
            mode_style_map, guidance_markers, wind_anchor_xyz = _extract_guidance_annotations(snapshots)
        # Annotation rendering is intentionally independent of guidance key names;
        # the plotter only consumes typed marker payloads.
        fig, _ = plot_3d_position(
            inertial_positions,
            "Inertial Position",
            mode_series=phases if has_mode_data else None,
            mode_style_map=mode_style_map if mode_style_map else None,
            guidance_markers=guidance_markers if guidance_markers else None,
        )
        figures.append(fig)

    if cfg.is_enabled(PlotKey.INERTIAL_POSITION_XY):
        inertial_positions = np.array([s.inertial_position for s in snapshots])
        phases = [s.phase for s in snapshots]
        has_mode_data = any(phase is not None and str(phase).strip() for phase in phases)
        mode_style_map: dict[str, str] | None = None
        guidance_markers: list[tuple[str, np.ndarray, dict[str, object]]] | None = None
        wind_anchor_xyz: np.ndarray | None = None
        if cfg.include_guidance_annotations:
            mode_style_map, guidance_markers, wind_anchor_xyz = _extract_guidance_annotations(snapshots)
        wind_xy = None
        if snapshots[-1].wind_estimate is not None:
            # Prefer estimator output when available so overlay matches guidance inputs.
            wind_xy = np.asarray(snapshots[-1].wind_estimate[:2], dtype=float)
        else:
            wind_xy = np.asarray(snapshots[-1].wind_inertial[:2], dtype=float)
        wind_anchor_xy = None if wind_anchor_xyz is None else np.asarray(wind_anchor_xyz[:2], dtype=float)
        fig, _ = plot_xy_position(
            inertial_positions,
            "Inertial Position XY",
            mode_series=phases if has_mode_data else None,
            mode_style_map=mode_style_map if mode_style_map else None,
            guidance_markers=guidance_markers if guidance_markers else None,
            wind_anchor_xy=wind_anchor_xy,
            wind_vector_xy=wind_xy,
        )
        figures.append(fig)

    return figures
