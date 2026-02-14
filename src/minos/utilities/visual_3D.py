"""Interactive 3D visualization for parafoil attitude and trajectory."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R

from .mode_colors import mode_color_lookup, normalize_mode_series

AngleUnit = Literal["rad", "deg"]
PositionFrame = Literal["NED", "ENU"]


def visualize_parafoil_pose(
    euler_series: np.ndarray | list[np.ndarray],
    position_series: np.ndarray | list[np.ndarray],
    interval: int = 100,
    slowmo_factor: float = 1.0,
    save_path: str | None = None,
    *,
    angle_unit: AngleUnit = "rad",
    frame: PositionFrame = "NED",
    show: bool = True,
    mode_series: Sequence[str | None] | None = None,
    wind_inertial_series: np.ndarray | list[np.ndarray] | None = None,
) -> tuple[plt.Figure, plt.Axes, animation.FuncAnimation | None]:
    """Animate parafoil attitude and trajectory in 3D.

    This utility visualizes body axes, canopy geometry, and trajectory history.
    It can optionally color trajectory by guidance mode and overlay wind in the
    **body frame** to aid interpretation of flight-relative disturbances.

    ### Parameters
    - `euler_series` (`np.ndarray | list[np.ndarray]`): Euler attitude series,
      shape `(N, 3)`.
    - `position_series` (`np.ndarray | list[np.ndarray]`): Position series,
      shape `(N, 3)`.
    - `interval` (`int`): Base animation interval in milliseconds.
    - `slowmo_factor` (`float`): Multiplier applied to `interval`.
    - `save_path` (`str | None`): Optional output GIF path.
    - `angle_unit` (`Literal["rad", "deg"]`): Units for `euler_series`.
    - `frame` (`Literal["NED", "ENU"]`): Display frame.
    - `show` (`bool`): If `True`, call `plt.show()`.
    - `mode_series` (`Sequence[str | None] | None`): Optional mode label per sample.
    - `wind_inertial_series` (`np.ndarray | list[np.ndarray] | None`):
      Optional wind in inertial **NED** frame, shape `(N, 3)`.

    ### Returns
    - `tuple[plt.Figure, plt.Axes, animation.FuncAnimation | None]`:
      Figure, axis, and animation object (if created).

    ### Raises
    - `ValueError`: If input shapes are inconsistent, or frame/unit options are invalid.

    ### Notes
    - `euler_series` and `position_series` are interpreted as NED-native data.
    - For `frame="ENU"`, both trajectory and attitude are transformed for display.
    - Wind overlay is computed as `wind_body = R_nb^T * wind_ned` per frame.
    - Animation timing assumes a fixed update interval across frames.

    ### Example
    ```python
    fig, ax, ani = visualize_parafoil_pose(
        euler_series=eulers,
        position_series=positions,
        mode_series=modes,
        wind_inertial_series=wind_series,
        show=False,
    )
    ```
    """
    eulers = np.asarray(euler_series, dtype=float)
    positions = np.asarray(position_series, dtype=float)
    if eulers.shape != positions.shape:
        raise ValueError("Euler and position series must share shape (N, 3).")
    if eulers.ndim != 2 or eulers.shape[1] != 3:
        raise ValueError("Input series must be shape (N, 3).")
    if angle_unit not in ("rad", "deg"):
        raise ValueError("angle_unit must be 'rad' or 'deg'.")
    frame_mode = frame.upper()
    if frame_mode not in ("NED", "ENU"):
        raise ValueError("frame must be 'NED' or 'ENU'.")

    eulers_deg = np.degrees(eulers) if angle_unit == "rad" else eulers.copy()
    num_frames = eulers_deg.shape[0]
    wind_ned: np.ndarray | None = None
    if wind_inertial_series is not None:
        wind_ned = np.asarray(wind_inertial_series, dtype=float)
        if wind_ned.shape != eulers.shape:
            raise ValueError("wind_inertial_series must have shape (N, 3) matching euler_series.")
    modes = normalize_mode_series(mode_series, num_frames)
    color_by_mode = mode_color_lookup(modes)
    has_mode_data = any(mode != "unknown" for mode in modes)
    ned_to_enu = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    if frame_mode == "ENU":
        # Convert NED position samples into ENU for visualization only.
        positions_view = positions @ ned_to_enu.T
    else:
        positions_view = positions.copy()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(bottom=0.3)
    if frame_mode == "NED":
        ax.set_xlabel("North")
        ax.set_ylabel("East")
        ax.set_zlabel("Down")
    else:
        ax.set_xlabel("East")
        ax.set_ylabel("North")
        ax.set_zlabel("Up")
    ax.set_title(f"Parafoil Pose ({frame_mode} Frame)")

    axes = np.eye(3)
    quivers: list = []
    labels: list = []
    parafoil_poly = None

    wing_span = 2.0
    chord_length = 0.6
    half_span = wing_span / 2.0
    half_chord = chord_length / 2.0
    parafoil_body = np.array(
        [
            [half_chord, -half_span, 0.0],
            [half_chord, half_span, 0.0],
            [-half_chord, half_span, 0.0],
            [-half_chord, -half_span, 0.0],
        ]
    )
    Rp = np.array([0.0, 0.0, -1.11])
    rigging_rotation = R.from_euler("y", -12.0, degrees=True)
    parafoil_body = rigging_rotation.apply(parafoil_body)
    Rp = rigging_rotation.apply(Rp)

    paused = [False]
    follow_mode = [True]
    current_frame = [0]

    pos_min = positions_view.min(axis=0)
    pos_max = positions_view.max(axis=0)
    center = (pos_max + pos_min) / 2.0
    span = (pos_max - pos_min) / 2.0
    padding = 1.0
    limits = np.array([center - span - padding, center + span + padding])

    slider_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    frame_slider = Slider(slider_ax, "Frame", 0, num_frames - 1, valinit=0, valfmt="%0.0f")
    button_ax = fig.add_axes([0.82, 0.11, 0.1, 0.04])
    play_button = Button(button_ax, "Pause")
    toggle_ax = fig.add_axes([0.02, 0.11, 0.15, 0.04])
    toggle_button = Button(toggle_ax, "Toggle View")

    def set_axes_equal() -> None:
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        x_mid = np.mean(x_limits)
        y_mid = np.mean(y_limits)
        z_mid = np.mean(z_limits)
        max_range = max(x_range, y_range, z_range) / 2.0
        ax.set_xlim3d([x_mid - max_range, x_mid + max_range])
        ax.set_ylim3d([y_mid - max_range, y_mid + max_range])
        ax.set_zlim3d([z_mid - max_range, z_mid + max_range])

    def configure_axis(title: str) -> None:
        """Apply consistent axis labels and title after axis clears."""
        if frame_mode == "NED":
            ax.set_xlabel("North")
            ax.set_ylabel("East")
            ax.set_zlabel("Down")
        else:
            ax.set_xlabel("East")
            ax.set_ylabel("North")
            ax.set_zlabel("Up")
        ax.set_title(title)

    def draw_frame(frame_idx: int) -> None:
        nonlocal quivers, labels, parafoil_poly
        current_frame[0] = frame_idx

        roll, pitch, yaw = eulers_deg[frame_idx]
        pos = positions_view[frame_idx]
        r_ned = R.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_matrix()
        if frame_mode == "ENU":
            # Rotate attitude basis into the displayed frame.
            r_view = ned_to_enu @ r_ned
        else:
            r_view = r_ned

        # Matplotlib/Tk can raise NotImplementedError for removing some 3D artists
        # (notably quiver artists), so we clear and redraw each frame instead.
        ax.cla()
        configure_axis(f"Parafoil Pose ({frame_mode} Frame)")

        rotated_axes = (r_view @ axes.T).T
        quivers = []
        labels = []
        for i, color in enumerate(["r", "g", "b"]):
            vec = rotated_axes[i]
            quivers.append(ax.quiver(*pos, *vec, color=color, length=1.0))
            labels.append(ax.text(*(pos + vec * 1.5), ["Body X", "Body Y", "Body Z"][i], color=color))

        parafoil_pos = pos + (r_view @ Rp)
        rotated_parafoil = (r_view @ parafoil_body.T).T + parafoil_pos
        parafoil_poly = Poly3DCollection([rotated_parafoil], alpha=0.4, color="gray")
        ax.add_collection3d(parafoil_poly)

        if wind_ned is not None:
            # Convert inertial wind to body-frame components, then render from body origin.
            wind_body = r_ned.T @ wind_ned[frame_idx]
            wind_view = r_view @ wind_body
            wind_mag = float(np.linalg.norm(wind_body))
            if wind_mag > 1.0e-9:
                wind_dir = wind_view / float(np.linalg.norm(wind_view))
                # Length is scaled for readability so the arrow remains visible
                # across a wide range of wind magnitudes.
                wind_len = float(np.clip(0.3 * wind_mag, 0.6, 2.0))
                wind_vec = wind_dir * wind_len
                ax.quiver(*pos, *wind_vec, color="tab:orange", linewidth=2.2)
                ax.text(*(pos + wind_vec * 1.1), "Wind (body)", color="tab:orange")

        if has_mode_data and frame_idx > 0:
            seen: set[str] = set()
            for idx in range(frame_idx):
                mode = modes[idx]
                segment = positions_view[idx : idx + 2]
                label = mode if mode not in seen else "_nolegend_"
                ax.plot3D(
                    segment[:, 0],
                    segment[:, 1],
                    segment[:, 2],
                    color=color_by_mode[mode],
                    linewidth=2,
                    label=label,
                )
                seen.add(mode)
            ax.legend(loc="upper right", title="Mode")
        else:
            trail = positions_view[: frame_idx + 1]
            ax.plot3D(trail[:, 0], trail[:, 1], trail[:, 2], color="black")

        if follow_mode[0]:
            zoom = 3.0
            ax.set_xlim([pos[0] - zoom, pos[0] + zoom])
            ax.set_ylim([pos[1] - zoom, pos[1] + zoom])
            ax.set_zlim([pos[2] - zoom, pos[2] + zoom])
        else:
            ax.set_xlim(limits[0][0], limits[1][0])
            ax.set_ylim(limits[0][1], limits[1][1])
            ax.set_zlim(limits[0][2], limits[1][2])
            set_axes_equal()
        fig.canvas.draw_idle()

    def slider_update(_val: float) -> None:
        if follow_mode[0]:
            paused[0] = True
            play_button.label.set_text("Play")
            draw_frame(int(frame_slider.val))

    def toggle_play(_event) -> None:
        if follow_mode[0]:
            paused[0] = not paused[0]
            play_button.label.set_text("Pause" if not paused[0] else "Play")

    def toggle_view(_event) -> None:
        follow_mode[0] = not follow_mode[0]
        paused[0] = False
        frame_slider.ax.set_visible(follow_mode[0])
        play_button.ax.set_visible(follow_mode[0])
        ax.clear()

        if follow_mode[0]:
            configure_axis(f"Parafoil Pose ({frame_mode} Frame)")
            draw_frame(current_frame[0])
        else:
            configure_axis(f"Full Trajectory ({frame_mode} Frame)")
            if has_mode_data and num_frames > 1:
                seen: set[str] = set()
                for idx in range(num_frames - 1):
                    mode = modes[idx]
                    segment = positions_view[idx : idx + 2]
                    label = mode if mode not in seen else "_nolegend_"
                    ax.plot3D(
                        segment[:, 0],
                        segment[:, 1],
                        segment[:, 2],
                        color=color_by_mode[mode],
                        linewidth=2,
                        label=label,
                    )
                    seen.add(mode)
                ax.legend(loc="upper right", title="Mode")
            else:
                ax.plot3D(positions_view[:, 0], positions_view[:, 1], positions_view[:, 2], color="black")
            ax.set_xlim(limits[0][0], limits[1][0])
            ax.set_ylim(limits[0][1], limits[1][1])
            ax.set_zlim(limits[0][2], limits[1][2])
            set_axes_equal()
            fig.canvas.draw_idle()

    frame_slider.on_changed(slider_update)
    play_button.on_clicked(toggle_play)
    toggle_button.on_clicked(toggle_view)

    actual_interval = max(1, int(interval * slowmo_factor))

    def update(frame_idx: int) -> None:
        if follow_mode[0]:
            if not paused[0]:
                draw_frame(current_frame[0])
                current_frame[0] = (current_frame[0] + 1) % num_frames
                frame_slider.eventson = False
                frame_slider.set_val(current_frame[0])
                frame_slider.eventson = True

    ani: animation.FuncAnimation | None = None
    if show or save_path:
        ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=actual_interval, blit=False, repeat=True)

    if save_path and ani is not None:
        from matplotlib.animation import PillowWriter

        writer = PillowWriter(fps=max(1, 1000 // actual_interval))
        ani.save(save_path, dpi=200, writer=writer)
    if not show and save_path is None:
        draw_frame(0)
    if show:
        plt.show()
    return fig, ax, ani
