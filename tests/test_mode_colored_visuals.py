"""Tests for mode-colored trajectory visualizations."""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from minos.utilities.graphs import plot_3d_position
from minos.utilities.visual_3D import visualize_parafoil_pose


def test_plot_3d_position_colors_segments_by_mode() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    fig, ax = plot_3d_position(positions, "Inertial Position", mode_series=["homing", "homing", "flare", "flare"])
    assert len(ax.lines) == 3
    assert ax.lines[0].get_color() == ax.lines[1].get_color()
    assert ax.lines[0].get_color() != ax.lines[2].get_color()
    labels = [line.get_label() for line in ax.lines]
    assert "homing" in labels
    assert "flare" in labels
    plt.close(fig)


def test_visualize_parafoil_pose_validates_mode_series_length() -> None:
    eulers = np.zeros((3, 3), dtype=float)
    positions = np.zeros((3, 3), dtype=float)
    try:
        visualize_parafoil_pose(
            euler_series=eulers,
            position_series=positions,
            show=False,
            mode_series=["homing", "flare"],
        )
        assert False, "Expected ValueError for mismatched mode_series length."
    except ValueError as exc:
        assert "same length" in str(exc)
