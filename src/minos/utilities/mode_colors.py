"""Shared mode-to-color helpers for trajectory plots."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt


def normalize_mode_series(mode_series: Sequence[str | None] | None, expected_len: int) -> list[str]:
    """Return a sanitized mode list with one entry per time sample."""
    if mode_series is None:
        return ["unknown"] * expected_len
    if len(mode_series) != expected_len:
        raise ValueError("mode_series must have the same length as the trajectory series.")
    normalized: list[str] = []
    for mode in mode_series:
        text = "" if mode is None else str(mode).strip()
        normalized.append(text or "unknown")
    return normalized


def mode_color_lookup(modes: Sequence[str], preferred_colors: dict[str, str] | None = None) -> dict[str, str]:
    """Build a deterministic color mapping for the provided mode labels."""
    unique_modes = list(dict.fromkeys(modes))
    color_by_mode: dict[str, str] = {} if preferred_colors is None else dict(preferred_colors)
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["tab:blue"])
    next_idx = 0
    for mode in unique_modes:
        if mode not in color_by_mode:
            color_by_mode[mode] = cycle[next_idx % len(cycle)]
            next_idx += 1
    return {mode: color_by_mode[mode] for mode in unique_modes}
