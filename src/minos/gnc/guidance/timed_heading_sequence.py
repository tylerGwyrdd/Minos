"""Guidance strategy for scripted heading step maneuvers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..interfaces import GuidanceCommand, GuidanceLaw, MissionContext, NavigationEstimate, Observation


def _wrap_angle_0_2pi(angle: float) -> float:
    return float((angle + 2.0 * np.pi) % (2.0 * np.pi))


@dataclass(frozen=True)
class HeadingStep:
    """One heading target that activates at/after ``time_s``."""

    time_s: float
    heading_rad: float


@dataclass
class TimedHeadingSequenceConfig:
    """Configuration for the timed heading sequence guidance strategy."""

    sequence: Sequence[HeadingStep] = field(
        default_factory=lambda: (
            HeadingStep(0.0, 0.0),
            HeadingStep(10.0, np.deg2rad(90.0)),
            HeadingStep(25.0, 0.0),
        )
    )


class TimedHeadingSequenceGuidance(GuidanceLaw):
    """Emit desired heading according to a time-indexed step sequence."""

    def __init__(self, config: TimedHeadingSequenceConfig | None = None) -> None:
        self.config = TimedHeadingSequenceConfig() if config is None else config
        if len(self.config.sequence) == 0:
            raise ValueError("TimedHeadingSequenceGuidance requires at least one HeadingStep.")
        self._sequence = sorted(self.config.sequence, key=lambda s: float(s.time_s))

    def update(
        self,
        observation: Observation,
        nav: NavigationEstimate,
        mission: MissionContext,
        dt: float,
    ) -> GuidanceCommand:
        del nav, dt
        t = float(observation.time_s)
        desired = self._sequence[0].heading_rad
        active_index = 0
        for i, step in enumerate(self._sequence):
            if t >= float(step.time_s):
                desired = step.heading_rad
                active_index = i
            else:
                break
        desired = _wrap_angle_0_2pi(float(desired))
        mission.phase = "heading-sequence"
        mission.extras["heading_sequence_index"] = active_index
        return GuidanceCommand(
            desired_heading=desired,
            flare_magnitude=0.0,
            extras={
                "mode": "heading-sequence",
                "active_step_index": active_index,
            },
        )
