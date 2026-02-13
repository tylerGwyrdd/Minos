"""RLS-based wind estimation strategy."""

from __future__ import annotations

import numpy as np

from minos.guidance.wind_estimation import WindRLS

from ..interfaces import MissionContext, NavigationEstimate, Navigator, Observation


class RlsWindEstimator(Navigator):
    """Estimate horizontal wind using recursive least squares."""

    def __init__(self, lambda_: float = 1.0e-4, delta: float = 1.0e13) -> None:
        self._rls = WindRLS(lambda_=lambda_, delta=delta)

    def update(self, observation: Observation, dt: float, mission: MissionContext) -> NavigationEstimate:
        del dt
        vx = float(observation.inertial_velocity[0])
        vy = float(observation.inertial_velocity[1])
        self._rls.update(vx, vy)
        wind_xy = self._rls.get_wind_estimate()
        wind_est = np.array([wind_xy[0], wind_xy[1], observation.wind_inertial[2]], dtype=float)

        wind_mag = float(np.linalg.norm(wind_est[:2]))
        if wind_mag < 0.3:
            wind_mag = 0.0
            wind_unit = np.array([1.0, 0.0], dtype=float)
            wind_heading = 0.0
        else:
            wind_unit = wind_est[:2] / wind_mag
            wind_heading = float(np.arctan2(wind_unit[1], wind_unit[0]))

        mission.extras["wind_magnitude"] = wind_mag
        mission.extras["wind_unit_vector"] = wind_unit
        mission.extras["wind_heading"] = wind_heading

        return NavigationEstimate(
            wind_inertial_estimate=wind_est,
            extras={
                "wind_magnitude": wind_mag,
                "wind_unit_vector": wind_unit.copy(),
                "wind_heading": wind_heading,
            },
        )
