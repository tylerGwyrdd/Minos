"""T-approach guidance strategy implementation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..interfaces import (
    GuidanceCommand,
    GuidanceLaw,
    GuidanceMarker,
    GuidanceVisualization,
    MissionContext,
    NavigationEstimate,
    Observation,
)


def _wrap_angle(angle: float) -> float:
    """Wrap angle to [0, 2*pi)."""
    return float((angle + 2.0 * np.pi) % (2.0 * np.pi))


def _smooth_heading_to_line_with_wind(
    position: np.ndarray,
    line_point: np.ndarray,
    line_direction: np.ndarray,
    lookahead_distance: float,
    wind_vector: np.ndarray,
    airspeed: float,
) -> float:
    """Compute required heading to follow a line while compensating for wind."""
    p = np.asarray(position, dtype=float)
    a = np.asarray(line_point, dtype=float)
    d = np.asarray(line_direction, dtype=float)
    d_norm = np.linalg.norm(d)
    if d_norm < 1.0e-9:
        return _wrap_angle(float(np.arctan2(line_point[1] - position[1], line_point[0] - position[0])))
    d = d / d_norm

    ap = p - a
    t = float(np.dot(ap, d))
    closest_point = a + t * d
    lookahead_point = closest_point + d * lookahead_distance
    desired_track = lookahead_point - p
    track_norm = np.linalg.norm(desired_track)
    if track_norm < 1.0e-9:
        return _wrap_angle(float(np.arctan2(d[1], d[0])))
    desired_track = desired_track / track_norm
    air_vector = desired_track * float(airspeed) - np.asarray(wind_vector, dtype=float)
    return _wrap_angle(float(np.arctan2(air_vector[1], air_vector[0])))


def _compute_required_heading(wind_vector: np.ndarray, airspeed: float, target_vector: np.ndarray) -> tuple[float, np.ndarray]:
    """Solve wind triangle and return heading + air-relative velocity."""
    w = np.asarray(wind_vector, dtype=float)
    d = np.asarray(target_vector, dtype=float)
    d_norm = np.linalg.norm(d)
    if d_norm < 1.0e-9:
        return 0.0, np.zeros(2, dtype=float)
    d_hat = d / d_norm

    a = 1.0
    b = -2.0 * float(np.dot(d_hat, w))
    c = float(np.dot(w, w) - airspeed**2)
    discriminant = b**2 - 4.0 * a * c
    if discriminant < 0.0:
        heading = float(np.arctan2(d_hat[1], d_hat[0]))
        return _wrap_angle(heading), d_hat * airspeed

    vg = (-b + np.sqrt(discriminant)) / (2.0 * a)
    v_g = vg * d_hat
    v_a = v_g - w
    heading = float(np.arctan2(v_a[1], v_a[0]))
    return _wrap_angle(heading), v_a


@dataclass
class TApproachConfig:
    """Configuration defaults for T-approach guidance."""

    final_approach_height: float = 100.0
    spirialing_radius: float = 20.0
    horizontal_velocity: float = 5.9
    sink_velocity: float = 4.9
    IPI: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    flare_height: float = 20.0
    desired_heading: float = 0.0
    mode: str = "initialising"


class TApproachGuidance(GuidanceLaw):
    """Guidance law that produces desired heading using the T-approach logic."""

    def __init__(self, config: TApproachConfig | None = None) -> None:
        self.config = TApproachConfig() if config is None else config

    def update(
        self,
        observation: Observation,
        nav: NavigationEstimate,
        mission: MissionContext,
        dt: float,
    ) -> GuidanceCommand:
        params = self._params_from_mission(mission=mission, observation=observation, dt=dt)
        wind_est = np.asarray(nav.wind_inertial_estimate, dtype=float)

        if "wind_magnitude" in nav.extras:
            params["wind_magnitude"] = float(nav.extras["wind_magnitude"])
        if "wind_unit_vector" in nav.extras:
            params["wind_unit_vector"] = np.asarray(nav.extras["wind_unit_vector"], dtype=float)
        if "wind_heading" in nav.extras:
            params["wind_heading"] = float(nav.extras["wind_heading"])

        params["wind_magnitude"] = float(np.linalg.norm(wind_est[:2])) if "wind_magnitude" not in nav.extras else params["wind_magnitude"]
        if params["wind_magnitude"] < 0.3:
            params["wind_magnitude"] = 0.0
            params["wind_unit_vector"] = np.array([1.0, 0.0], dtype=float)
            params["wind_heading"] = 0.0
        else:
            params["wind_unit_vector"] = wind_est[:2] / params["wind_magnitude"]
            params["wind_heading"] = float(np.arctan2(params["wind_unit_vector"][1], params["wind_unit_vector"][0]))

        desired_heading, flare_magnitude = self._guidance_update(params, observation)
        mission.extras.update(params)
        mission.phase = str(params.get("mode", mission.phase))
        ipi = np.asarray(params["IPI"], dtype=float).reshape(3)
        ftp = np.asarray(params["FTP_centre"], dtype=float).reshape(2)
        markers = [
            GuidanceMarker(marker_id="ipi", label="IPI", xyz=ipi.copy(), style_hint={"wind_anchor": True}),
            GuidanceMarker(
                marker_id="ftp",
                label="FTP",
                xyz=np.array([ftp[0], ftp[1], float(params["final_approach_height"])], dtype=float),
            ),
        ]

        return GuidanceCommand(
            desired_heading=float(desired_heading),
            flare_magnitude=float(flare_magnitude),
            visualization=GuidanceVisualization(markers=markers, mode_styles=dict(self._MODE_STYLES)),
            extras={
                "mode": params.get("mode", self.config.mode),
            },
        )

    def _guidance_update(self, params: dict, observation: Observation) -> tuple[float, float]:
        flare_magnitude = 0.0
        position = observation.inertial_position[:2]
        current_velocity = observation.inertial_velocity[:2]
        current_height = float(observation.inertial_position[2])
        current_heading = _wrap_angle(float(observation.state.eulers[2]))

        if params["mode"] != "Final Approach":
            vertical_time_to_ipi = (current_height - params["IPI"][2]) / params["sink_velocity"]
            vector_to_ipi = params["IPI"][:2] - position
            ipi_norm = np.linalg.norm(vector_to_ipi)
            direction_to_ipi = vector_to_ipi / ipi_norm if ipi_norm > 1.0e-9 else np.array([1.0, 0.0], dtype=float)
            effective_vel = params["horizontal_velocity"] * direction_to_ipi + params["wind_magnitude"] * params["wind_unit_vector"]
            groundspeed_along_path = float(np.dot(effective_vel, direction_to_ipi))
            if groundspeed_along_path <= 0.0:
                horizontal_time_to_ipi = vertical_time_to_ipi + 1.0
            else:
                horizontal_time_to_ipi = ipi_norm / groundspeed_along_path

            if vertical_time_to_ipi < horizontal_time_to_ipi:
                params["desired_heading"] = _smooth_heading_to_line_with_wind(
                    position=position,
                    line_point=params["IPI"][:2],
                    line_direction=vector_to_ipi if ipi_norm > 1.0e-9 else direction_to_ipi,
                    lookahead_distance=10.0,
                    wind_vector=params["wind_unit_vector"] * params["wind_magnitude"],
                    airspeed=params["horizontal_velocity"],
                )
                return params["desired_heading"], flare_magnitude

        time_to_ftp = (current_height - params["final_approach_height"]) / params["sink_velocity"]
        spiral_centre_current = params["FTP_centre"] - time_to_ftp * params["wind_unit_vector"] * params["wind_magnitude"]
        if time_to_ftp < 0.0:
            params["mode"] = "Final Approach"

        if params["mode"] == "Final Approach":
            params["desired_heading"] = _smooth_heading_to_line_with_wind(
                position=position,
                line_point=params["FTP_centre"],
                line_direction=-params["wind_unit_vector"],
                lookahead_distance=10.0,
                wind_vector=params["wind_unit_vector"] * params["wind_magnitude"],
                airspeed=params["horizontal_velocity"],
            )
            if current_height < params["flare_height"]:
                flare_magnitude = 1.0
            return _wrap_angle(params["desired_heading"]), flare_magnitude

        if params["mode"] == "initialising":
            if not params["initialised"]:
                params["start_heading"] = current_heading
                params["initialised"] = True
            if observation.state.eulers[2] - params["start_heading"] > np.deg2rad(360.0):
                params["mode"] = "homing"
            else:
                angular_vel = params["horizontal_velocity"] / params["spirialing_radius"]
                params["desired_heading"] += angular_vel * params["update_rate"]

        if params["mode"] == "homing":
            vector_to_centre = spiral_centre_current - position
            perp_vector = np.array([vector_to_centre[1], -vector_to_centre[0]], dtype=float)
            distance_to_tangent = float(np.linalg.norm(vector_to_centre))
            perp_norm = np.linalg.norm(perp_vector)
            if perp_norm < 1.0e-9:
                perp_vector = np.array([1.0, 0.0], dtype=float)
                perp_norm = 1.0
            vector_to_tangent = (
                spiral_centre_current
                + perp_vector / perp_norm * params["spirialing_radius"] * 0.9
                - position
            )
            params["desired_heading"], _ = _compute_required_heading(
                params["wind_unit_vector"] * params["wind_magnitude"],
                params["horizontal_velocity"],
                vector_to_tangent,
            )
            if distance_to_tangent < 1.2 * params["spirialing_radius"]:
                params["mode"] = "energy_management"

        if params["mode"] == "energy_management":
            dist_to_ctp = float(np.linalg.norm(position - spiral_centre_current))
            if dist_to_ctp > 5.0 * params["spirialing_radius"]:
                params["mode"] = "homing"
            angular_vel = params["horizontal_velocity"] / params["spirialing_radius"]
            params["desired_heading"] += angular_vel * params["update_rate"]

        params["FTP_centre"] = params["IPI"][:2] + (
            (params["horizontal_velocity"] - params["wind_magnitude"])
            * (params["final_approach_height"] / params["sink_velocity"])
            * params["wind_unit_vector"]
        )
        params["desired_heading"] = _wrap_angle(params["desired_heading"])
        return params["desired_heading"], flare_magnitude

    def _params_from_mission(self, mission: MissionContext, observation: Observation, dt: float) -> dict:
        if mission.extras:
            params = mission.extras
        else:
            params = {}
            mission.extras = params

        params.setdefault("deployment_pos", observation.inertial_position.copy())
        params.setdefault("final_approach_height", float(self.config.final_approach_height))
        params.setdefault("spirialing_radius", float(self.config.spirialing_radius))
        params.setdefault("update_rate", float(dt))
        params.setdefault("wind_unit_vector", np.array([1.0, 0.0], dtype=float))
        params.setdefault("wind_magnitude", 0.0)
        params.setdefault("wind_heading", 0.0)
        params.setdefault("wind_v_list", [])
        params.setdefault("horizontal_velocity", float(self.config.horizontal_velocity))
        params.setdefault("sink_velocity", float(self.config.sink_velocity))
        params.setdefault("IPI", np.asarray(self.config.IPI, dtype=float))
        params.setdefault("flare_height", float(self.config.flare_height))
        params.setdefault("initialised", False)
        params.setdefault("mode", self.config.mode)
        params.setdefault("start_heading", 0.0)
        params.setdefault("desired_heading", float(self.config.desired_heading))
        params.setdefault("FTP_centre", np.array([0.0, 0.0], dtype=float))
        params["update_rate"] = float(dt)
        return params
    _MODE_STYLES: dict[str, dict[str, object]] = {
        "initialising": {"color": "tab:gray"},
        "homing": {"color": "tab:blue"},
        "energy_management": {"color": "tab:orange"},
        "Final Approach": {"color": "tab:green"},
    }
