"""Alternative T-approach guidance strategy from pseudocode flow.

This implementation follows the requested mode/state machine:
homing -> energy-management -> approach-FTP -> turn-into-wind -> flare,
with a backup branch for low-altitude contingencies.
"""

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
    disc = b**2 - 4.0 * a * c
    if disc < 0.0:
        heading = float(np.arctan2(d_hat[1], d_hat[0]))
        return _wrap_angle(heading), d_hat * airspeed

    vg = (-b + np.sqrt(disc)) / (2.0 * a)
    v_g = vg * d_hat
    v_a = v_g - w
    heading = float(np.arctan2(v_a[1], v_a[0]))
    return _wrap_angle(heading), v_a


@dataclass
class TApproach2Config:
    """Configuration for pseudocode-based T-approach strategy."""

    mode: str = "homing"
    IPI: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    final_approach_height: float = 100.0
    flare_height: float = 20.0
    backup_height: float = 60.0
    horizontal_velocity: float = 5.9
    sink_velocity: float = 4.9
    spirialing_radius: float = 20.0
    emc_offset: float = 40.0
    reach_radius: float = 12.0
    close_ipi_radius: float = 20.0
    low_height_threshold: float = 90.0
    backup_leg_distance: float = 50.0
    desired_heading: float = 0.0


class TApproachGuidance2(GuidanceLaw):
    """T-approach mode machine implementing the provided pseudocode."""

    def __init__(self, config: TApproach2Config | None = None) -> None:
        self.config = TApproach2Config() if config is None else config

    def update(
        self,
        observation: Observation,
        nav: NavigationEstimate,
        mission: MissionContext,
        dt: float,
    ) -> GuidanceCommand:
        params = self._params_from_mission(mission=mission, observation=observation, dt=dt)
        wind_est = np.asarray(nav.wind_inertial_estimate, dtype=float)

        wind_mag = float(np.linalg.norm(wind_est[:2]))
        if wind_mag < 0.3:
            wind_mag = 0.0
            wind_unit = np.array([1.0, 0.0], dtype=float)
            wind_heading = 0.0
        else:
            wind_unit = wind_est[:2] / wind_mag
            wind_heading = float(np.arctan2(wind_unit[1], wind_unit[0]))

        params["wind_magnitude"] = wind_mag
        params["wind_unit_vector"] = wind_unit
        params["wind_heading"] = wind_heading

        wp, desired_heading, flare = self._guidance_update(params, observation)
        mission.extras.update(params)
        mission.phase = str(params.get("mode", mission.phase))
        ipi = np.asarray(params["IPI"], dtype=float).reshape(3)
        ftp = np.asarray(params["FTP"], dtype=float).reshape(2)
        ftp_z = float(params["final_approach_height"])
        markers = [
            GuidanceMarker(marker_id="ipi", label="IPI", xyz=ipi.copy(), style_hint={"wind_anchor": True}),
            GuidanceMarker(marker_id="ipi_app", label="IPI app", xyz=np.array([params["IPI_app"][0], params["IPI_app"][1], ipi[2]], dtype=float)),
            GuidanceMarker(marker_id="ftp", label="FTP", xyz=np.array([ftp[0], ftp[1], ftp_z], dtype=float)),
            GuidanceMarker(marker_id="emc", label="EMC", xyz=np.array([params["EMC"][0], params["EMC"][1], ftp_z], dtype=float)),
            GuidanceMarker(marker_id="emtp1", label="EMTP1", xyz=np.array([params["EMTP1"][0], params["EMTP1"][1], ftp_z], dtype=float)),
            GuidanceMarker(marker_id="emtp2", label="EMTP2", xyz=np.array([params["EMTP2"][0], params["EMTP2"][1], ftp_z], dtype=float)),
            GuidanceMarker(
                marker_id="active_wp",
                label="Active WP",
                xyz=np.array([wp[0], wp[1], float(observation.inertial_position[2])], dtype=float),
                kind="active",
                style_hint={"color": "black"},
            ),
        ]

        return GuidanceCommand(
            desired_heading=float(desired_heading),
            flare_magnitude=float(flare),
            visualization=GuidanceVisualization(markers=markers, mode_styles=dict(self._MODE_STYLES)),
            extras={
                "mode": params.get("mode", self.config.mode),
                "heading_target": params.get("em_heading"),
            },
        )

    def _guidance_update(self, params: dict, observation: Observation) -> tuple[np.ndarray, float, float]:
        pos_xy = np.asarray(observation.inertial_position[:2], dtype=float)
        z = float(observation.inertial_position[2])
        ipi = np.asarray(params["IPI"], dtype=float).reshape(3)
        wind_u = np.asarray(params["wind_unit_vector"], dtype=float)
        wind_mag = float(params["wind_magnitude"])
        wind_xy = wind_u * wind_mag
        flare = 0.0

        # 1-3: drift + apparent IPI + wind orientation.
        time_to_ipi = max(0.0, (z - ipi[2]) / max(float(params["sink_velocity"]), 1.0e-6))
        xy_drift = wind_xy * time_to_ipi
        params["xy_drift"] = xy_drift
        params["IPI_app"] = ipi.copy()
        params["IPI_app"][:2] = ipi[:2] - xy_drift
        params["wind_orientation"] = float(np.arctan2(wind_u[1], wind_u[0]))

        # 5: waypoints.
        cross = np.array([-wind_u[1], wind_u[0]], dtype=float)
        ftp = self._compute_ftp(params, ipi)
        emc = ftp + float(params["emc_offset"]) * cross
        emtp1 = emc + float(params["spirialing_radius"]) * cross
        emtp2 = emc - float(params["spirialing_radius"]) * cross
        params["FTP"] = ftp
        params["EMC"] = emc
        params["EMTP1"] = emtp1
        params["EMTP2"] = emtp2

        wp = ftp.copy()
        mode = str(params["mode"])

        if mode in ("homing", "energy-management"):
            if mode == "homing":
                if z <= float(params["low_height_threshold"]):
                    params["mode"] = "backup"
                    mode = "backup"
                elif self._reached(pos_xy, emc, params["reach_radius"]) or z <= float(params["final_approach_height"]):
                    params["mode"] = "energy-management"
                    mode = "energy-management"
                else:
                    wp = emc

            if mode == "energy-management":
                remaining_glide = self._remaining_glide_distance(z, ipi[2], params["horizontal_velocity"], params["sink_velocity"])
                dist_to_ftp = float(np.linalg.norm(ftp - pos_xy))
                if remaining_glide <= dist_to_ftp:
                    params["mode"] = "approach-FTP"
                    mode = "approach-FTP"
                else:
                    em_heading = params.get("em_heading")
                    if em_heading is not None:
                        if em_heading == "EMTP1":
                            if self._reached(pos_xy, emtp1, params["reach_radius"]):
                                wp = emtp2
                                params["em_heading"] = "EMTP2"
                            else:
                                wp = emtp1
                        else:
                            if self._reached(pos_xy, emtp2, params["reach_radius"]):
                                wp = emtp1
                                params["em_heading"] = "EMTP1"
                            else:
                                wp = emtp2
                    else:
                        d1 = float(np.linalg.norm(emtp1 - pos_xy))
                        d2 = float(np.linalg.norm(emtp2 - pos_xy))
                        if d1 <= d2:
                            wp = emtp1
                            params["em_heading"] = "EMTP1"
                        else:
                            wp = emtp2
                            params["em_heading"] = "EMTP2"

        if mode == "approach-FTP":
            ftp = params["FTP"]
            if self._reached(pos_xy, ftp, params["reach_radius"]):
                params["mode"] = "turn-into-wind"
                mode = "turn-into-wind"
            else:
                wp = ftp

        if mode == "turn-into-wind":
            if z < float(params["flare_height"]):
                params["mode"] = "flare"
                mode = "flare"
            else:
                ipi_app_xy = np.asarray(params["IPI_app"][:2], dtype=float)
                if self._reached(pos_xy, ipi_app_xy, params["close_ipi_radius"]):
                    wp = ipi[:2]
                else:
                    wp = ipi_app_xy

        if mode == "backup":
            if (z < float(params["backup_height"])) and (z > float(params["flare_height"])):
                # Turn against wind and prepare for landing.
                wp = pos_xy - wind_u * float(params["backup_leg_distance"])
            elif z <= float(params["flare_height"]):
                params["mode"] = "flare"
                mode = "flare"
            else:
                ftp = params["FTP"]
                wp = ftp

        if mode == "flare":
            wp = ipi[:2]
            flare = 1.0

        target_vector = wp - pos_xy
        desired_heading, _ = _compute_required_heading(wind_xy, float(params["horizontal_velocity"]), target_vector)
        params["desired_heading"] = float(desired_heading)
        return wp, float(desired_heading), float(flare)

    @staticmethod
    def _reached(position_xy: np.ndarray, wp_xy: np.ndarray, radius: float) -> bool:
        return float(np.linalg.norm(np.asarray(wp_xy, dtype=float) - np.asarray(position_xy, dtype=float))) <= float(radius)

    @staticmethod
    def _remaining_glide_distance(z: float, z_target: float, horizontal_velocity: float, sink_velocity: float) -> float:
        t_remain = max(0.0, (float(z) - float(z_target)) / max(float(sink_velocity), 1.0e-6))
        return float(horizontal_velocity) * t_remain

    @staticmethod
    def _compute_ftp(params: dict, ipi: np.ndarray) -> np.ndarray:
        wind_mag = float(params["wind_magnitude"])
        wind_unit = np.asarray(params["wind_unit_vector"], dtype=float)
        glide_time = float(params["final_approach_height"]) / max(float(params["sink_velocity"]), 1.0e-6)
        upwind_distance = (float(params["horizontal_velocity"]) - wind_mag) * glide_time
        return ipi[:2] + upwind_distance * wind_unit

    def _params_from_mission(self, mission: MissionContext, observation: Observation, dt: float) -> dict:
        if mission.extras:
            params = mission.extras
        else:
            params = {}
            mission.extras = params

        params.setdefault("deployment_pos", observation.inertial_position.copy())
        params.setdefault("update_rate", float(dt))
        params.setdefault("mode", self.config.mode)
        params.setdefault("IPI", np.asarray(self.config.IPI, dtype=float))
        params.setdefault("final_approach_height", float(self.config.final_approach_height))
        params.setdefault("flare_height", float(self.config.flare_height))
        params.setdefault("backup_height", float(self.config.backup_height))
        params.setdefault("horizontal_velocity", float(self.config.horizontal_velocity))
        params.setdefault("sink_velocity", float(self.config.sink_velocity))
        params.setdefault("spirialing_radius", float(self.config.spirialing_radius))
        params.setdefault("emc_offset", float(self.config.emc_offset))
        params.setdefault("reach_radius", float(self.config.reach_radius))
        params.setdefault("close_ipi_radius", float(self.config.close_ipi_radius))
        params.setdefault("low_height_threshold", float(self.config.low_height_threshold))
        params.setdefault("backup_leg_distance", float(self.config.backup_leg_distance))
        params.setdefault("desired_heading", float(self.config.desired_heading))
        params.setdefault("wind_unit_vector", np.array([1.0, 0.0], dtype=float))
        params.setdefault("wind_magnitude", 0.0)
        params.setdefault("wind_heading", 0.0)
        params.setdefault("em_heading", None)
        params.setdefault("xy_drift", np.zeros(2, dtype=float))
        params.setdefault("IPI_app", np.asarray(self.config.IPI, dtype=float).copy())
        params["update_rate"] = float(dt)
        return params
    _MODE_STYLES: dict[str, dict[str, object]] = {
        "homing": {"color": "tab:blue"},
        "energy-management": {"color": "tab:orange"},
        "approach-FTP": {"color": "tab:green"},
        "turn-into-wind": {"color": "tab:purple"},
        "flare": {"color": "tab:red"},
        "backup": {"color": "tab:brown"},
    }
