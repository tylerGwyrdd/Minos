"""Pure 6-DoF dynamics computation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .aero import (
    calculate_aero_force_coeff,
    calculate_aero_forces,
    calculate_aero_moment_coeff,
    calculate_aero_moments,
)
from .frames import (
    body_to_inertial,
    get_angular_vel_skew,
    get_angular_vel_to_euler_rates_matrix,
    get_cdm,
)
from .types import AeroCoefficients, Inputs, PhysicalParams, State, StateDerivative


@dataclass
class DynamicsDiagnostics:
    """Rich intermediate/output values produced during derivative evaluation.

    This payload is intended for analysis, debugging, and plotting and mirrors
    key intermediate computations from the aerodynamic and rigid-body pipeline.
    """

    cdm: np.ndarray
    r_wb: np.ndarray
    t_angvel_to_euler: np.ndarray
    angular_vel_skew: np.ndarray
    angle_of_attack: float
    sideslip_angle: float
    va: np.ndarray
    va_mag: float
    w: np.ndarray
    CL: float
    CD: float
    CY: float
    Cl: float
    Cm: float
    Cn: float
    F_aero: np.ndarray
    F_g: np.ndarray
    F_fictious: np.ndarray
    M_aero: np.ndarray
    M_f_aero: np.ndarray
    M_fictious: np.ndarray
    M_total: np.ndarray
    acc: np.ndarray
    angular_acc: np.ndarray
    euler_rates: np.ndarray
    singularity_warning: bool
    invalid_airspeed_warning: bool


def update_flaps(
    flap_l: float,
    flap_r: float,
    flap_l_desired: float,
    flap_r_desired: float,
    dt: float,
    flap_time_constant: float,
) -> tuple[float, float, float, float]:
    """Advance flap actuators with a first-order rate limit.

    Parameters
    ----------
    flap_l, flap_r
        Current left/right flap deflections.
    flap_l_desired, flap_r_desired
        Target left/right flap deflections.
    dt
        Time increment in seconds.
    flap_time_constant
        Time (seconds) used to move from 0 to full command.

    Returns
    -------
    tuple[float, float, float, float]
        Updated ``(flap_l, flap_r, delta_a, delta_s)``.
    """
    max_rate = dt / flap_time_constant
    flap_l_new = flap_l + np.clip(flap_l_desired - flap_l, -max_rate, max_rate)
    flap_r_new = flap_r + np.clip(flap_r_desired - flap_r, -max_rate, max_rate)
    delta_a = flap_r_new - flap_l_new
    delta_s = 0.5 * (flap_l_new + flap_r_new)
    return flap_l_new, flap_r_new, delta_a, delta_s


def _wind_kinematics(
    state: State,
    wind_inertial: np.ndarray,
    cdm: np.ndarray,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, float, float, float, np.ndarray, bool]:
    """Compute relative wind and derived aerodynamic angles.

    Parameters
    ----------
    state
        Current vehicle state.
    wind_inertial
        Ambient inertial-frame wind vector.
    cdm
        Body-to-inertial rotation matrix.
    epsilon
        Numerical floor used to avoid division by zero.

    Returns
    -------
    tuple[np.ndarray, float, float, float, np.ndarray, bool]
        ``(va, va_mag, alpha, beta, r_wb, invalid_airspeed_warning)``.
    """
    va = state.velocity_body - body_to_inertial(wind_inertial, cdm, inverse=True)
    invalid_airspeed_warning = False
    if not np.all(np.isfinite(va)):
        invalid_airspeed_warning = True
        va = np.array([epsilon, epsilon, epsilon], dtype=float)
    va_mag = np.linalg.norm(va)
    if va_mag < epsilon:
        va_mag = epsilon
    x_safe = va[0] if abs(va[0]) > epsilon else epsilon * np.sign(va[0] if va[0] != 0 else 1.0)
    angle_of_attack = np.arctan2(va[2], x_safe)
    denom = np.sqrt(va[0] ** 2 + va[2] ** 2)
    denom_safe = denom if denom > epsilon else epsilon
    sideslip_angle = np.arctan2(va[1], denom_safe)
    r_wb = np.array(
        [
            [
                np.cos(angle_of_attack) * np.cos(sideslip_angle),
                -np.sin(sideslip_angle),
                np.cos(sideslip_angle) * np.sin(angle_of_attack),
            ],
            [
                np.sin(angle_of_attack) * np.cos(sideslip_angle),
                np.cos(angle_of_attack),
                np.sin(angle_of_attack) * np.sin(sideslip_angle),
            ],
            [-np.sin(angle_of_attack), 0.0, np.cos(angle_of_attack)],
        ]
    )
    return va, va_mag, angle_of_attack, sideslip_angle, r_wb, invalid_airspeed_warning


def compute_derivatives(
    state: State,
    inputs: Inputs,
    params: PhysicalParams,
    coeffs: AeroCoefficients,
    delta_a: float,
    delta_s: float,
) -> tuple[StateDerivative, DynamicsDiagnostics]:
    """Compute 6-DoF rigid-body state derivatives and diagnostics.

    Parameters
    ----------
    state
        Current state.
    inputs
        Current control and wind inputs.
    params
        Physical constants/geometry.
    coeffs
        Aerodynamic coefficients.
    delta_a
        Asymmetric flap deflection term used by moment equations.
    delta_s
        Symmetric flap deflection term used by lift equation.

    Returns
    -------
    tuple[StateDerivative, DynamicsDiagnostics]
        Derivative vector and full diagnostics snapshot for this evaluation.
    """
    cdm = get_cdm(state.eulers)
    angular_vel_skew = get_angular_vel_skew(state.angular_velocity)
    t_angvel_to_euler, singularity_warning = get_angular_vel_to_euler_rates_matrix(state.eulers)
    va, va_mag, angle_of_attack, sideslip_angle, r_wb, invalid_airspeed_warning = _wind_kinematics(
        state, inputs.wind_inertial, cdm
    )

    cd, cy, cl = calculate_aero_force_coeff(coeffs, angle_of_attack, sideslip_angle, params.rigging_angle, delta_s)
    cl_roll, cm_pitch, cn_yaw = calculate_aero_moment_coeff(
        coeffs,
        angle_of_attack,
        sideslip_angle,
        delta_a,
        va_mag,
        params.c,
        state.angular_velocity,
        params.rigging_angle,
    )

    f_aero = calculate_aero_forces(params, va_mag, cd, cy, cl, r_wb)
    f_g = body_to_inertial(np.array([0.0, 0.0, params.m * params.gravity]), cdm, inverse=True)
    f_fictious = -params.m * (angular_vel_skew @ state.velocity_body)
    f_total = f_aero + f_g + f_fictious
    acc = f_total / params.m

    m_aero = calculate_aero_moments(params, va_mag, cl_roll, cm_pitch, cn_yaw)
    m_f_aero = np.cross(params.Rp, f_aero)
    m_fictious = -(angular_vel_skew @ (params.I @ state.angular_velocity))
    m_total = m_aero + m_fictious
    angular_acc = params.I_inv @ m_total

    euler_rates = t_angvel_to_euler @ state.angular_velocity
    position_dot = body_to_inertial(state.velocity_body, cdm, inverse=False)
    state_dot = StateDerivative(position_dot=position_dot, velocity_dot=acc, eulers_dot=euler_rates, angular_velocity_dot=angular_acc)

    diagnostics = DynamicsDiagnostics(
        cdm=cdm,
        r_wb=r_wb,
        t_angvel_to_euler=t_angvel_to_euler,
        angular_vel_skew=angular_vel_skew,
        angle_of_attack=angle_of_attack,
        sideslip_angle=sideslip_angle,
        va=va,
        va_mag=va_mag,
        w=inputs.wind_inertial.copy(),
        CL=cl,
        CD=cd,
        CY=cy,
        Cl=cl_roll,
        Cm=cm_pitch,
        Cn=cn_yaw,
        F_aero=f_aero,
        F_g=f_g,
        F_fictious=f_fictious,
        M_aero=m_aero,
        M_f_aero=m_f_aero,
        M_fictious=m_fictious,
        M_total=m_total,
        acc=acc,
        angular_acc=angular_acc,
        euler_rates=euler_rates,
        singularity_warning=singularity_warning,
        invalid_airspeed_warning=invalid_airspeed_warning,
    )
    return state_dot, diagnostics
