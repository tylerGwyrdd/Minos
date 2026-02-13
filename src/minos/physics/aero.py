"""Aerodynamic coefficient and load calculations."""

from __future__ import annotations

import numpy as np

from .frames import body_to_wind
from .types import AeroCoefficients, PhysicalParams, clamp_vector


def calculate_aero_force_coeff(
    coeffs: AeroCoefficients,
    angle_of_attack: float,
    sideslip_angle: float,
    rigging_angle: float,
    delta_s: float,
) -> tuple[float, float, float]:
    """Compute aerodynamic force coefficients ``(CD, CY, CL)``.

    Parameters
    ----------
    coeffs
        Aerodynamic coefficient bundle.
    angle_of_attack
        Angle of attack in radians.
    sideslip_angle
        Sideslip angle in radians.
    rigging_angle
        Fixed canopy rigging angle in radians.
    delta_s
        Symmetric flap deflection term.

    Returns
    -------
    tuple[float, float, float]
        ``(CD, CY, CL)``.
    """
    cl = coeffs.CLo + coeffs.CLa * (angle_of_attack + rigging_angle) + coeffs.CL_sym * delta_s
    cd = coeffs.CDo + coeffs.CDa * (angle_of_attack + rigging_angle)
    cy = -coeffs.CYB * sideslip_angle
    return cd, cy, cl


def calculate_aero_moment_coeff(
    coeffs: AeroCoefficients,
    angle_of_attack: float,
    sideslip_angle: float,
    delta_a: float,
    va_mag: float,
    c: float,
    angular_vel: np.ndarray,
    rigging_angle: float,
) -> tuple[float, float, float]:
    """Compute aerodynamic moment coefficients ``(Cl, Cm, Cn)``.

    Parameters
    ----------
    coeffs
        Aerodynamic coefficient bundle.
    angle_of_attack
        Angle of attack in radians.
    sideslip_angle
        Sideslip angle in radians.
    delta_a
        Asymmetric flap deflection term.
    va_mag
        Airspeed magnitude in m/s (must be positive).
    c
        Mean aerodynamic chord length in meters.
    angular_vel
        Body angular rates ``[p, q, r]`` in rad/s.
    rigging_angle
        Fixed canopy rigging angle in radians.

    Returns
    -------
    tuple[float, float, float]
        ``(Cl, Cm, Cn)``.
    """
    cl = (
        coeffs.ClB * sideslip_angle
        + coeffs.Cl_asym * delta_a
        + coeffs.Clp * c / (2.0 * va_mag) * angular_vel[0]
        + coeffs.Clr * c / (2.0 * va_mag) * angular_vel[2]
    )
    cm = coeffs.Cmo + coeffs.Cma * (angle_of_attack + rigging_angle) + coeffs.Cmq * c / (2.0 * va_mag) * angular_vel[1]
    cn = (
        coeffs.CnB * sideslip_angle
        + coeffs.Cn_asym * delta_a
        + coeffs.Cn_p * c / (2.0 * va_mag) * angular_vel[0]
        + coeffs.Cn_r * c / (2.0 * va_mag) * angular_vel[2]
    )
    return cl, cm, cn


def calculate_aero_forces(
    params: PhysicalParams,
    va_mag: float,
    cd: float,
    cy: float,
    cl: float,
    r_wb: np.ndarray,
) -> np.ndarray:
    """Compute aerodynamic force vector in body coordinates.

    Parameters
    ----------
    params
        Physical parameters containing density and reference area.
    va_mag
        Airspeed magnitude in m/s.
    cd, cy, cl
        Aerodynamic force coefficients.
    r_wb
        Wind-to-body rotation matrix.

    Returns
    -------
    np.ndarray
        Aerodynamic force vector in body frame, shape ``(3,)``.
    """
    scale = 0.5 * params.air_density * va_mag**2 * params.S
    f_aero_a = clamp_vector(np.array([scale * cd, scale * cy, scale * cl]))
    return -body_to_wind(f_aero_a, r_wb, inverse=False)


def calculate_aero_moments(
    params: PhysicalParams,
    va_mag: float,
    cl_roll: float,
    cm_pitch: float,
    cn_yaw: float,
) -> np.ndarray:
    """Compute aerodynamic moment vector in body coordinates.

    Parameters
    ----------
    params
        Physical parameters containing density and reference geometry.
    va_mag
        Airspeed magnitude in m/s.
    cl_roll, cm_pitch, cn_yaw
        Rolling, pitching, and yawing moment coefficients.

    Returns
    -------
    np.ndarray
        Aerodynamic moment vector ``[L, M, N]`` in body frame.
    """
    scale = 0.5 * params.air_density * va_mag**2 * params.S
    l = scale * params.b * cl_roll
    m = scale * params.c * cm_pitch
    n = scale * params.b * cn_yaw
    return clamp_vector(np.array([l, m, n]))
