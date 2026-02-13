"""Coordinate frame transforms and kinematic mappings."""

from __future__ import annotations

import numpy as np


def get_cdm(eulers: np.ndarray) -> np.ndarray:
    """Compute direction cosine matrix from body to inertial frame.

    Parameters
    ----------
    eulers
        Euler angles ``[phi, theta, psi]`` in radians.

    Returns
    -------
    np.ndarray
        Rotation matrix with shape ``(3, 3)``.
    """
    phi, theta, psi = eulers
    return np.array(
        [
            [
                np.cos(theta) * np.cos(psi),
                np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi),
                np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi),
            ],
            [
                np.cos(theta) * np.sin(psi),
                np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi),
                np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi),
            ],
            [-np.sin(theta), np.sin(phi) * np.cos(theta), np.cos(phi) * np.cos(theta)],
        ]
    )


def get_angular_vel_skew(angular_vel: np.ndarray) -> np.ndarray:
    """Return skew matrix for cross-products with angular velocity.

    Parameters
    ----------
    angular_vel
        Body angular velocity vector ``[p, q, r]``.

    Returns
    -------
    np.ndarray
        Skew-symmetric matrix ``Omega_x`` so ``Omega_x @ v == omega x v``.
    """
    p, q, r = angular_vel
    return np.array([[0.0, -r, q], [r, 0.0, -p], [-q, p, 0.0]])


def get_angular_vel_to_euler_rates_matrix(
    eulers: np.ndarray,
    singularity_margin: float = 0.1,
) -> tuple[np.ndarray, bool]:
    """Compute mapping from body angular velocity to Euler rates.

    Parameters
    ----------
    eulers
        Euler angles ``[phi, theta, psi]`` in radians.
    singularity_margin
        Guard band around ``theta = pi/2`` used to avoid tangent blow-up.

    Returns
    -------
    tuple[np.ndarray, bool]
        ``(T, singularity_hit)`` where ``T @ [p, q, r]`` gives Euler rates.
    """
    phi, theta, _ = eulers
    singularity_hit = False
    if abs(theta - np.pi / 2.0) < singularity_margin:
        singularity_hit = True
        theta = np.pi / 2.0 * 0.999
    matrix = np.array(
        [
            [1.0, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
            [0.0, np.cos(phi), -np.sin(phi)],
            [0.0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)],
        ]
    )
    return matrix, singularity_hit


def body_to_inertial(vector: np.ndarray, cdm: np.ndarray, inverse: bool = False) -> np.ndarray:
    """Rotate vectors between body and inertial frames.

    Parameters
    ----------
    vector
        Vector to rotate.
    cdm
        Body-to-inertial direction cosine matrix.
    inverse
        If ``False`` rotate body -> inertial, else inertial -> body.

    Returns
    -------
    np.ndarray
        Rotated vector.
    """
    r_bi = cdm.T if inverse else cdm
    return r_bi @ vector


def body_to_wind(vector: np.ndarray, r_wb: np.ndarray, inverse: bool = False) -> np.ndarray:
    """Rotate vectors between body and wind frames.

    Parameters
    ----------
    vector
        Vector to rotate.
    r_wb
        Wind-to-body rotation matrix.
    inverse
        If ``False`` rotate body -> wind, else wind -> body.

    Returns
    -------
    np.ndarray
        Rotated vector.
    """
    r_bw = r_wb if inverse else r_wb.T
    return r_bw @ vector
