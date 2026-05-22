"""
Coordinate transform utilities.

Pure-numpy helpers for working with rigid-body transformations represented
as (4, 4) homogeneous matrices.

Functions
---------
- :func:`make_transform`        — build T from rotation + translation
- :func:`rotation_from_euler`   — Euler angles → 3×3 rotation matrix
- :func:`euler_from_rotation`   — 3×3 rotation matrix → Euler angles
- :func:`apply_transform`       — apply T to (N, 3) point array
- :func:`compose_transforms`    — chain two transforms
- :func:`invert_transform`      — efficiently invert a rigid T
- :func:`interpolate_transforms`— SLERP-based interpolation
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def rotation_from_euler(
    roll: float,
    pitch: float,
    yaw: float,
    degrees: bool = True,
) -> np.ndarray:
    """
    Build a 3×3 rotation matrix from ZYX Euler angles.

    The convention is: yaw (Z) first, then pitch (Y), then roll (X).
    This matches the convention used by most automotive / robotics systems.

    Parameters
    ----------
    roll, pitch, yaw:
        Rotation angles.
    degrees:
        If ``True`` (default), angles are in degrees; otherwise radians.

    Returns
    -------
    (3, 3) float64 rotation matrix.
    """
    if degrees:
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    # ZYX convention: R = Rz @ Ry @ Rx
    R = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )
    return R


def euler_from_rotation(
    R: np.ndarray,
    degrees: bool = True,
) -> Tuple[float, float, float]:
    """
    Extract ZYX Euler angles from a 3×3 rotation matrix.

    Returns
    -------
    (roll, pitch, yaw) tuple, in degrees if *degrees* is ``True``.
    """
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0

    if degrees:
        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)
    return roll, pitch, yaw


def make_transform(
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """
    Build a (4, 4) homogeneous transformation matrix.

    Parameters
    ----------
    rotation:
        (3, 3) rotation matrix.
    translation:
        (3,) translation vector.

    Returns
    -------
    (4, 4) float64 homogeneous matrix.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def make_transform_from_euler(
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    tx: float = 0.0,
    ty: float = 0.0,
    tz: float = 0.0,
    degrees: bool = True,
) -> np.ndarray:
    """
    Convenience wrapper: build a transform from Euler angles + translation.

    Returns
    -------
    (4, 4) float64 homogeneous matrix.
    """
    R = rotation_from_euler(roll, pitch, yaw, degrees=degrees)
    t = np.array([tx, ty, tz], dtype=np.float64)
    return make_transform(R, t)


def apply_transform(
    points: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """
    Apply a (4, 4) homogeneous transform to an (N, 3) point array.

    Parameters
    ----------
    points:
        (N, 3) or (N, 3+) array.  Only the first 3 columns are transformed.
    T:
        (4, 4) homogeneous transformation matrix.

    Returns
    -------
    (N, 3) float64 array of transformed points.
    """
    xyz = points[:, :3].astype(np.float64)
    ones = np.ones((len(xyz), 1), dtype=np.float64)
    hom = np.hstack([xyz, ones])          # (N, 4)
    transformed = (T @ hom.T).T           # (N, 4)
    return transformed[:, :3]


def compose_transforms(
    T1: np.ndarray,
    T2: np.ndarray,
) -> np.ndarray:
    """
    Compose two homogeneous transforms: ``T_out = T1 @ T2``.

    In robotics notation this chains a transform that maps A→B (T1) with
    one that maps B→C (T2) to produce one that maps A→C.

    Returns
    -------
    (4, 4) float64 matrix.
    """
    return (T1 @ T2).astype(np.float64)


def invert_transform(T: np.ndarray) -> np.ndarray:
    """
    Efficiently invert a rigid-body homogeneous transformation.

    For a proper rigid transform this is faster and more numerically stable
    than :func:`numpy.linalg.inv`.

    Returns
    -------
    (4, 4) float64 matrix.
    """
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -(R.T @ t)
    return T_inv


def interpolate_transforms(
    T1: np.ndarray,
    T2: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Linearly interpolate between two transforms.

    Translation is linearly interpolated; rotation uses SLERP via quaternions.

    Parameters
    ----------
    T1, T2:
        (4, 4) homogeneous transforms.
    alpha:
        Interpolation factor in [0, 1].  ``0`` returns T1, ``1`` returns T2.

    Returns
    -------
    (4, 4) interpolated transform.
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))

    # Interpolate translation
    t = (1 - alpha) * T1[:3, 3] + alpha * T2[:3, 3]

    # Interpolate rotation via quaternions (SLERP)
    q1 = _rotation_to_quaternion(T1[:3, :3])
    q2 = _rotation_to_quaternion(T2[:3, :3])
    q = _slerp(q1, q2, alpha)
    R = _quaternion_to_rotation(q)

    return make_transform(R, t)


# ---------------------------------------------------------------------------
# Internal quaternion helpers
# ---------------------------------------------------------------------------

def _rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert a 3×3 rotation matrix to a (w, x, y, z) unit quaternion."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


def _quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    """Convert a (w, x, y, z) unit quaternion to a 3×3 rotation matrix."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """SLERP between two unit quaternions."""
    dot = float(np.dot(q1, q2))
    # Ensure shortest path
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    dot = min(dot, 1.0)

    if dot > 0.9995:
        # Nearly identical — linear interpolation is fine
        return (q1 + t * (q2 - q1)) / np.linalg.norm(q1 + t * (q2 - q1))

    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    s1 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0
    return s1 * q1 + s2 * q2
