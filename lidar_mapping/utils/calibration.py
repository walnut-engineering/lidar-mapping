"""
Extrinsic calibration helpers.

Provides utilities for computing rigid-body transforms between sensors:

    - ``imu_to_lidar`` : a typed transform container.
    - ``compute_extrinsic_svd`` : Kabsch-style point-cloud → point-cloud fit.
    - ``hand_eye_calibration``  : AX=XB linear solver for IMU↔LiDAR offset.
    - ``apply_extrinsic``       : transform points/poses using an extrinsic.

These tools are exercised with synthetic data while the hardware is being
built; once the kit is assembled they can be reused with recorded sessions
to estimate the true offsets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from lidar_mapping.utils.transforms import (
    apply_transform,
    make_transform,
    compose_transforms,
)


@dataclass
class Extrinsic:
    """
    Rigid-body transform between two sensor frames.

    ``points_in_dst = transform @ points_in_src``
    """
    src_frame: str
    dst_frame: str
    transform: np.ndarray  # (4, 4)

    def inverse(self) -> "Extrinsic":
        return Extrinsic(
            src_frame=self.dst_frame,
            dst_frame=self.src_frame,
            transform=np.linalg.inv(self.transform),
        )


# ---------------------------------------------------------------------------
# Kabsch / SVD point-set alignment
# ---------------------------------------------------------------------------

def compute_extrinsic_svd(
    src_points: np.ndarray,
    dst_points: np.ndarray,
) -> np.ndarray:
    """
    Kabsch algorithm: find the rigid transform that aligns ``src_points`` to
    ``dst_points`` in the least-squares sense.

    Parameters
    ----------
    src_points, dst_points:
        (N, 3) corresponding 3-D points in the two frames.

    Returns
    -------
    (4, 4) homogeneous transform such that
    ``dst ≈ apply_transform(src, T)``.
    """
    src = np.asarray(src_points, dtype=np.float64)
    dst = np.asarray(dst_points, dtype=np.float64)
    if src.shape != dst.shape or src.shape[1] != 3:
        raise ValueError("src/dst must be (N,3) of same length")
    if len(src) < 3:
        raise ValueError("Need at least 3 correspondences")

    c_src = src.mean(axis=0)
    c_dst = dst.mean(axis=0)
    H = (src - c_src).T @ (dst - c_dst)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = c_dst - R @ c_src
    return make_transform(R, t)


# ---------------------------------------------------------------------------
# Hand-eye calibration (AX = XB)
# ---------------------------------------------------------------------------

def hand_eye_calibration(
    motions_a: List[np.ndarray],
    motions_b: List[np.ndarray],
) -> np.ndarray:
    """
    Solve the classical hand-eye problem AX = XB for the rigid transform X.

    Use cases:
        * IMU ↔ LiDAR extrinsic given paired motion segments
          (A = IMU relative motion, B = LiDAR relative motion).
        * Camera ↔ LiDAR extrinsic.

    Implementation: log-quaternion linearisation (Tsai-Lenz style).

    Parameters
    ----------
    motions_a, motions_b:
        Lists of N (4, 4) relative motions of the two sensors observed in
        the same time intervals.  N ≥ 2 required, more is better.

    Returns
    -------
    (4, 4) homogeneous transform X such that ``A_i @ X ≈ X @ B_i``.
    """
    if len(motions_a) != len(motions_b):
        raise ValueError("motions_a and motions_b must be the same length")
    n = len(motions_a)
    if n < 2:
        raise ValueError("Need at least 2 motion pairs")

    # --- Rotation step ---
    # Build big linear system K = [k_a × k_b ...] using rotation-axis logs
    M = np.zeros((3, 3), dtype=np.float64)
    for A, B in zip(motions_a, motions_b):
        alpha = _rot_to_axis_angle(A[:3, :3])
        beta = _rot_to_axis_angle(B[:3, :3])
        M += np.outer(beta, alpha)
    U, _, Vt = np.linalg.svd(M)
    Rx = Vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T

    # --- Translation step ---
    rows = []
    rhs = []
    I3 = np.eye(3)
    for A, B in zip(motions_a, motions_b):
        Ra = A[:3, :3]
        ta = A[:3, 3]
        tb = B[:3, 3]
        rows.append(Ra - I3)
        rhs.append(Rx @ tb - ta)
    Astack = np.vstack(rows)
    bstack = np.concatenate(rhs)
    tx, *_ = np.linalg.lstsq(Astack, bstack, rcond=None)
    return make_transform(Rx, tx)


def _rot_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Return rotation as axis × angle vector (3,)."""
    R = np.asarray(R, dtype=np.float64)
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = np.arccos(cos_theta)
    if theta < 1e-9:
        return np.zeros(3)
    if abs(theta - np.pi) < 1e-6:
        # Near-180° — fall back to axis from diag
        d = np.array([R[0, 0], R[1, 1], R[2, 2]])
        axis = np.sqrt(np.clip((d + 1.0) / 2.0, 0.0, None))
        return axis * theta
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2.0 * np.sin(theta))
    return axis * theta


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------

def apply_extrinsic_to_points(
    ext: Extrinsic, points_src: np.ndarray
) -> np.ndarray:
    """Transform a (N, 3) array of points from src frame to dst frame."""
    return apply_transform(points_src, ext.transform)


def chain_extrinsics(*extrinsics: Extrinsic) -> Extrinsic:
    """
    Compose a chain of extrinsics, e.g.  imu→lidar  ∘  lidar→camera
    →  imu→camera.

    Validates that adjacent dst/src frames match.
    """
    if not extrinsics:
        raise ValueError("Need at least one Extrinsic")
    cur = extrinsics[0]
    for ext in extrinsics[1:]:
        if cur.dst_frame != ext.src_frame:
            raise ValueError(
                f"Chain mismatch: {cur.dst_frame} → {ext.src_frame}"
            )
        cur = Extrinsic(
            src_frame=cur.src_frame,
            dst_frame=ext.dst_frame,
            transform=compose_transforms(ext.transform, cur.transform),
        )
    return cur
