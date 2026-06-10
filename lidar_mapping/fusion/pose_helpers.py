"""
Helper utilities for 4-DOF pose conversion and manipulation.

Converts between SE(3) 4x4 matrices and 4-DOF (x, y, z, yaw) representations.
"""

from __future__ import annotations

import numpy as np


def extract_4dof_from_se3(T: np.ndarray) -> np.ndarray:
    """
    Extract 4-DOF pose from SE(3) matrix.
    
    Parameters
    ----------
    T : np.ndarray
        4x4 SE(3) matrix
    
    Returns
    -------
    np.ndarray
        (4,) array [x, y, z, yaw]
    """
    # Extract position
    x, y, z = T[:3, 3]
    
    # Extract yaw from rotation matrix
    # For yaw-only rotation: R = Rz(yaw)
    # R[0,0] = cos(yaw), R[0,1] = -sin(yaw)
    # R[1,0] = sin(yaw), R[1,1] = cos(yaw)
    # yaw = atan2(R[1,0], R[0,0])
    R = T[:3, :3]
    yaw = np.arctan2(R[1, 0], R[0, 0])
    
    return np.array([x, y, z, yaw], dtype=np.float64)


def se3_from_4dof(x: float, y: float, z: float, yaw: float) -> np.ndarray:
    """
    Create SE(3) matrix from 4-DOF pose.
    
    Parameters
    ----------
    x, y, z : float
        Position coordinates
    yaw : float
        Rotation around Z axis (radians)
    
    Returns
    -------
    np.ndarray
        4x4 SE(3) matrix
    """
    T = np.eye(4, dtype=np.float64)
    
    # Position
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    
    # Rotation (only yaw around Z)
    c, s = np.cos(yaw), np.sin(yaw)
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    # T[2,2] = 1 (already set by eye)
    
    return T


def pose_4dof_delta(pose_a: np.ndarray, pose_b: np.ndarray) -> np.ndarray:
    """
    Compute delta between two 4-DOF poses.
    
    Parameters
    ----------
    pose_a, pose_b : np.ndarray
        (4,) poses [x, y, z, yaw]
    
    Returns
    -------
    np.ndarray
        (4,) delta pose (pose_b - pose_a)
    """
    return pose_b - pose_a


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle
