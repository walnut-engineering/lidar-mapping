"""
Calibration configuration for the fusion pipeline.

For Phase 1/3 testing the sensors are colocated and we use identity
extrinsics. The dataclass below makes it trivial to swap in real
calibrated values later (typically loaded from a YAML file).

Conventions
-----------
* All transforms are 4x4 homogeneous matrices, ``T_a_b`` meaning
  "transform that maps a point expressed in frame ``b`` to frame ``a``".
* Camera intrinsics ``K`` are 3x3 row-major. ``dist`` is the OpenCV
  5-element distortion vector (k1, k2, p1, p2, k3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


def _default_K(width: int = 1280, height: int = 720, fov_deg: float = 70.0) -> np.ndarray:
    """A reasonable identity-extrinsics intrinsic guess for an uncalibrated camera.

    Uses pinhole approximation with focal length derived from the supplied
    horizontal field of view.
    """
    fx = (width / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    fy = fx  # square pixels
    cx, cy = width / 2.0, height / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


# Canonical rotation mapping VLP-16 sensor frame (x-forward, y-left, z-up)
# into OpenCV camera frame (x-right, y-down, z-forward), assuming the
# camera looks in the same forward direction as the LiDAR.
#   cam_x = -lidar_y
#   cam_y = -lidar_z
#   cam_z =  lidar_x
_R_CAM_FROM_LIDAR_FORWARD = np.array([
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
    [1.0,  0.0,  0.0],
], dtype=np.float64)


def _default_T_imu_cam() -> np.ndarray:
    """Camera frame transform relative to IMU (== LiDAR for colocated rig).

    With T_imu_lidar = I and T_imu_cam = this, we get
    T_cam_lidar = inv(T_imu_cam) @ I = canonical axis swap, which lets
    LiDAR points project sensibly into the camera image.
    """
    T = np.eye(4)
    # We want T_cam_lidar = inv(T_imu_cam). With T_imu_lidar = I, that means
    # T_imu_cam such that inv(T_imu_cam)[:3,:3] = R_cam_from_lidar_forward.
    # → T_imu_cam[:3,:3] = R^T.
    T[:3, :3] = _R_CAM_FROM_LIDAR_FORWARD.T
    return T


@dataclass
class CalibrationConfig:
    """Sensor extrinsics + camera intrinsics.

    Defaults: identity extrinsics (colocated sensors) and a 1280x720
    pinhole camera with 70° horizontal FOV. Swap in real values via
    ``CalibrationConfig.load_yaml(path)`` once calibrated.
    """

    # Extrinsics (4x4). Body / IMU frame is the reference.
    T_imu_lidar: np.ndarray = field(default_factory=lambda: np.eye(4))
    T_imu_cam: np.ndarray = field(default_factory=_default_T_imu_cam)

    # Camera intrinsics
    image_width: int = 1280
    image_height: int = 720
    K: np.ndarray = field(default_factory=lambda: _default_K(1280, 720, 70.0))
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float64))

    # Fraction of image width to keep for visual odometry feature
    # detection. 1.0 = use full frame. Lower values (e.g. 0.6) crop to
    # the central column so wide-FOV lens distortion at the edges does
    # not break the no-distortion pinhole assumption used by solvePnP.
    # Full image is still used for LiDAR projection / overlay / depth.
    vo_center_fraction: float = 1.0

    # --------------------------------------------------------------
    # Derived helpers
    # --------------------------------------------------------------
    @property
    def T_cam_lidar(self) -> np.ndarray:
        """Transform a point in LiDAR frame to camera frame."""
        return np.linalg.inv(self.T_imu_cam) @ self.T_imu_lidar

    @property
    def T_cam_imu(self) -> np.ndarray:
        return np.linalg.inv(self.T_imu_cam)

    # --------------------------------------------------------------
    # I/O
    # --------------------------------------------------------------
    @classmethod
    def load_yaml(cls, path: str | Path) -> "CalibrationConfig":
        """Load calibration from a simple YAML file (lazy import)."""
        import yaml  # noqa: PLC0415

        data = yaml.safe_load(Path(path).read_text())
        cfg = cls()
        if "T_imu_lidar" in data:
            cfg.T_imu_lidar = np.asarray(data["T_imu_lidar"], dtype=np.float64)
        if "T_imu_cam" in data:
            cfg.T_imu_cam = np.asarray(data["T_imu_cam"], dtype=np.float64)
        if "K" in data:
            cfg.K = np.asarray(data["K"], dtype=np.float64)
        if "dist" in data:
            cfg.dist = np.asarray(data["dist"], dtype=np.float64)
        if "image_width" in data:
            cfg.image_width = int(data["image_width"])
        if "image_height" in data:
            cfg.image_height = int(data["image_height"])
        if "vo_center_fraction" in data:
            cfg.vo_center_fraction = float(data["vo_center_fraction"])
        return cfg

    @classmethod
    def default(cls, width: Optional[int] = None, height: Optional[int] = None,
                fov_deg: float = 70.0) -> "CalibrationConfig":
        w = width or 1280
        h = height or 720
        return cls(image_width=w, image_height=h, K=_default_K(w, h, fov_deg),
                   T_imu_cam=_default_T_imu_cam())
