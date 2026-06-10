"""
Camera ↔ LiDAR fusion: point cloud colorization.

Given:

* a LiDAR point cloud in the LiDAR frame
* a synchronized camera image
* the camera intrinsics
* the extrinsic transform from LiDAR → camera frame

this module projects each LiDAR point into the image plane and samples
the colour at the projected pixel.  Points behind the camera or outside
the image bounds are masked out.

The implementation is pure numpy (OpenCV is only consulted, lazily, for
optional lens distortion handling).

Typical use::

    from lidar_mapping.processing.fusion import (
        CameraIntrinsics, colorize_points,
    )

    intr = CameraIntrinsics(fx=900, fy=900, cx=640, cy=360,
                            width=1280, height=720)
    coloured = colorize_points(lidar_xyz, image_bgr, intr, T_cam_lidar)
    # coloured.points  -> (M, 3) float64 LiDAR-frame XYZ (masked subset)
    # coloured.colors  -> (M, 3) uint8 BGR
    # coloured.mask    -> (N,)  bool, which input points were kept
    # coloured.uv      -> (M, 2) float pixel coordinates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------

@dataclass
class CameraIntrinsics:
    """
    Pinhole camera intrinsics + optional Brown-Conrady distortion.

    Parameters
    ----------
    fx, fy:
        Focal lengths in pixels.
    cx, cy:
        Principal point in pixels.
    width, height:
        Image size in pixels.  Used to mask out-of-frame projections.
    distortion:
        Optional 5-vector ``[k1, k2, p1, p2, k3]`` (OpenCV convention).
        ``None`` (default) means no distortion.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("focal length must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width/height must be positive")
        if self.distortion is not None:
            d = np.asarray(self.distortion, dtype=np.float64).flatten()
            if d.shape != (5,):
                raise ValueError(
                    "distortion must be a 5-element vector [k1,k2,p1,p2,k3]"
                )
            self.distortion = d

    @property
    def K(self) -> np.ndarray:
        """(3, 3) camera matrix."""
        return np.array([
            [self.fx, 0.0,     self.cx],
            [0.0,     self.fy, self.cy],
            [0.0,     0.0,     1.0],
        ], dtype=np.float64)

    @classmethod
    def from_matrix(
        cls,
        K: np.ndarray,
        width: int,
        height: int,
        distortion: Optional[np.ndarray] = None,
    ) -> "CameraIntrinsics":
        K = np.asarray(K, dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError("K must be a (3, 3) matrix")
        return cls(
            fx=float(K[0, 0]), fy=float(K[1, 1]),
            cx=float(K[0, 2]), cy=float(K[1, 2]),
            width=int(width), height=int(height),
            distortion=distortion,
        )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ColorizedCloud:
    """Output of :func:`colorize_points`."""

    points: np.ndarray   # (M, 3) float64 XYZ in the original input frame
    colors: np.ndarray   # (M, 3) uint8, channel order matches the image
    mask: np.ndarray     # (N,) bool, which input points were retained
    uv: np.ndarray       # (M, 2) float, pixel coordinates (col, row)
    depths: np.ndarray   # (M,) float, depth in camera frame (Z)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def _apply_distortion(
    xn: np.ndarray, yn: np.ndarray, d: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Brown-Conrady forward distortion in normalised image coords."""
    k1, k2, p1, p2, k3 = d
    r2 = xn * xn + yn * yn
    r4 = r2 * r2
    r6 = r4 * r2
    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    xd = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
    yd = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
    return xd, yd


def project_points(
    points_cam: np.ndarray,
    intrinsics: CameraIntrinsics,
    min_depth: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project camera-frame points to pixel coordinates.

    Parameters
    ----------
    points_cam:
        (N, 3) array of XYZ in the camera frame (+Z forward, +X right,
        +Y down — standard OpenCV camera convention).
    intrinsics:
        :class:`CameraIntrinsics`.
    min_depth:
        Minimum Z to accept (points behind/at the camera are masked).

    Returns
    -------
    uv:
        (N, 2) array of (col, row) pixel coordinates.  Entries for points
        with Z < ``min_depth`` are NaN.
    valid:
        (N,) bool mask: True where the point is in front of the camera
        AND inside the image bounds.
    """
    pts = np.asarray(points_cam, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("points_cam must be (N, 3+)")

    z = pts[:, 2]
    in_front = z > min_depth

    # Avoid division warnings: replace bad z with 1 temporarily
    safe_z = np.where(in_front, z, 1.0)
    xn = pts[:, 0] / safe_z
    yn = pts[:, 1] / safe_z

    if intrinsics.distortion is not None:
        xn, yn = _apply_distortion(xn, yn, intrinsics.distortion)

    u = intrinsics.fx * xn + intrinsics.cx
    v = intrinsics.fy * yn + intrinsics.cy

    in_bounds = (
        (u >= 0) & (u < intrinsics.width)
        & (v >= 0) & (v < intrinsics.height)
    )
    valid = in_front & in_bounds

    uv = np.column_stack([u, v])
    uv[~in_front] = np.nan
    return uv, valid


# ---------------------------------------------------------------------------
# Colorization
# ---------------------------------------------------------------------------

def colorize_points(
    points: np.ndarray,
    image: np.ndarray,
    intrinsics: CameraIntrinsics,
    T_cam_from_points: Optional[np.ndarray] = None,
    sampling: str = "nearest",
    min_depth: float = 1e-3,
) -> ColorizedCloud:
    """
    Project ``points`` into ``image`` and sample colours.

    Parameters
    ----------
    points:
        (N, 3+) point cloud in any frame.  Use ``T_cam_from_points`` to
        transform them into the camera frame.
    image:
        (H, W, 3) uint8 image.  Channel order is preserved (BGR if the
        image came from OpenCV).
    intrinsics:
        Camera intrinsics matching ``image``.
    T_cam_from_points:
        Optional (4, 4) transform that takes ``points`` into the camera
        frame.  ``None`` ⇒ points are already in the camera frame.
    sampling:
        ``"nearest"`` (default) or ``"bilinear"``.
    min_depth:
        Minimum camera-frame depth to accept (metres).

    Returns
    -------
    :class:`ColorizedCloud`
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("points must be (N, 3+)")
    img = np.asarray(image)
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError("image must be an (H, W, 3 or 4) array")
    if img.shape[0] != intrinsics.height or img.shape[1] != intrinsics.width:
        raise ValueError(
            f"image shape {img.shape[:2]} does not match intrinsics "
            f"{(intrinsics.height, intrinsics.width)}"
        )
    if sampling not in ("nearest", "bilinear"):
        raise ValueError("sampling must be 'nearest' or 'bilinear'")

    xyz = pts[:, :3]

    if T_cam_from_points is not None:
        T = np.asarray(T_cam_from_points, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError("T_cam_from_points must be (4, 4)")
        ones = np.ones((len(xyz), 1), dtype=np.float64)
        cam_xyz = (T @ np.hstack([xyz, ones]).T).T[:, :3]
    else:
        cam_xyz = xyz

    uv, valid = project_points(cam_xyz, intrinsics, min_depth=min_depth)

    if not np.any(valid):
        return ColorizedCloud(
            points=np.zeros((0, 3), dtype=np.float64),
            colors=np.zeros((0, img.shape[2]), dtype=np.uint8),
            mask=valid,
            uv=np.zeros((0, 2), dtype=np.float64),
            depths=np.zeros((0,), dtype=np.float64),
        )

    uv_v = uv[valid]
    if sampling == "nearest":
        col = np.clip(np.round(uv_v[:, 0]).astype(int), 0,
                      intrinsics.width - 1)
        row = np.clip(np.round(uv_v[:, 1]).astype(int), 0,
                      intrinsics.height - 1)
        colors = img[row, col]
    else:  # bilinear
        u = np.clip(uv_v[:, 0], 0.0, intrinsics.width - 1.0001)
        v = np.clip(uv_v[:, 1], 0.0, intrinsics.height - 1.0001)
        u0 = np.floor(u).astype(int)
        v0 = np.floor(v).astype(int)
        u1 = u0 + 1
        v1 = v0 + 1
        du = (u - u0)[:, None]
        dv = (v - v0)[:, None]
        c00 = img[v0, u0].astype(np.float64)
        c01 = img[v0, u1].astype(np.float64)
        c10 = img[v1, u0].astype(np.float64)
        c11 = img[v1, u1].astype(np.float64)
        c0 = c00 * (1 - du) + c01 * du
        c1 = c10 * (1 - du) + c11 * du
        colors = np.clip(c0 * (1 - dv) + c1 * dv, 0, 255).astype(np.uint8)

    return ColorizedCloud(
        points=xyz[valid].astype(np.float64, copy=False),
        colors=colors,
        mask=valid,
        uv=uv_v,
        depths=cam_xyz[valid, 2],
    )


# ---------------------------------------------------------------------------
# Convenience: build a depth image from a coloured projection
# ---------------------------------------------------------------------------

def depth_image_from_points(
    points: np.ndarray,
    intrinsics: CameraIntrinsics,
    T_cam_from_points: Optional[np.ndarray] = None,
    min_depth: float = 1e-3,
) -> np.ndarray:
    """
    Render a sparse depth image by splatting ``points`` to pixels (nearest
    neighbour, closest depth wins).

    Returns an ``(H, W)`` float32 array where pixels with no point are
    ``np.inf``.
    """
    pts = np.asarray(points, dtype=np.float64)[:, :3]
    if T_cam_from_points is not None:
        T = np.asarray(T_cam_from_points, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError("T_cam_from_points must be (4, 4)")
        ones = np.ones((len(pts), 1), dtype=np.float64)
        pts = (T @ np.hstack([pts, ones]).T).T[:, :3]

    uv, valid = project_points(pts, intrinsics, min_depth=min_depth)
    depth = np.full((intrinsics.height, intrinsics.width),
                    np.inf, dtype=np.float32)
    if not np.any(valid):
        return depth

    uv_v = uv[valid]
    z_v = pts[valid, 2]
    col = np.clip(np.round(uv_v[:, 0]).astype(int), 0, intrinsics.width - 1)
    row = np.clip(np.round(uv_v[:, 1]).astype(int), 0, intrinsics.height - 1)

    # For each pixel keep the closest depth.  Sort by descending depth so
    # later (smaller) writes overwrite.
    order = np.argsort(-z_v)
    row_s = row[order]
    col_s = col[order]
    z_s = z_v[order]
    depth[row_s, col_s] = z_s.astype(np.float32)
    return depth
