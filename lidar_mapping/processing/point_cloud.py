"""
Point cloud processing utilities.

Wraps Open3D operations with sensible defaults and type-safe helpers so the
rest of the codebase can work with both raw numpy arrays and Open3D objects
interchangeably.

Key functions
-------------
- :func:`numpy_to_o3d`  / :func:`o3d_to_numpy`   — array ↔ Open3D conversion
- :func:`voxel_downsample`                         — voxel-grid subsampling
- :func:`remove_statistical_outliers`              — noise removal
- :func:`estimate_normals`                         — surface normal estimation
- :func:`remove_ground_plane`                      — RANSAC ground removal
- :func:`crop_box`                                 — bounding-box crop
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import open3d as o3d

    _O3D_AVAILABLE = True
except ImportError:  # pragma: no cover
    _O3D_AVAILABLE = False


def _require_o3d() -> None:
    if not _O3D_AVAILABLE:
        raise ImportError(
            "open3d is required for point cloud processing. "
            "Install it with: pip install open3d"
        )


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def numpy_to_o3d(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
) -> "o3d.geometry.PointCloud":
    """
    Convert a numpy array to an Open3D :class:`~open3d.geometry.PointCloud`.

    Parameters
    ----------
    points:
        (N, 3) or (N, 4) float array.  If 4 columns are present the 4th
        column is treated as per-point intensity and mapped to a greyscale
        colour unless *colors* is provided.
    colors:
        Optional (N, 3) float array with RGB values in ``[0, 1]``.

    Returns
    -------
    :class:`~open3d.geometry.PointCloud`
    """
    _require_o3d()
    pcd = o3d.geometry.PointCloud()
    xyz = points[:, :3].astype(np.float64)
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(
            colors[:, :3].astype(np.float64)
        )
    elif points.shape[1] >= 4:
        intensity = points[:, 3].astype(np.float64)
        # Normalise intensity to [0, 1] for display
        i_min, i_max = intensity.min(), intensity.max()
        if i_max > i_min:
            intensity = (intensity - i_min) / (i_max - i_min)
        grey = np.stack([intensity, intensity, intensity], axis=1)
        pcd.colors = o3d.utility.Vector3dVector(grey)

    return pcd


def o3d_to_numpy(
    pcd: "o3d.geometry.PointCloud",
    include_colors: bool = False,
) -> np.ndarray:
    """
    Convert an Open3D :class:`~open3d.geometry.PointCloud` to numpy.

    Parameters
    ----------
    pcd:
        Input point cloud.
    include_colors:
        If ``True`` and the cloud has colours, return (N, 6) array
        [x, y, z, r, g, b]; otherwise return (N, 3).

    Returns
    -------
    numpy array (N, 3) or (N, 6).
    """
    _require_o3d()
    xyz = np.asarray(pcd.points, dtype=np.float32)
    if include_colors and pcd.has_colors():
        rgb = np.asarray(pcd.colors, dtype=np.float32)
        return np.hstack([xyz, rgb])
    return xyz


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def voxel_downsample(
    points: np.ndarray,
    voxel_size: float = 0.05,
) -> np.ndarray:
    """
    Voxel-grid downsample a point cloud.

    Parameters
    ----------
    points:
        (N, 3+) float array.
    voxel_size:
        Side length of each voxel in metres.

    Returns
    -------
    (M, K) float array where M ≤ N.
    """
    _require_o3d()
    pcd = numpy_to_o3d(points)
    down = pcd.voxel_down_sample(voxel_size=voxel_size)
    result = np.asarray(down.points, dtype=np.float32)
    # Re-attach extra columns (intensity etc.) cannot be preserved here
    return result


def remove_statistical_outliers(
    points: np.ndarray,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove statistical outliers using Open3D.

    Parameters
    ----------
    points:
        (N, 3+) float array.
    nb_neighbors:
        Number of neighbours to consider.
    std_ratio:
        Standard-deviation multiplier threshold.

    Returns
    -------
    inliers:
        Filtered (M, 3) point array.
    mask:
        Boolean mask of length N (``True`` = kept).
    """
    _require_o3d()
    pcd = numpy_to_o3d(points)
    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    mask = np.zeros(len(points), dtype=bool)
    mask[np.asarray(ind)] = True
    return np.asarray(cl.points, dtype=np.float32), mask


def remove_radius_outliers(
    points: np.ndarray,
    nb_points: int = 16,
    radius: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove radius outliers (points with fewer than *nb_points* neighbours
    within *radius* metres).

    Returns
    -------
    inliers, mask
    """
    _require_o3d()
    pcd = numpy_to_o3d(points)
    cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    mask = np.zeros(len(points), dtype=bool)
    mask[np.asarray(ind)] = True
    return np.asarray(cl.points, dtype=np.float32), mask


def estimate_normals(
    pcd: "o3d.geometry.PointCloud",
    radius: float = 0.1,
    max_nn: int = 30,
) -> "o3d.geometry.PointCloud":
    """
    Estimate surface normals for an Open3D point cloud in-place.

    Parameters
    ----------
    pcd:
        Input/output point cloud.
    radius:
        Search radius for KD-tree neighbour queries.
    max_nn:
        Maximum number of neighbours to use.

    Returns
    -------
    The same *pcd* object (modified in-place) for chaining.
    """
    _require_o3d()
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=max_nn
        )
    )
    pcd.orient_normals_towards_camera_location()
    return pcd


def crop_box(
    points: np.ndarray,
    x_range: Tuple[float, float] = (-50.0, 50.0),
    y_range: Tuple[float, float] = (-50.0, 50.0),
    z_range: Tuple[float, float] = (-5.0, 20.0),
) -> np.ndarray:
    """
    Crop a point cloud to an axis-aligned bounding box.

    Parameters
    ----------
    points:
        (N, 3+) float array.
    x_range, y_range, z_range:
        ``(min, max)`` limits for each axis.

    Returns
    -------
    (M, K) subset array.
    """
    mask = (
        (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1])
        & (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
        & (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
    )
    return points[mask]


# ---------------------------------------------------------------------------
# Ground plane removal
# ---------------------------------------------------------------------------

def remove_ground_plane(
    points: np.ndarray,
    distance_threshold: float = 0.15,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    height_threshold: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect and remove the dominant ground plane using RANSAC.

    Parameters
    ----------
    points:
        (N, 3+) float array.
    distance_threshold:
        Maximum distance from inlier points to the fitted plane (metres).
    ransac_n:
        Minimum points to fit a plane candidate.
    num_iterations:
        RANSAC iterations.
    height_threshold:
        Points with z below this value (in the sensor frame) are seeded for
        RANSAC ground sampling.

    Returns
    -------
    above_ground:
        Points classified as non-ground.
    ground:
        Points classified as ground.
    plane_model:
        (4,) array ``[a, b, c, d]`` for the plane equation ``ax+by+cz+d=0``.
    """
    _require_o3d()
    pcd = numpy_to_o3d(points)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    inlier_set = set(inliers)
    mask_ground = np.array(
        [i in inlier_set for i in range(len(points))], dtype=bool
    )
    return points[~mask_ground], points[mask_ground], np.array(plane_model)
