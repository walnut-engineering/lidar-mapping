"""
Point cloud filters — thin wrappers around commonly used operations.

These helpers are deliberately kept separate from the heavier
:mod:`~lidar_mapping.processing.point_cloud` module so that they can be
imported with zero Open3D dependency (they only require numpy).

Included filters
----------------
- :func:`range_filter`          — keep points within a distance range
- :func:`intensity_filter`      — keep points above an intensity threshold
- :func:`passthrough_filter`    — 1-D axis-aligned slice
- :func:`random_downsample`     — random sub-sampling
- :func:`farthest_point_sample` — FPS for uniform coverage
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def range_filter(
    points: np.ndarray,
    min_range: float = 0.5,
    max_range: float = 100.0,
) -> np.ndarray:
    """
    Keep only points whose Euclidean distance from the origin is within
    [*min_range*, *max_range*] (metres).

    Parameters
    ----------
    points:
        (N, 3+) float array.
    min_range, max_range:
        Distance limits in metres.

    Returns
    -------
    (M, K) subset array.
    """
    dist = np.linalg.norm(points[:, :3], axis=1)
    mask = (dist >= min_range) & (dist <= max_range)
    return points[mask]


def intensity_filter(
    points: np.ndarray,
    min_intensity: float = 1.0,
    intensity_column: int = 3,
) -> np.ndarray:
    """
    Keep only points with intensity above *min_intensity*.

    Parameters
    ----------
    points:
        (N, 4+) float array where column *intensity_column* holds intensity.
    min_intensity:
        Minimum intensity value (raw, 0–255 for VLP-16).
    intensity_column:
        Column index that contains the intensity value.

    Returns
    -------
    (M, K) subset array.
    """
    if points.shape[1] <= intensity_column:
        return points
    mask = points[:, intensity_column] >= min_intensity
    return points[mask]


def passthrough_filter(
    points: np.ndarray,
    axis: int = 2,
    min_val: float = -10.0,
    max_val: float = 10.0,
) -> np.ndarray:
    """
    Keep only points where the value along *axis* is in [*min_val*, *max_val*].

    Parameters
    ----------
    points:
        (N, 3+) float array.
    axis:
        Column index (0=x, 1=y, 2=z).
    min_val, max_val:
        Inclusive range limits.

    Returns
    -------
    (M, K) subset array.
    """
    col = points[:, axis]
    mask = (col >= min_val) & (col <= max_val)
    return points[mask]


def random_downsample(
    points: np.ndarray,
    n_keep: int,
    seed: int = 0,
) -> np.ndarray:
    """
    Randomly sub-sample *points* to *n_keep* points.

    Parameters
    ----------
    points:
        (N, K) float array.
    n_keep:
        Target number of points.  If N ≤ n_keep the original array is returned.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    (min(N, n_keep), K) float array.
    """
    if len(points) <= n_keep:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=n_keep, replace=False)
    return points[idx]


def farthest_point_sample(
    points: np.ndarray,
    n_keep: int,
    seed: int = 0,
) -> np.ndarray:
    """
    Farthest Point Sampling (FPS) for spatially uniform coverage.

    This is an O(N × n_keep) pure-numpy implementation.  It is slower than
    voxel-grid downsampling but produces a more uniform spatial distribution,
    which can improve ICP convergence on sparse areas.

    Parameters
    ----------
    points:
        (N, 3+) float array (only XYZ used for distance calculation).
    n_keep:
        Target number of points.
    seed:
        Index of the starting point.

    Returns
    -------
    (n_keep, K) float array.
    """
    n = len(points)
    if n <= n_keep:
        return points

    xyz = points[:, :3].astype(np.float64)
    selected = np.zeros(n_keep, dtype=np.int64)
    distances = np.full(n, np.inf)

    current = seed % n
    selected[0] = current

    for i in range(1, n_keep):
        diff = xyz - xyz[current]
        dist = np.einsum("ij,ij->i", diff, diff)  # squared distances
        distances = np.minimum(distances, dist)
        current = int(np.argmax(distances))
        selected[i] = current

    return points[selected]


def compute_point_density(
    points: np.ndarray,
    radius: float = 1.0,
) -> np.ndarray:
    """
    Estimate local point density for each point.

    Returns the count of neighbours within *radius* metres for each point.
    Uses a brute-force approach; fine for point clouds up to ~50 k points.

    Parameters
    ----------
    points:
        (N, 3+) float array.
    radius:
        Search radius in metres.

    Returns
    -------
    (N,) int array of neighbour counts.
    """
    xyz = points[:, :3]
    n = len(xyz)
    counts = np.zeros(n, dtype=np.int32)
    r2 = radius * radius

    # Process in chunks to avoid O(N²) memory
    chunk = 1024
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        batch = xyz[start:end]  # (B, 3)
        diff = xyz[None, :, :] - batch[:, None, :]  # (B, N, 3)
        sq_dist = np.einsum("ijk,ijk->ij", diff, diff)  # (B, N)
        counts[start:end] = (sq_dist <= r2).sum(axis=1) - 1  # exclude self

    return counts
