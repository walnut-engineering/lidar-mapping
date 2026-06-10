"""
Point cloud and map visualisation helpers.

All functions require Open3D.  Interactive functions (``show_*``) open a
GUI window and block until the user closes it.  ``save_screenshot`` uses
Open3D's headless off-screen renderer and never opens a window, making it
safe to call in CI / SSH sessions.

Usage::

    from lidar_mapping.visualization import show_point_cloud, save_screenshot
    import numpy as np

    pts = np.random.randn(5000, 3).astype(np.float32)
    save_screenshot(pts, "cloud.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

try:
    import open3d as o3d

    _O3D_AVAILABLE = True
except ImportError:  # pragma: no cover
    _O3D_AVAILABLE = False


def _require_o3d() -> None:
    if not _O3D_AVAILABLE:
        raise ImportError(
            "open3d is required for visualisation. "
            "Install it with: pip install open3d"
        )


# ---------------------------------------------------------------------------
# Cloud colouring
# ---------------------------------------------------------------------------

def create_colored_cloud(
    points: np.ndarray,
    color_by: str = "z",
    intensity_column: int = 3,
) -> "o3d.geometry.PointCloud":
    """
    Convert a numpy point array into a coloured Open3D point cloud.

    Parameters
    ----------
    points:
        (N, 3+) float array.  XYZ in the first three columns; an optional
        fourth column is treated as intensity.
    color_by:
        ``"z"``         — colour by height (blue=low, red=high).
        ``"intensity"`` — colour by the fourth column (requires N×4+ input).
        ``"uniform"``   — uniform grey ``[0.7, 0.7, 0.7]``.
    intensity_column:
        Column index of the intensity channel when ``color_by="intensity"``.

    Returns
    -------
    :class:`open3d.geometry.PointCloud`
    """
    _require_o3d()
    pts = np.asarray(points, dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])

    if color_by == "uniform":
        colors = np.full((len(pts), 3), 0.7, dtype=np.float64)

    elif color_by == "z":
        z = pts[:, 2]
        z_min, z_max = float(z.min()), float(z.max())
        z_range = z_max - z_min
        if z_range < 1e-9:
            t = np.full(len(z), 0.5)
        else:
            t = (z - z_min) / z_range  # 0 (low) → 1 (high)
        # Blue → cyan → green → yellow → red  (jet-like)
        colors = np.zeros((len(pts), 3), dtype=np.float64)
        colors[:, 0] = np.clip(1.5 - np.abs(t - 0.75) * 4, 0, 1)  # red channel
        colors[:, 1] = np.clip(1.5 - np.abs(t - 0.5)  * 4, 0, 1)  # green channel
        colors[:, 2] = np.clip(1.5 - np.abs(t - 0.25) * 4, 0, 1)  # blue channel

    elif color_by == "intensity":
        if pts.shape[1] <= intensity_column:
            raise ValueError(
                f"points has only {pts.shape[1]} columns; "
                f"intensity_column={intensity_column} is out of range."
            )
        val = pts[:, intensity_column].astype(np.float64)
        v_min, v_max = float(val.min()), float(val.max())
        v_range = v_max - v_min
        if v_range < 1e-9:
            t = np.full(len(val), 0.5)
        else:
            t = (val - v_min) / v_range
        colors = np.stack([t, t, t], axis=1)  # greyscale

    else:
        raise ValueError(
            f"color_by must be 'z', 'intensity', or 'uniform', got {color_by!r}"
        )

    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


# ---------------------------------------------------------------------------
# Interactive viewers (block until window closed)
# ---------------------------------------------------------------------------

def show_point_cloud(
    points: np.ndarray,
    window_name: str = "Point Cloud",
    point_size: float = 1.0,
    color_by: str = "z",
) -> None:
    """
    Open an interactive Open3D viewer for a point cloud.

    This function **blocks** until the user closes the window.

    Parameters
    ----------
    points:
        (N, 3+) float array.
    window_name:
        Window title.
    point_size:
        Rendered point size in pixels.
    color_by:
        Passed to :func:`create_colored_cloud`.
    """
    _require_o3d()
    pcd = create_colored_cloud(points, color_by=color_by)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size = point_size
    vis.run()
    vis.destroy_window()


def show_map_with_trajectory(
    map_points: np.ndarray,
    poses: list[np.ndarray],
    window_name: str = "Map + Trajectory",
    point_size: float = 1.0,
    color_by: str = "z",
) -> None:
    """
    Open an interactive viewer showing the accumulated map and the sensor
    trajectory extracted from a list of 4×4 pose matrices.

    This function **blocks** until the user closes the window.

    Parameters
    ----------
    map_points:
        (N, 3+) float array — the accumulated 3-D map.
    poses:
        List of (4, 4) homogeneous pose matrices (world-frame positions of
        each scan origin).
    window_name:
        Window title.
    point_size:
        Point size for the map cloud.
    color_by:
        Colour scheme for the map; passed to :func:`create_colored_cloud`.
    """
    _require_o3d()
    pcd = create_colored_cloud(map_points, color_by=color_by)

    geometries = [pcd]

    # Build trajectory as a LineSet
    if len(poses) >= 2:
        origins = np.array(
            [np.asarray(p, dtype=np.float64)[:3, 3] for p in poses]
        )
        lines = [[i, i + 1] for i in range(len(origins) - 1)]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(origins),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(
            [[1.0, 0.5, 0.0]] * len(lines)  # orange trajectory
        )
        geometries.append(line_set)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    for geom in geometries:
        vis.add_geometry(geom)
    opt = vis.get_render_option()
    opt.point_size = point_size
    vis.run()
    vis.destroy_window()


# ---------------------------------------------------------------------------
# Headless screenshot
# ---------------------------------------------------------------------------

def save_screenshot(
    points: np.ndarray,
    path: Union[str, Path],
    width: int = 1280,
    height: int = 720,
    color_by: str = "z",
    zoom: float = 0.7,
) -> None:
    """
    Render a point cloud to a PNG file without opening a GUI window.

    Uses Open3D's :class:`~open3d.visualization.rendering.OffscreenRenderer`
    so it works in headless environments (CI, SSH, Docker).

    Parameters
    ----------
    points:
        (N, 3+) float array.
    path:
        Output PNG file path.
    width, height:
        Image dimensions in pixels.
    color_by:
        Colour scheme; passed to :func:`create_colored_cloud`.
    zoom:
        Camera zoom factor — increase to show more of the scene.
    """
    _require_o3d()
    pcd = create_colored_cloud(points, color_by=color_by)

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([0.1, 0.1, 0.1, 1.0])

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 2.0
    renderer.scene.add_geometry("cloud", pcd, mat)

    # Auto-fit camera to the bounding box
    bounds = pcd.get_axis_aligned_bounding_box()
    renderer.setup_camera(60.0, bounds, zoom)

    img = renderer.render_to_image()
    o3d.io.write_image(str(path), img)
