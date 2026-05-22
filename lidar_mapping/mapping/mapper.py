"""
3D map builder.

The :class:`Mapper` class maintains a global point-cloud map by fusing
successive LiDAR scans using ICP-based odometry.  It operates in two modes:

- **Online mode** – receives frames one at a time from
  :class:`~lidar_mapping.sensors.vlp16.VLP16Driver` and updates the map
  incrementally.
- **Offline / batch mode** – processes a list of numpy arrays in order.

The map can be saved and loaded in PCD or PLY format via Open3D.

Usage::

    from lidar_mapping.mapping.mapper import Mapper

    mapper = Mapper(voxel_size=0.1)
    for frame in frames:
        pts = frame.to_numpy()[:, :3]
        mapper.add_scan(pts)

    mapper.save_map("output/map.pcd")
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import open3d as o3d

    _O3D_AVAILABLE = True
except ImportError:  # pragma: no cover
    _O3D_AVAILABLE = False

from lidar_mapping.processing.filters import passthrough_filter, range_filter
from lidar_mapping.processing.point_cloud import (
    numpy_to_o3d,
    o3d_to_numpy,
    remove_ground_plane,
    voxel_downsample,
)
from lidar_mapping.processing.registration import (
    ICPRegistration,
    RegistrationResult,
)
from lidar_mapping.utils.transforms import apply_transform, compose_transforms

logger = logging.getLogger(__name__)


class Mapper:
    """
    Incremental 3D terrain mapper.

    Parameters
    ----------
    voxel_size:
        Voxel-grid resolution for both per-scan downsampling and map
        downsampling (metres).
    min_range, max_range:
        Distance filter applied to every incoming scan.
    z_min, z_max:
        Vertical (Z-axis) passthrough filter applied to every scan.
    remove_ground:
        If ``True``, the ground plane is detected and removed from each
        scan before fusion.
    icp_max_correspondence_distance:
        ICP correspondence threshold (metres).
    max_map_points:
        If the accumulated map exceeds this count the map is voxel-
        downsampled to keep memory usage bounded.
    """

    def __init__(
        self,
        voxel_size: float = 0.1,
        min_range: float = 0.5,
        max_range: float = 80.0,
        z_min: float = -3.0,
        z_max: float = 20.0,
        remove_ground: bool = False,
        icp_max_correspondence_distance: Optional[float] = None,
        max_map_points: int = 2_000_000,
    ) -> None:
        self._voxel = voxel_size
        self._min_range = min_range
        self._max_range = max_range
        self._z_min = z_min
        self._z_max = z_max
        self._remove_ground = remove_ground
        self._max_map = max_map_points

        self._icp = ICPRegistration(
            voxel_size=voxel_size,
            max_correspondence_distance=icp_max_correspondence_distance,
        )

        # Global accumulated map (M, 3) float32
        self._map: Optional[np.ndarray] = None
        # Running pose estimate  (world ← current sensor frame)
        self._pose: np.ndarray = np.eye(4, dtype=np.float64)
        # History of per-scan transforms
        self._poses: List[np.ndarray] = [np.eye(4, dtype=np.float64)]
        # Metrics
        self.scans_processed: int = 0
        self.total_processing_time: float = 0.0

        self._last_scan: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def map_points(self) -> Optional[np.ndarray]:
        """The current accumulated map as a (M, 3) float32 numpy array."""
        return self._map

    @property
    def current_pose(self) -> np.ndarray:
        """Current 6-DOF pose as a (4, 4) homogeneous transformation."""
        return self._pose.copy()

    @property
    def pose_history(self) -> List[np.ndarray]:
        """List of (4, 4) poses for each processed scan."""
        return list(self._poses)

    def add_scan(
        self,
        points: np.ndarray,
        transform_hint: Optional[np.ndarray] = None,
    ) -> RegistrationResult:
        """
        Integrate one LiDAR scan into the map.

        Parameters
        ----------
        points:
            (N, 3+) array of 3-D points in the *sensor frame*.
        transform_hint:
            (4, 4) initial transform guess for ICP (e.g. from IMU or wheel
            odometry).  If ``None``, the identity (or last registered pose)
            is used.

        Returns
        -------
        :class:`~lidar_mapping.processing.registration.RegistrationResult`
        """
        t0 = time.monotonic()
        pts = self._preprocess(points)

        result = RegistrationResult(
            transform=np.eye(4, dtype=np.float64), converged=True
        )

        if self._map is None or self._last_scan is None:
            # First scan — initialise the map
            world_pts = pts
            self._map = world_pts.astype(np.float32)
            logger.info(
                "Map initialised with %d points.", len(self._map)
            )
        else:
            # Register against the previous scan
            hint = (
                transform_hint
                if transform_hint is not None
                else np.eye(4, dtype=np.float64)
            )
            result = self._icp.register(
                source=pts,
                target=self._last_scan,
                initial_transform=hint,
            )

            if not result.converged:
                logger.warning(
                    "ICP did not converge (fitness=%.4f). "
                    "Scan skipped.",
                    result.fitness,
                )
            else:
                # Update global pose: current_pose = prev_pose ∘ rel_transform
                self._pose = compose_transforms(
                    self._pose, result.transform
                )
                # Transform points to world frame and merge
                world_pts = apply_transform(pts, self._pose)
                self._map = np.vstack(
                    [self._map, world_pts.astype(np.float32)]
                )

                if len(self._map) > self._max_map:
                    self._map = voxel_downsample(
                        self._map, voxel_size=self._voxel
                    )
                    logger.debug(
                        "Map downsampled to %d points.", len(self._map)
                    )

        self._last_scan = pts
        self._poses.append(self._pose.copy())
        self.scans_processed += 1
        self.total_processing_time += time.monotonic() - t0

        logger.debug(
            "Scan %d integrated: %d pts, map size=%d, "
            "fitness=%.4f, rmse=%.4f",
            self.scans_processed,
            len(pts),
            len(self._map) if self._map is not None else 0,
            result.fitness,
            result.inlier_rmse,
        )
        return result

    def get_map_o3d(self) -> "o3d.geometry.PointCloud":
        """Return the accumulated map as an Open3D PointCloud."""
        if not _O3D_AVAILABLE:
            raise ImportError("open3d is required.")
        if self._map is None:
            return o3d.geometry.PointCloud()
        return numpy_to_o3d(self._map)

    def save_map(
        self,
        path: str | Path,
        voxel_size: Optional[float] = None,
    ) -> None:
        """
        Save the accumulated map to a file.

        Supported formats: ``.pcd``, ``.ply``, ``.xyz``.

        Parameters
        ----------
        path:
            Output file path.
        voxel_size:
            If provided, downsample the map before saving.
        """
        if not _O3D_AVAILABLE:
            raise ImportError("open3d is required for saving point clouds.")
        if self._map is None:
            raise RuntimeError("No map data to save.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        pts = self._map
        if voxel_size is not None:
            pts = voxel_downsample(pts, voxel_size=voxel_size)

        pcd = numpy_to_o3d(pts)
        o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)
        logger.info("Map saved to '%s' (%d points).", path, len(pts))

    def load_map(self, path: str | Path) -> None:
        """
        Load a previously saved map.

        Parameters
        ----------
        path:
            Path to a ``.pcd``, ``.ply``, or ``.xyz`` file.
        """
        if not _O3D_AVAILABLE:
            raise ImportError("open3d is required for loading point clouds.")

        path = Path(path)
        pcd = o3d.io.read_point_cloud(str(path))
        self._map = np.asarray(pcd.points, dtype=np.float32)
        logger.info(
            "Map loaded from '%s' (%d points).", path, len(self._map)
        )

    def reset(self) -> None:
        """Reset the mapper to its initial state."""
        self._map = None
        self._pose = np.eye(4, dtype=np.float64)
        self._poses = [np.eye(4, dtype=np.float64)]
        self._last_scan = None
        self.scans_processed = 0
        self.total_processing_time = 0.0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _preprocess(self, points: np.ndarray) -> np.ndarray:
        """Apply filters and downsampling to a raw scan."""
        pts = points[:, :3].astype(np.float32)

        # Range filter
        pts = range_filter(pts, self._min_range, self._max_range)

        # Z passthrough
        pts = passthrough_filter(pts, axis=2, min_val=self._z_min, max_val=self._z_max)

        if len(pts) == 0:
            return pts

        # Ground removal (optional)
        if self._remove_ground and len(pts) > 100:
            pts, _, _ = remove_ground_plane(pts)

        # Voxel downsample
        if len(pts) > 0:
            pts = voxel_downsample(pts, self._voxel)

        return pts
