"""
LiDAR frontend — scan-to-map registration with IMU rotation prior.

Pulls timestamped frames from :class:`SensorHub`, derives a delta-rotation
prior from the IMU's absolute orientation between successive LiDAR frames,
and feeds them into the existing :class:`Mapper` for ICP refinement and
map accumulation.

Results (current pose, latest cloud, map stats) are published into the
shared :class:`FusionState` so the observability server can render them.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

from lidar_mapping.fusion.sensor_hub import SensorHub
from lidar_mapping.mapping.mapper import Mapper
from lidar_mapping.observability.state import FusionState, get_state

logger = logging.getLogger(__name__)


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """(w,x,y,z) → 3x3."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def _delta_rotation(q_prev: np.ndarray, q_curr: np.ndarray) -> np.ndarray:
    """4x4 homogeneous transform for the body-frame rotation between
    two absolute orientations (world-from-body).

    R_delta_body = R_prev^T @ R_curr  (rotates points expressed at t_prev
    into the body frame at t_curr).
    """
    R_prev = _quat_to_rot(q_prev)
    R_curr = _quat_to_rot(q_curr)
    R_delta = R_prev.T @ R_curr
    T = np.eye(4)
    T[:3, :3] = R_delta
    return T


class LidarFrontend:
    """
    Background worker that consumes ``SensorHub.lidar`` frames in arrival
    order, builds an IMU-rotation prior between consecutive frames, calls
    ``Mapper.add_scan`` for each, and pushes results into ``FusionState``.

    Parameters
    ----------
    hub:
        The :class:`SensorHub` instance populated by sensor ingest threads.
    mapper:
        A configured :class:`Mapper`. If ``None``, a default one is created.
    state:
        Shared observability state. Defaults to the global singleton.
    use_imu_prior:
        If ``True`` (default), use IMU delta rotation as ICP hint.
    """

    def __init__(
        self,
        hub: SensorHub,
        mapper: Optional[Mapper] = None,
        state: Optional[FusionState] = None,
        use_imu_prior: bool = True,
        pose_source: str = "imu_only",
        max_map_points: int = 500_000,
    ) -> None:
        """
        pose_source:
            "imu_only" — trust the IMU absolute orientation as the pose.
                Skips ICP entirely; accumulates the world-frame point cloud
                directly. Robust on platforms where Open3D ICP is unstable
                (e.g. Mali-G610 / Orange Pi 5). Recommended for Phase 1
                stationary rotation tests.
            "icp" — run ``Mapper.add_scan`` with IMU rotation hint.
        """
        if pose_source not in ("imu_only", "icp"):
            raise ValueError(f"pose_source must be 'imu_only' or 'icp', got {pose_source!r}")
        self.hub = hub
        self.state = state or get_state()
        self.use_imu_prior = use_imu_prior
        self.pose_source = pose_source
        self.max_map_points = int(max_map_points)
        self.mapper = mapper if mapper is not None or pose_source == "imu_only" \
            else Mapper(voxel_size=0.15, min_range=0.5, max_range=80.0)
        # imu_only map accumulator (world frame)
        self._world_map: Optional[np.ndarray] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_lidar_t: Optional[float] = None
        self._last_imu_quat: Optional[np.ndarray] = None
        self._scan_errors = 0
        self.scans_processed = 0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="lidar-frontend"
        )
        self._thread.start()
        logger.info("LidarFrontend started.")

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self) -> None:
        while self._running:
            try:
                self._tick()
            except Exception as exc:  # noqa: BLE001
                self._scan_errors += 1
                logger.exception("LidarFrontend tick error: %s", exc)
                time.sleep(0.05)

    def _tick(self) -> None:
        # Wait for a new LiDAR frame newer than the last processed.
        latest = self.hub.lidar.latest()
        if latest is None:
            time.sleep(0.02)
            return
        t_curr, frame = latest
        if self._last_lidar_t is not None and t_curr <= self._last_lidar_t:
            time.sleep(0.02)
            return

        # Look up bracketing IMU absolute orientation.
        imu_reading = self.hub.imu_nearest(t_curr)
        q_curr = None
        if imu_reading is not None:
            q_curr = np.asarray(imu_reading.quaternion, dtype=np.float64)

        cloud = frame.to_numpy()  # (N,4) [x,y,z,intensity]

        if self.pose_source == "imu_only":
            self._tick_imu_only(t_curr, cloud, imu_reading, q_curr)
        else:
            self._tick_icp(t_curr, cloud, imu_reading, q_curr)

        self._last_imu_quat = q_curr
        self._last_lidar_t = t_curr
        self.scans_processed += 1

    # ------------------------------------------------------------------
    # imu_only pose source
    # ------------------------------------------------------------------
    def _tick_imu_only(self, t_curr, cloud, imu_reading, q_curr) -> None:
        if q_curr is None:
            # Nothing to do without orientation.
            return
        R = _quat_to_rot(q_curr)
        pose = np.eye(4)
        pose[:3, :3] = R
        # Transform sensor-frame points to world frame and accumulate.
        xyz = cloud[:, :3]
        world = xyz @ R.T
        if self._world_map is None:
            self._world_map = world.astype(np.float32)
        else:
            self._world_map = np.vstack(
                [self._world_map, world.astype(np.float32)]
            )
            if self._world_map.shape[0] > self.max_map_points:
                # Random subsample to bound memory.
                idx = np.random.choice(
                    self._world_map.shape[0],
                    size=self.max_map_points,
                    replace=False,
                )
                self._world_map = self._world_map[idx]

        with self.state.lock:
            self.state.current_pose = pose
            self.state.trajectory.append((t_curr, pose.copy()))
            self.state.map_points_count = int(self._world_map.shape[0])
            self.state.latest_cloud = cloud
            self.state.lidar_frames_total += 1
            self.state.rates.lidar_hz = float(self.hub.lidar_hz)
            self.state.rates.imu_hz = float(self.hub.imu_hz)
            self.state.rates.camera_hz = float(self.hub.camera_hz)
            self.state.last_imu_reading = imu_reading
            self.state.imu_roll_deg = float(imu_reading.roll_deg)
            self.state.imu_pitch_deg = float(imu_reading.pitch_deg)
            self.state.imu_yaw_deg = float(imu_reading.yaw_deg)

    # ------------------------------------------------------------------
    # ICP pose source (uses Mapper; may be unstable on Mali-G610)
    # ------------------------------------------------------------------
    def _tick_icp(self, t_curr, cloud, imu_reading, q_curr) -> None:
        # Delta-rotation prior from absolute IMU orientations.
        hint = None
        if self.use_imu_prior and q_curr is not None and self._last_imu_quat is not None:
            hint = _delta_rotation(self._last_imu_quat, q_curr)

        try:
            result = self.mapper.add_scan(cloud[:, :3], transform_hint=hint)
        except Exception as exc:  # noqa: BLE001
            self._scan_errors += 1
            logger.warning("Mapper.add_scan failed: %s", exc)
            return

        with self.state.lock:
            self.state.current_pose = self.mapper.current_pose.copy()
            self.state.trajectory.append((t_curr, self.state.current_pose.copy()))
            self.state.map_points_count = int(self.mapper.map_points.shape[0]) \
                if self.mapper.map_points is not None else 0
            self.state.latest_cloud = cloud
            self.state.lidar_frames_total += 1
            self.state.rates.lidar_hz = float(self.hub.lidar_hz)
            self.state.rates.imu_hz = float(self.hub.imu_hz)
            self.state.rates.camera_hz = float(self.hub.camera_hz)
            if imu_reading is not None:
                self.state.last_imu_reading = imu_reading
                self.state.imu_roll_deg = float(imu_reading.roll_deg)
                self.state.imu_pitch_deg = float(imu_reading.pitch_deg)
                self.state.imu_yaw_deg = float(imu_reading.yaw_deg)
        if self.scans_processed % 10 == 0:
            fitness = getattr(result, "fitness", float("nan"))
            logger.debug("scan %d: fitness=%.3f map_pts=%d",
                         self.scans_processed, fitness,
                         self.state.map_points_count)

    @property
    def world_map(self) -> Optional[np.ndarray]:
        """For imu_only mode: the accumulated world-frame point cloud."""
        return self._world_map
