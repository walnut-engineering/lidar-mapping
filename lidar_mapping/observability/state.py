"""
Thread-safe shared state describing the live fusion pipeline.

A single ``FusionState`` singleton is updated by the live pipeline threads
(LiDAR frontend, visual frontend, viewer) and read by the observability
HTTP server / MCP tools. All access is guarded by a single re-entrant lock;
heavy payloads (point clouds, images) are stored as references — readers
copy on demand under the lock.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import numpy as np


@dataclass
class SensorRates:
    lidar_hz: float = 0.0
    imu_hz: float = 0.0
    camera_hz: float = 0.0


@dataclass
class LoopConstraint:
    """A detected and verified loop closure between two poses."""
    
    keyframe_id_a: int
    keyframe_id_b: int
    transform: np.ndarray  # 4x4 relative pose (B from A)
    match_count: int
    inlier_count: int
    confidence: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class FusionState:
    """Snapshot of the live pipeline. All fields are intentionally
    independent so partial updates from different threads remain coherent."""

    # Lifecycle
    started_at: float = field(default_factory=time.time)
    running: bool = False

    # Pose / trajectory
    current_pose: np.ndarray = field(default_factory=lambda: np.eye(4))
    """World-from-body 4x4."""
    trajectory: Deque[tuple[float, np.ndarray]] = field(
        default_factory=lambda: deque(maxlen=20000)
    )

    # IMU
    last_imu_reading: Optional[object] = None  # IMUReading
    imu_roll_deg: float = 0.0
    imu_pitch_deg: float = 0.0
    imu_yaw_deg: float = 0.0

    # Maps & frames
    map_points_count: int = 0
    keyframe_count: int = 0
    loop_constraint_count: int = 0
    loop_constraints: Deque[LoopConstraint] = field(
        default_factory=lambda: deque(maxlen=1000)
    )
    latest_cloud: Optional[np.ndarray] = None  # (N,4) most recent LiDAR frame
    latest_camera_bgr: Optional[np.ndarray] = None  # raw BGR
    latest_camera_overlay_bgr: Optional[np.ndarray] = None  # debug overlay

    # Rates / health
    rates: SensorRates = field(default_factory=SensorRates)
    lidar_frames_total: int = 0
    imu_samples_total: int = 0
    camera_frames_total: int = 0

    # Renderer (set by viewer so MCP can render fresh PNG)
    viewer_canvas: Optional[object] = None

    # Live-tunable calibration shared with the visual frontend.
    # Use the lock when mutating fields on the calibration object.
    calibration: Optional[object] = None

    # Lock guarding all mutations / reads of mutable fields above
    lock: threading.RLock = field(default_factory=threading.RLock)

    # ------------------------------------------------------------------
    # Helpers (must be called under .lock OR they take it internally)
    # ------------------------------------------------------------------
    def set_pose(self, pose: np.ndarray, timestamp: Optional[float] = None) -> None:
        with self.lock:
            self.current_pose = np.asarray(pose, dtype=np.float64).copy()
            self.trajectory.append(
                (timestamp if timestamp is not None else time.monotonic(),
                 self.current_pose.copy())
            )

    def set_imu(self, reading) -> None:
        with self.lock:
            self.last_imu_reading = reading
            self.imu_roll_deg = float(reading.roll_deg)
            self.imu_pitch_deg = float(reading.pitch_deg)
            self.imu_yaw_deg = float(reading.yaw_deg)
            self.imu_samples_total += 1

    def set_camera(self, bgr: np.ndarray, overlay_bgr: Optional[np.ndarray] = None) -> None:
        with self.lock:
            self.latest_camera_bgr = bgr
            if overlay_bgr is not None:
                self.latest_camera_overlay_bgr = overlay_bgr
            self.camera_frames_total += 1

    def set_lidar_frame(self, cloud_xyzi: np.ndarray) -> None:
        with self.lock:
            self.latest_cloud = cloud_xyzi
            self.lidar_frames_total += 1

    def trajectory_snapshot(self, max_points: int = 1000) -> list[tuple[float, np.ndarray]]:
        """Return a copy of the trajectory, decimated to ``max_points``."""
        with self.lock:
            traj = list(self.trajectory)
        if len(traj) <= max_points:
            return traj
        step = max(1, len(traj) // max_points)
        return traj[::step]


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_STATE: Optional[FusionState] = None
_STATE_LOCK = threading.Lock()


def get_state() -> FusionState:
    """Return the process-wide ``FusionState`` singleton."""
    global _STATE
    if _STATE is None:
        with _STATE_LOCK:
            if _STATE is None:
                _STATE = FusionState()
    return _STATE


def reset_state() -> FusionState:
    """Replace the singleton with a fresh state (mainly for tests)."""
    global _STATE
    with _STATE_LOCK:
        _STATE = FusionState()
    return _STATE
