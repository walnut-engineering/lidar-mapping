"""
Time synchronisation and sensor fusion helpers.

Provides:
    - TimeAlignedBuffer  : ring buffer that returns the closest sample to a
                           query timestamp (used to align IMU samples with
                           LiDAR frame timestamps).
    - interpolate_pose   : SLERP+LERP between two timestamped poses.
    - estimate_clock_offset : robust offset between two monotonic clocks given
                              paired observations.
"""

from __future__ import annotations

import bisect
import threading
from dataclasses import dataclass
from typing import Generic, List, Optional, Tuple, TypeVar

import numpy as np

from lidar_mapping.utils.transforms import _quaternion_to_rotation, make_transform


T = TypeVar("T")


@dataclass
class _Stamped(Generic[T]):
    timestamp: float
    value: T


class TimeAlignedBuffer(Generic[T]):
    """
    Bounded ring buffer of timestamped samples that supports fast nearest-
    neighbour and bracketing queries.

    Thread-safe for single-producer / multi-consumer use.

    Parameters
    ----------
    max_samples:
        Maximum number of samples retained.  When exceeded, oldest are dropped.
    """

    def __init__(self, max_samples: int = 10_000) -> None:
        self._max = int(max_samples)
        self._timestamps: List[float] = []
        self._values: List[T] = []
        self._lock = threading.Lock()

    def push(self, timestamp: float, value: T) -> None:
        with self._lock:
            # Reject out-of-order beyond a small slop
            if self._timestamps and timestamp < self._timestamps[-1]:
                # Insert in sorted position to keep monotonic
                idx = bisect.bisect_left(self._timestamps, timestamp)
                self._timestamps.insert(idx, timestamp)
                self._values.insert(idx, value)
            else:
                self._timestamps.append(timestamp)
                self._values.append(value)
            if len(self._timestamps) > self._max:
                drop = len(self._timestamps) - self._max
                del self._timestamps[:drop]
                del self._values[:drop]

    def __len__(self) -> int:
        with self._lock:
            return len(self._timestamps)

    def clear(self) -> None:
        with self._lock:
            self._timestamps.clear()
            self._values.clear()

    # ------------------------------------------------------------------
    def nearest(self, t: float) -> Optional[Tuple[float, T]]:
        """Return (timestamp, value) closest to *t* or ``None`` if empty."""
        with self._lock:
            if not self._timestamps:
                return None
            idx = bisect.bisect_left(self._timestamps, t)
            if idx == 0:
                return self._timestamps[0], self._values[0]
            if idx == len(self._timestamps):
                return self._timestamps[-1], self._values[-1]
            before = self._timestamps[idx - 1]
            after = self._timestamps[idx]
            if abs(t - before) <= abs(after - t):
                return before, self._values[idx - 1]
            return after, self._values[idx]

    def bracket(self, t: float) -> Optional[Tuple[Tuple[float, T], Tuple[float, T]]]:
        """
        Return ((t_before, v_before), (t_after, v_after)) bracketing *t*.
        Returns ``None`` if *t* is outside the buffered range.
        """
        with self._lock:
            if len(self._timestamps) < 2:
                return None
            if t < self._timestamps[0] or t > self._timestamps[-1]:
                return None
            idx = bisect.bisect_left(self._timestamps, t)
            if idx == 0:
                idx = 1
            if idx == len(self._timestamps):
                idx -= 1
            return (
                (self._timestamps[idx - 1], self._values[idx - 1]),
                (self._timestamps[idx], self._values[idx]),
            )

    def range(self, t_start: float, t_end: float) -> List[Tuple[float, T]]:
        """Return all samples with timestamps in [t_start, t_end]."""
        with self._lock:
            lo = bisect.bisect_left(self._timestamps, t_start)
            hi = bisect.bisect_right(self._timestamps, t_end)
            return list(zip(self._timestamps[lo:hi], self._values[lo:hi]))


# ---------------------------------------------------------------------------
# Quaternion / pose interpolation
# ---------------------------------------------------------------------------

def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two (w,x,y,z) unit quaternions."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        # Nearly parallel — linear interp + renormalise
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)
    theta_0 = np.arccos(dot)
    sin_0 = np.sin(theta_0)
    s0 = np.sin((1.0 - t) * theta_0) / sin_0
    s1 = np.sin(t * theta_0) / sin_0
    return s0 * q0 + s1 * q1


def interpolate_pose(
    t0: float,
    pose0: np.ndarray,
    t1: float,
    pose1: np.ndarray,
    t: float,
) -> np.ndarray:
    """
    Interpolate a 4x4 SE(3) pose at time *t* using LERP for translation and
    SLERP for rotation.

    Extrapolates linearly outside [t0, t1].
    """
    if t1 == t0:
        return pose0.copy()
    alpha = (t - t0) / (t1 - t0)
    # Rotation → quaternion
    q0 = _rotation_to_quaternion(pose0[:3, :3])
    q1 = _rotation_to_quaternion(pose1[:3, :3])
    q = slerp(q0, q1, float(np.clip(alpha, 0.0, 1.0))) if 0 <= alpha <= 1 else (
        slerp(q0, q1, 1.0) if alpha > 1 else slerp(q0, q1, 0.0)
    )
    R = _quaternion_to_rotation(q)
    t_vec = pose0[:3, 3] + alpha * (pose1[:3, 3] - pose0[:3, 3])
    return make_transform(R, t_vec)


def _rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to (w,x,y,z) quaternion."""
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


# ---------------------------------------------------------------------------
# Clock offset estimation
# ---------------------------------------------------------------------------

def estimate_clock_offset(
    pairs: List[Tuple[float, float]],
    method: str = "median",
) -> float:
    """
    Given a list of (local_time, remote_time) pairs estimate the
    constant offset (remote − local) so that
    ``remote ≈ local + offset``.

    Parameters
    ----------
    pairs:
        Iterable of (local, remote) timestamps.
    method:
        "median" (default, robust) or "mean".
    """
    if not pairs:
        raise ValueError("Need at least one (local, remote) pair.")
    diffs = np.array([r - l for (l, r) in pairs], dtype=np.float64)
    if method == "median":
        return float(np.median(diffs))
    if method == "mean":
        return float(np.mean(diffs))
    raise ValueError(f"Unknown method: {method}")
