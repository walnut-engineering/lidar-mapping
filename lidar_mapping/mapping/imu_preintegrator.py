"""
IMU-aided scan preintegration for LiDAR mapping.

The :class:`IMUPreintegrator` collects :class:`~lidar_mapping.sensors.imu.IMUReading`
objects between successive LiDAR frames and produces a **relative rotation
transform** that the :class:`~lidar_mapping.mapping.mapper.Mapper` uses as an
initial-guess hint for ICP registration.

Using IMU pre-integration as a motion prior dramatically improves ICP
convergence when:

* The platform is moving quickly between scans.
* The environment is geometrically ambiguous (long straight corridors, open
  terrain).
* You want to reduce ICP iterations (and therefore CPU load on the RPi).

How it works
------------
Between each pair of LiDAR scans the preintegrator:

1. Collects all IMU readings in the interval.
2. Integrates gyroscope angular-velocity measurements to get a ΔRotation.
3. (Optional) corrects for gravity tilt using the accelerometer.
4. Returns the resulting (4×4) homogeneous rotation matrix as the ICP hint.

Translation is *not* estimated — purely rotational — because without wheel
odometry or GPS we cannot reliably integrate accelerometer data for position.
The ICP registration handles the translation part.

Usage::

    from lidar_mapping.mapping.imu_preintegrator import IMUPreintegrator
    from lidar_mapping.sensors.imu import MPU9250Driver
    from lidar_mapping.mapping.mapper import Mapper

    imu = MPU9250Driver(i2c_bus=1)
    imu.start()

    preint = IMUPreintegrator()
    mapper = Mapper(voxel_size=0.1)

    while capturing:
        lidar_frame = lidar_driver.get_frame()
        # Drain all IMU readings accumulated since the last LiDAR frame
        while imu.readings_available():
            reading = imu.get_reading(timeout=0)
            if reading:
                preint.push(reading)

        hint = preint.consume()          # (4,4) rotation hint
        mapper.add_scan(lidar_frame.to_numpy(), transform_hint=hint)
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from lidar_mapping.sensors.imu import IMUReading
from lidar_mapping.utils.transforms import (
    _quaternion_to_rotation,
    make_transform,
)

logger = logging.getLogger(__name__)


class IMUPreintegrator:
    """
    Accumulates IMU readings and integrates gyro to get a relative rotation.

    Parameters
    ----------
    use_quaternion_integration:
        If ``True`` (default), integrate angular velocity directly into a
        quaternion (more accurate for large rotations).  If ``False``, use
        small-angle matrix integration (slightly faster, fine for slow
        platforms).
    max_readings:
        Maximum number of readings to buffer.  Older readings are discarded.
    min_readings:
        Minimum readings required before :meth:`consume` returns a non-identity
        transform.  Prevents noisy single-sample hints.
    gyro_noise_threshold:
        Angular velocity magnitude below which a reading is treated as
        stationary and ignored (rad/s).  Reduces drift from gyro bias when
        the platform is still.
    """

    def __init__(
        self,
        use_quaternion_integration: bool = True,
        max_readings: int = 5000,
        min_readings: int = 2,
        gyro_noise_threshold: float = 0.005,
    ) -> None:
        self._use_quat = use_quaternion_integration
        self._max_readings = max_readings
        self._min_readings = min_readings
        self._noise_thr = gyro_noise_threshold
        self._readings: List[IMUReading] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, reading: IMUReading) -> None:
        """
        Add one IMU reading to the buffer.

        Parameters
        ----------
        reading:
            A :class:`~lidar_mapping.sensors.imu.IMUReading` instance.
        """
        self._readings.append(reading)
        if len(self._readings) > self._max_readings:
            self._readings.pop(0)

    def consume(self) -> np.ndarray:
        """
        Compute and return the integrated relative rotation, then clear
        the buffer.

        Returns
        -------
        (4, 4) float64 homogeneous transformation matrix representing the
        relative rotation since the last :meth:`consume` call.
        Returns the identity matrix if fewer than *min_readings* samples
        are available.
        """
        readings = self._readings
        self._readings = []

        if len(readings) < self._min_readings:
            return np.eye(4, dtype=np.float64)

        if self._use_quat:
            R = self._integrate_quaternion(readings)
        else:
            R = self._integrate_matrix(readings)

        T = make_transform(R, np.zeros(3, dtype=np.float64))
        logger.debug(
            "IMUPreintegrator: integrated %d readings → ΔR det=%.6f",
            len(readings),
            float(np.linalg.det(R)),
        )
        return T

    def peek(self) -> np.ndarray:
        """
        Same as :meth:`consume` but **does not** clear the buffer.

        Useful for inspecting the current integrated rotation without
        consuming it.

        Returns
        -------
        (4, 4) float64 homogeneous transformation matrix.
        """
        saved = list(self._readings)
        result = self.consume()
        self._readings = saved
        return result

    @property
    def buffered_count(self) -> int:
        """Number of IMU readings currently buffered."""
        return len(self._readings)

    def reset(self) -> None:
        """Discard all buffered readings."""
        self._readings.clear()

    # ------------------------------------------------------------------
    # Integration methods
    # ------------------------------------------------------------------

    def _integrate_quaternion(self, readings: List[IMUReading]) -> np.ndarray:
        """
        Integrate gyroscope measurements using quaternion kinematics.

        This is numerically superior to matrix integration for large angular
        velocities (fast-moving platforms).
        """
        # (w, x, y, z) — starts at identity
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        for i in range(1, len(readings)):
            dt = readings[i].timestamp - readings[i - 1].timestamp
            if dt <= 0 or dt > 1.0:
                # Bad timestamp delta (wrap or gap) — skip
                continue

            gyro = readings[i].gyro_rads.astype(np.float64)

            # Skip readings below the noise threshold
            omega = float(np.linalg.norm(gyro))
            if omega < self._noise_thr:
                continue

            # Quaternion rate: q_dot = 0.5 * q ⊗ [0, gx, gy, gz]
            w, x, y, z = q
            gx, gy, gz = gyro[0], gyro[1], gyro[2]
            dw = 0.5 * (-x * gx - y * gy - z * gz)
            dx = 0.5 * (w * gx + y * gz - z * gy)
            dy = 0.5 * (w * gy - x * gz + z * gx)
            dz = 0.5 * (w * gz + x * gy - y * gx)

            q = q + np.array([dw, dx, dy, dz]) * dt

            # Normalise
            norm = float(np.linalg.norm(q))
            if norm > 1e-10:
                q /= norm

        return _quaternion_to_rotation(q)

    def _integrate_matrix(self, readings: List[IMUReading]) -> np.ndarray:
        """
        Integrate gyroscope measurements using small-angle matrix increments.

        Slightly faster than quaternion integration; accurate for slow
        rotation rates (< ~30°/s).
        """
        R = np.eye(3, dtype=np.float64)

        for i in range(1, len(readings)):
            dt = readings[i].timestamp - readings[i - 1].timestamp
            if dt <= 0 or dt > 1.0:
                continue

            gyro = readings[i].gyro_rads.astype(np.float64)
            omega = float(np.linalg.norm(gyro))
            if omega < self._noise_thr:
                continue

            # Rodrigues' rotation formula for increment dR
            angle = omega * dt
            axis = gyro / omega
            K = _skew(axis)
            dR = (
                np.eye(3)
                + np.sin(angle) * K
                + (1.0 - np.cos(angle)) * (K @ K)
            )
            R = R @ dR

        # Re-orthogonalise via SVD to prevent numerical drift
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

        return R

    # ------------------------------------------------------------------
    # Gravity-based tilt correction (optional helper)
    # ------------------------------------------------------------------

    @staticmethod
    def tilt_from_accel(accel_mss: np.ndarray) -> tuple[float, float]:
        """
        Estimate roll and pitch from a static accelerometer reading.

        Parameters
        ----------
        accel_mss:
            (3,) accelerometer measurement in m/s².

        Returns
        -------
        (roll_rad, pitch_rad)
        """
        import math
        ax, ay, az = float(accel_mss[0]), float(accel_mss[1]), float(accel_mss[2])
        roll = math.atan2(ay, az)
        pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))
        return roll, pitch


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _skew(v: np.ndarray) -> np.ndarray:
    """Return the 3×3 skew-symmetric matrix of vector *v*."""
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )
