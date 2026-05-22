"""
Pure-numpy AHRS (Attitude and Heading Reference System) filters.

Two complementary filter implementations are provided:

Madgwick filter
    Sebastian Madgwick's gradient-descent AHRS algorithm.
    Ref: https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/
    Handles 6-DOF (accel + gyro) and 9-DOF (accel + gyro + mag) inputs.
    Good all-round accuracy; recommended for most use cases.

Mahony filter
    Robert Mahony's complementary filter using proportional-integral feedback.
    Slightly simpler and faster than Madgwick; works well at higher sample rates
    or on memory-constrained hardware (e.g. Raspberry Pi Zero).

Both classes maintain internal quaternion state and expose orientation as
roll / pitch / yaw (degrees) as well as the raw quaternion.

Usage::

    from lidar_mapping.sensors.ahrs import MadgwickAHRS

    ahrs = MadgwickAHRS(sample_rate=100.0, beta=0.1)

    # Call in your sensor loop:
    ahrs.update(gyro_rads, accel_mss, mag_ut)   # 9-DOF
    # or
    ahrs.update_imu(gyro_rads, accel_mss)         # 6-DOF

    roll, pitch, yaw = ahrs.euler_degrees
    q = ahrs.quaternion   # (w, x, y, z)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Madgwick AHRS
# ---------------------------------------------------------------------------

class MadgwickAHRS:
    """
    Madgwick AHRS filter (pure numpy).

    The filter fuses gyroscope, accelerometer and (optionally) magnetometer
    data into a quaternion attitude estimate.

    Parameters
    ----------
    sample_rate:
        Sensor sample rate in Hz.
    beta:
        Algorithm gain.  Higher values converge faster but are noisier.
        Typical range: 0.033 – 0.1.  Use ~0.033 for steady-state,
        ~0.1 during initialisation.
    """

    def __init__(
        self,
        sample_rate: float = 100.0,
        beta: float = 0.033,
    ) -> None:
        self._dt = 1.0 / sample_rate
        self.beta = beta
        # Quaternion (w, x, y, z), initialised to identity
        self._q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def quaternion(self) -> np.ndarray:
        """Current orientation as a (w, x, y, z) unit quaternion."""
        return self._q.copy()

    @property
    def euler_degrees(self) -> Tuple[float, float, float]:
        """
        Current orientation as ``(roll, pitch, yaw)`` in degrees.

        Convention: ZYX Euler angles (same as ``lidar_mapping.utils.transforms``).
        """
        return _quaternion_to_euler_degrees(self._q)

    def reset(self) -> None:
        """Reset the filter to the identity quaternion."""
        self._q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def update(
        self,
        gyro: np.ndarray,
        accel: np.ndarray,
        mag: np.ndarray,
    ) -> None:
        """
        9-DOF update step (gyro + accel + magnetometer).

        Parameters
        ----------
        gyro:
            (3,) angular velocity in rad/s [gx, gy, gz].
        accel:
            (3,) acceleration in m/s² or any consistent unit [ax, ay, az].
            Magnitude is normalised internally.
        mag:
            (3,) magnetic field in µT or any consistent unit [mx, my, mz].
            Magnitude is normalised internally.
        """
        q = self._q
        gx, gy, gz = float(gyro[0]), float(gyro[1]), float(gyro[2])
        ax, ay, az = float(accel[0]), float(accel[1]), float(accel[2])
        mx, my, mz = float(mag[0]), float(mag[1]), float(mag[2])

        # Normalise accelerometer — skip if near-zero
        norm_a = math.sqrt(ax * ax + ay * ay + az * az)
        if norm_a < 1e-10:
            self.update_imu(gyro, accel)
            return
        ax, ay, az = ax / norm_a, ay / norm_a, az / norm_a

        # Normalise magnetometer — skip if near-zero
        norm_m = math.sqrt(mx * mx + my * my + mz * mz)
        if norm_m < 1e-10:
            self.update_imu(gyro, accel)
            return
        mx, my, mz = mx / norm_m, my / norm_m, mz / norm_m

        q0, q1, q2, q3 = q

        # Reference direction of Earth's magnetic field
        hx = (
            2.0 * mx * (0.5 - q2 * q2 - q3 * q3)
            + 2.0 * my * (q1 * q2 - q0 * q3)
            + 2.0 * mz * (q1 * q3 + q0 * q2)
        )
        hy = (
            2.0 * mx * (q1 * q2 + q0 * q3)
            + 2.0 * my * (0.5 - q1 * q1 - q3 * q3)
            + 2.0 * mz * (q2 * q3 - q0 * q1)
        )
        hz = (
            2.0 * mx * (q1 * q3 - q0 * q2)
            + 2.0 * my * (q2 * q3 + q0 * q1)
            + 2.0 * mz * (0.5 - q1 * q1 - q2 * q2)
        )
        bx = math.sqrt(hx * hx + hy * hy)
        bz = hz

        # Gradient descent step (objective function Jacobian)
        # Equations (25)–(34) from Madgwick (2010)
        s0 = (
            -2.0 * q2 * (2.0 * (q1 * q3 - q0 * q2) - ax)
            + 2.0 * q1 * (2.0 * (q0 * q1 + q2 * q3) - ay)
            - 2.0 * bz * q2 * (2.0 * bx * (0.5 - q2 * q2 - q3 * q3) + 2.0 * bz * (q1 * q3 - q0 * q2) - mx)
            + (-2.0 * bx * q3 + 2.0 * bz * q1) * (2.0 * bx * (q1 * q2 - q0 * q3) + 2.0 * bz * (q0 * q1 + q2 * q3) - my)
            + 2.0 * bx * q2 * (2.0 * bx * (q0 * q2 + q1 * q3) + 2.0 * bz * (0.5 - q1 * q1 - q2 * q2) - mz)
        )
        s1 = (
            2.0 * q3 * (2.0 * (q1 * q3 - q0 * q2) - ax)
            + 2.0 * q0 * (2.0 * (q0 * q1 + q2 * q3) - ay)
            - 4.0 * q1 * (1.0 - 2.0 * (q1 * q1 + q2 * q2) - az)
            + 2.0 * bz * q3 * (2.0 * bx * (0.5 - q2 * q2 - q3 * q3) + 2.0 * bz * (q1 * q3 - q0 * q2) - mx)
            + (2.0 * bx * q2 + 2.0 * bz * q0) * (2.0 * bx * (q1 * q2 - q0 * q3) + 2.0 * bz * (q0 * q1 + q2 * q3) - my)
            + (2.0 * bx * q3 - 4.0 * bz * q1) * (2.0 * bx * (q0 * q2 + q1 * q3) + 2.0 * bz * (0.5 - q1 * q1 - q2 * q2) - mz)
        )
        s2 = (
            -2.0 * q0 * (2.0 * (q1 * q3 - q0 * q2) - ax)
            + 2.0 * q3 * (2.0 * (q0 * q1 + q2 * q3) - ay)
            - 4.0 * q2 * (1.0 - 2.0 * (q1 * q1 + q2 * q2) - az)
            + (-4.0 * bx * q2 - 2.0 * bz * q0) * (2.0 * bx * (0.5 - q2 * q2 - q3 * q3) + 2.0 * bz * (q1 * q3 - q0 * q2) - mx)
            + (2.0 * bx * q1 + 2.0 * bz * q3) * (2.0 * bx * (q1 * q2 - q0 * q3) + 2.0 * bz * (q0 * q1 + q2 * q3) - my)
            + (2.0 * bx * q0 - 4.0 * bz * q2) * (2.0 * bx * (q0 * q2 + q1 * q3) + 2.0 * bz * (0.5 - q1 * q1 - q2 * q2) - mz)
        )
        s3 = (
            2.0 * q1 * (2.0 * (q1 * q3 - q0 * q2) - ax)
            + 2.0 * q2 * (2.0 * (q0 * q1 + q2 * q3) - ay)
            + (-4.0 * bx * q3 + 2.0 * bz * q1) * (2.0 * bx * (0.5 - q2 * q2 - q3 * q3) + 2.0 * bz * (q1 * q3 - q0 * q2) - mx)
            + (-2.0 * bx * q0 + 2.0 * bz * q2) * (2.0 * bx * (q1 * q2 - q0 * q3) + 2.0 * bz * (q0 * q1 + q2 * q3) - my)
            + (2.0 * bx * q1) * (2.0 * bx * (q0 * q2 + q1 * q3) + 2.0 * bz * (0.5 - q1 * q1 - q2 * q2) - mz)
        )

        # Normalise gradient
        norm_s = math.sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3)
        if norm_s > 1e-10:
            s0, s1, s2, s3 = s0 / norm_s, s1 / norm_s, s2 / norm_s, s3 / norm_s

        # Rate of change of quaternion from gyroscope
        dq0 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz)
        dq1 = 0.5 * (q0 * gx + q2 * gz - q3 * gy)
        dq2 = 0.5 * (q0 * gy - q1 * gz + q3 * gx)
        dq3 = 0.5 * (q0 * gz + q1 * gy - q2 * gx)

        # Apply feedback
        beta = self.beta
        dq0 -= beta * s0
        dq1 -= beta * s1
        dq2 -= beta * s2
        dq3 -= beta * s3

        # Integrate
        dt = self._dt
        q0 += dq0 * dt
        q1 += dq1 * dt
        q2 += dq2 * dt
        q3 += dq3 * dt

        # Normalise
        norm_q = math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
        self._q = np.array(
            [q0 / norm_q, q1 / norm_q, q2 / norm_q, q3 / norm_q],
            dtype=np.float64,
        )

    def update_imu(
        self,
        gyro: np.ndarray,
        accel: np.ndarray,
    ) -> None:
        """
        6-DOF update step (gyro + accel only, no magnetometer).

        Yaw will drift over time without magnetometer correction.

        Parameters
        ----------
        gyro:
            (3,) angular velocity in rad/s.
        accel:
            (3,) acceleration in m/s².
        """
        q = self._q
        gx, gy, gz = float(gyro[0]), float(gyro[1]), float(gyro[2])
        ax, ay, az = float(accel[0]), float(accel[1]), float(accel[2])

        norm_a = math.sqrt(ax * ax + ay * ay + az * az)
        if norm_a < 1e-10:
            # Pure gyro integration fallback
            self._integrate_gyro(gyro)
            return
        ax, ay, az = ax / norm_a, ay / norm_a, az / norm_a

        q0, q1, q2, q3 = q

        # Gradient descent step
        s0 = (
            4.0 * q0 * (q2 * q2 + q1 * q1 - 0.5)
            + 2.0 * q2 * (2.0 * (q1 * q3 - q0 * q2) - ax)
            - 2.0 * q1 * (2.0 * (q0 * q1 + q2 * q3) - ay)
        )
        s1 = (
            4.0 * q1 * (q1 * q1 + q2 * q2 - 0.5)
            - 2.0 * q3 * (2.0 * (q1 * q3 - q0 * q2) - ax)
            - 2.0 * q0 * (2.0 * (q0 * q1 + q2 * q3) - ay)
            + 4.0 * q1 * (1.0 - 2.0 * q1 * q1 - 2.0 * q2 * q2 - az)
        )
        s2 = (
            4.0 * q2 * (q1 * q1 + q2 * q2 - 0.5)
            + 2.0 * q0 * (2.0 * (q1 * q3 - q0 * q2) - ax)
            - 2.0 * q3 * (2.0 * (q0 * q1 + q2 * q3) - ay)
            + 4.0 * q2 * (1.0 - 2.0 * q1 * q1 - 2.0 * q2 * q2 - az)
        )
        s3 = (
            2.0 * q1 * (2.0 * (q1 * q3 - q0 * q2) - ax)
            + 2.0 * q2 * (2.0 * (q0 * q1 + q2 * q3) - ay)
        )

        norm_s = math.sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3)
        if norm_s > 1e-10:
            s0, s1, s2, s3 = s0 / norm_s, s1 / norm_s, s2 / norm_s, s3 / norm_s

        dq0 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz)
        dq1 = 0.5 * (q0 * gx + q2 * gz - q3 * gy)
        dq2 = 0.5 * (q0 * gy - q1 * gz + q3 * gx)
        dq3 = 0.5 * (q0 * gz + q1 * gy - q2 * gx)

        beta = self.beta
        dq0 -= beta * s0
        dq1 -= beta * s1
        dq2 -= beta * s2
        dq3 -= beta * s3

        dt = self._dt
        q0 += dq0 * dt
        q1 += dq1 * dt
        q2 += dq2 * dt
        q3 += dq3 * dt

        norm_q = math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
        self._q = np.array(
            [q0 / norm_q, q1 / norm_q, q2 / norm_q, q3 / norm_q],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _integrate_gyro(self, gyro: np.ndarray) -> None:
        """Pure gyro integration (used when accel norm ≈ 0)."""
        q0, q1, q2, q3 = self._q
        gx, gy, gz = float(gyro[0]), float(gyro[1]), float(gyro[2])
        dt = self._dt
        q0 += 0.5 * (-q1 * gx - q2 * gy - q3 * gz) * dt
        q1 += 0.5 * (q0 * gx + q2 * gz - q3 * gy) * dt
        q2 += 0.5 * (q0 * gy - q1 * gz + q3 * gx) * dt
        q3 += 0.5 * (q0 * gz + q1 * gy - q2 * gx) * dt
        norm_q = math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
        if norm_q > 1e-10:
            self._q = np.array(
                [q0 / norm_q, q1 / norm_q, q2 / norm_q, q3 / norm_q],
                dtype=np.float64,
            )


# ---------------------------------------------------------------------------
# Mahony AHRS
# ---------------------------------------------------------------------------

class MahonyAHRS:
    """
    Mahony complementary AHRS filter (pure numpy).

    Uses a PI controller on the estimated gravity/north-direction errors to
    correct gyro integration.  Faster and simpler than Madgwick; recommended
    for high-rate sensors (≥ 200 Hz) or embedded platforms.

    Parameters
    ----------
    sample_rate:
        Sensor sample rate in Hz.
    kp:
        Proportional gain (corrects current error).
    ki:
        Integral gain (corrects gyro bias drift).
    """

    def __init__(
        self,
        sample_rate: float = 100.0,
        kp: float = 2.0,
        ki: float = 0.005,
    ) -> None:
        self._dt = 1.0 / sample_rate
        self.kp = kp
        self.ki = ki
        self._q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._integral_fb = np.zeros(3, dtype=np.float64)  # gyro bias estimate

    @property
    def quaternion(self) -> np.ndarray:
        """Current orientation as a (w, x, y, z) unit quaternion."""
        return self._q.copy()

    @property
    def euler_degrees(self) -> Tuple[float, float, float]:
        """Current orientation as ``(roll, pitch, yaw)`` in degrees."""
        return _quaternion_to_euler_degrees(self._q)

    def reset(self) -> None:
        """Reset the filter state."""
        self._q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._integral_fb = np.zeros(3, dtype=np.float64)

    def update(
        self,
        gyro: np.ndarray,
        accel: np.ndarray,
        mag: np.ndarray,
    ) -> None:
        """
        9-DOF update step (gyro + accel + magnetometer).

        Parameters
        ----------
        gyro:
            (3,) angular velocity in rad/s.
        accel:
            (3,) acceleration in m/s².
        mag:
            (3,) magnetic field in any consistent unit.
        """
        q0, q1, q2, q3 = self._q
        gx, gy, gz = float(gyro[0]), float(gyro[1]), float(gyro[2])

        # Normalise accel
        norm_a = math.sqrt(
            accel[0] ** 2 + accel[1] ** 2 + accel[2] ** 2
        )
        if norm_a < 1e-10:
            self.update_imu(gyro, accel)
            return
        ax = accel[0] / norm_a
        ay = accel[1] / norm_a
        az = accel[2] / norm_a

        # Normalise mag
        norm_m = math.sqrt(mag[0] ** 2 + mag[1] ** 2 + mag[2] ** 2)
        if norm_m < 1e-10:
            self.update_imu(gyro, accel)
            return
        mx = mag[0] / norm_m
        my = mag[1] / norm_m
        mz = mag[2] / norm_m

        # Reference direction of Earth's magnetic field (horizontal component)
        hx = (
            2.0 * mx * (0.5 - q2 * q2 - q3 * q3)
            + 2.0 * my * (q1 * q2 - q0 * q3)
            + 2.0 * mz * (q1 * q3 + q0 * q2)
        )
        hy = (
            2.0 * mx * (q1 * q2 + q0 * q3)
            + 2.0 * my * (0.5 - q1 * q1 - q3 * q3)
            + 2.0 * mz * (q2 * q3 - q0 * q1)
        )
        bx = math.sqrt(hx * hx + hy * hy)
        bz = (
            2.0 * mx * (q1 * q3 - q0 * q2)
            + 2.0 * my * (q2 * q3 + q0 * q1)
            + 2.0 * mz * (0.5 - q1 * q1 - q2 * q2)
        )

        # Estimated direction of gravity and flux
        vx = 2.0 * (q1 * q3 - q0 * q2)
        vy = 2.0 * (q0 * q1 + q2 * q3)
        vz = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3
        wx = 2.0 * bx * (0.5 - q2 * q2 - q3 * q3) + 2.0 * bz * (q1 * q3 - q0 * q2)
        wy = 2.0 * bx * (q1 * q2 - q0 * q3) + 2.0 * bz * (q0 * q1 + q2 * q3)
        wz = 2.0 * bx * (q0 * q2 + q1 * q3) + 2.0 * bz * (0.5 - q1 * q1 - q2 * q2)

        # Error: cross product of estimated and measured direction
        ex = (ay * vz - az * vy) + (my * wz - mz * wy)
        ey = (az * vx - ax * vz) + (mz * wx - mx * wz)
        ez = (ax * vy - ay * vx) + (mx * wy - my * wx)

        self._apply_feedback(gx, gy, gz, ex, ey, ez)

    def update_imu(
        self,
        gyro: np.ndarray,
        accel: np.ndarray,
    ) -> None:
        """
        6-DOF update step (gyro + accel, no magnetometer).

        Parameters
        ----------
        gyro:
            (3,) angular velocity in rad/s.
        accel:
            (3,) acceleration in m/s².
        """
        q0, q1, q2, q3 = self._q
        gx, gy, gz = float(gyro[0]), float(gyro[1]), float(gyro[2])

        norm_a = math.sqrt(accel[0] ** 2 + accel[1] ** 2 + accel[2] ** 2)
        if norm_a < 1e-10:
            return
        ax = accel[0] / norm_a
        ay = accel[1] / norm_a
        az = accel[2] / norm_a

        vx = 2.0 * (q1 * q3 - q0 * q2)
        vy = 2.0 * (q0 * q1 + q2 * q3)
        vz = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3

        ex = ay * vz - az * vy
        ey = az * vx - ax * vz
        ez = ax * vy - ay * vx

        self._apply_feedback(gx, gy, gz, ex, ey, ez)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _apply_feedback(
        self,
        gx: float, gy: float, gz: float,
        ex: float, ey: float, ez: float,
    ) -> None:
        ki = self.ki
        if ki > 0.0:
            self._integral_fb[0] += ex * self._dt
            self._integral_fb[1] += ey * self._dt
            self._integral_fb[2] += ez * self._dt
            gx += ki * self._integral_fb[0]
            gy += ki * self._integral_fb[1]
            gz += ki * self._integral_fb[2]

        kp = self.kp
        gx += kp * ex
        gy += kp * ey
        gz += kp * ez

        q0, q1, q2, q3 = self._q
        dt = self._dt
        q0 += 0.5 * (-q1 * gx - q2 * gy - q3 * gz) * dt
        q1 += 0.5 * (q0 * gx + q2 * gz - q3 * gy) * dt
        q2 += 0.5 * (q0 * gy - q1 * gz + q3 * gx) * dt
        q3 += 0.5 * (q0 * gz + q1 * gy - q2 * gx) * dt

        norm_q = math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
        if norm_q > 1e-10:
            self._q = np.array(
                [q0 / norm_q, q1 / norm_q, q2 / norm_q, q3 / norm_q],
                dtype=np.float64,
            )


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def _quaternion_to_euler_degrees(
    q: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Convert a (w, x, y, z) unit quaternion to ZYX Euler angles in degrees.

    Returns
    -------
    (roll, pitch, yaw) in degrees.
    """
    w, x, y, z = q

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)
