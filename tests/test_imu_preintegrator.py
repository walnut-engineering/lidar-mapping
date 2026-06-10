"""
Tests for IMUPreintegrator — gyroscope-based rotation integration.

Pure-Python tests with synthetic IMUReading samples; no hardware required.
"""

import math

import numpy as np
import pytest

from lidar_mapping.sensors.imu import IMUReading
from lidar_mapping.mapping.imu_preintegrator import IMUPreintegrator, _skew


def _reading(t: float, gyro=(0.0, 0.0, 0.0), accel=(0.0, 0.0, 9.80665)) -> IMUReading:
    return IMUReading(
        timestamp=float(t),
        accel_mss=np.array(accel, dtype=np.float64),
        gyro_rads=np.array(gyro, dtype=np.float64),
    )


def _constant_gyro_stream(axis, omega_rads, duration_s, rate_hz=200, t0=0.0):
    """Generate readings with constant angular velocity along `axis`."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    gyro = (axis * omega_rads).tolist()
    n = int(duration_s * rate_hz) + 1
    dt = 1.0 / rate_hz
    return [_reading(t0 + i * dt, gyro=gyro) for i in range(n)]


# ---------------------------------------------------------------------------
# Buffer behaviour
# ---------------------------------------------------------------------------

class TestBuffer:
    def test_initial_state(self):
        p = IMUPreintegrator()
        assert p.buffered_count == 0

    def test_push_increments_count(self):
        p = IMUPreintegrator()
        p.push(_reading(0.0))
        p.push(_reading(0.005))
        assert p.buffered_count == 2

    def test_consume_clears_buffer(self):
        p = IMUPreintegrator()
        for r in _constant_gyro_stream([0, 0, 1], 0.5, 0.1):
            p.push(r)
        p.consume()
        assert p.buffered_count == 0

    def test_peek_preserves_buffer(self):
        p = IMUPreintegrator()
        for r in _constant_gyro_stream([0, 0, 1], 0.5, 0.1):
            p.push(r)
        n = p.buffered_count
        T = p.peek()
        assert p.buffered_count == n
        assert T.shape == (4, 4)

    def test_reset_clears_buffer(self):
        p = IMUPreintegrator()
        p.push(_reading(0.0))
        p.push(_reading(0.005))
        p.reset()
        assert p.buffered_count == 0

    def test_max_readings_capped(self):
        p = IMUPreintegrator(max_readings=10)
        for i in range(50):
            p.push(_reading(i * 0.005))
        assert p.buffered_count == 10


# ---------------------------------------------------------------------------
# Consume edge cases
# ---------------------------------------------------------------------------

class TestConsumeEdgeCases:
    def test_empty_returns_identity(self):
        p = IMUPreintegrator()
        T = p.consume()
        assert np.allclose(T, np.eye(4))

    def test_below_min_readings_returns_identity(self):
        p = IMUPreintegrator(min_readings=5)
        p.push(_reading(0.0, gyro=(0, 0, 1.0)))
        p.push(_reading(0.005, gyro=(0, 0, 1.0)))
        T = p.consume()
        assert np.allclose(T, np.eye(4))

    def test_translation_is_zero(self):
        p = IMUPreintegrator()
        for r in _constant_gyro_stream([0, 0, 1], 0.5, 0.1):
            p.push(r)
        T = p.consume()
        assert np.allclose(T[:3, 3], 0.0)
        assert T[3, 3] == 1.0


# ---------------------------------------------------------------------------
# Quaternion integration
# ---------------------------------------------------------------------------

class TestQuaternionIntegration:
    def test_no_motion_returns_identity_rotation(self):
        p = IMUPreintegrator(use_quaternion_integration=True, gyro_noise_threshold=0.0)
        for t in np.arange(0, 1.0, 0.005):
            p.push(_reading(t, gyro=(0.0, 0.0, 0.0)))
        T = p.consume()
        assert np.allclose(T[:3, :3], np.eye(3), atol=1e-6)

    def test_rotation_about_z(self):
        """Integrate 1 rad/s about Z for 1 s → ~57.3° rotation."""
        p = IMUPreintegrator(use_quaternion_integration=True)
        for r in _constant_gyro_stream([0, 0, 1], 1.0, 1.0, rate_hz=500):
            p.push(r)
        T = p.consume()
        R = T[:3, :3]
        # Expected R = Rz(1 rad)
        c, s = math.cos(1.0), math.sin(1.0)
        expected = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        assert np.allclose(R, expected, atol=0.02)

    def test_rotation_about_x(self):
        p = IMUPreintegrator(use_quaternion_integration=True)
        for r in _constant_gyro_stream([1, 0, 0], 0.5, 1.0, rate_hz=500):
            p.push(r)
        T = p.consume()
        R = T[:3, :3]
        c, s = math.cos(0.5), math.sin(0.5)
        expected = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        assert np.allclose(R, expected, atol=0.02)

    def test_rotation_is_proper_orthogonal(self):
        p = IMUPreintegrator(use_quaternion_integration=True)
        for r in _constant_gyro_stream([1, 1, 1], 2.0, 1.0, rate_hz=500):
            p.push(r)
        R = p.consume()[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-4)
        assert math.isclose(float(np.linalg.det(R)), 1.0, abs_tol=1e-4)

    def test_noise_threshold_suppresses_small_gyro(self):
        """Gyro below threshold should not produce rotation."""
        p = IMUPreintegrator(
            use_quaternion_integration=True, gyro_noise_threshold=0.01
        )
        # Magnitude 0.001 << 0.01 threshold
        for r in _constant_gyro_stream([0, 0, 1], 0.001, 1.0, rate_hz=200):
            p.push(r)
        T = p.consume()
        assert np.allclose(T[:3, :3], np.eye(3), atol=1e-6)

    def test_bad_dt_skipped(self):
        """Readings with dt <= 0 or > 1.0 are skipped."""
        p = IMUPreintegrator(use_quaternion_integration=True, gyro_noise_threshold=0.0)
        p.push(_reading(0.0, gyro=(0, 0, 1.0)))
        p.push(_reading(0.0, gyro=(0, 0, 1.0)))  # dt = 0
        p.push(_reading(100.0, gyro=(0, 0, 1.0)))  # dt = 100
        T = p.consume()
        assert np.allclose(T[:3, :3], np.eye(3), atol=1e-6)


# ---------------------------------------------------------------------------
# Matrix integration
# ---------------------------------------------------------------------------

class TestMatrixIntegration:
    def test_no_motion_identity(self):
        # Use default threshold (0.005) — pure-zero gyro will be skipped
        p = IMUPreintegrator(use_quaternion_integration=False)
        for t in np.arange(0, 1.0, 0.005):
            p.push(_reading(t))
        R = p.consume()[:3, :3]
        assert np.allclose(R, np.eye(3), atol=1e-6)

    def test_small_rotation_about_z(self):
        p = IMUPreintegrator(use_quaternion_integration=False)
        for r in _constant_gyro_stream([0, 0, 1], 0.1, 1.0, rate_hz=500):
            p.push(r)
        R = p.consume()[:3, :3]
        c, s = math.cos(0.1), math.sin(0.1)
        expected = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        assert np.allclose(R, expected, atol=0.01)

    def test_matrix_remains_orthogonal(self):
        """SVD re-orthogonalisation keeps R proper-orthogonal."""
        p = IMUPreintegrator(use_quaternion_integration=False)
        for r in _constant_gyro_stream([1, 1, 0], 1.0, 1.0, rate_hz=500):
            p.push(r)
        R = p.consume()[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
        assert math.isclose(float(np.linalg.det(R)), 1.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Tilt helper
# ---------------------------------------------------------------------------

class TestTiltFromAccel:
    def test_level_sensor(self):
        roll, pitch = IMUPreintegrator.tilt_from_accel(np.array([0, 0, 9.81]))
        assert math.isclose(roll, 0.0, abs_tol=1e-6)
        assert math.isclose(pitch, 0.0, abs_tol=1e-6)

    def test_pitch_forward_90deg(self):
        # +X tipped down → all gravity along -X
        roll, pitch = IMUPreintegrator.tilt_from_accel(np.array([-9.81, 0, 0]))
        assert math.isclose(roll, 0.0, abs_tol=1e-6)
        assert math.isclose(pitch, math.pi / 2, abs_tol=1e-4)

    def test_roll_right_90deg(self):
        # +Y tipped down → all gravity along +Y
        roll, pitch = IMUPreintegrator.tilt_from_accel(np.array([0, 9.81, 0]))
        assert math.isclose(roll, math.pi / 2, abs_tol=1e-4)
        assert math.isclose(pitch, 0.0, abs_tol=1e-4)


# ---------------------------------------------------------------------------
# Skew helper
# ---------------------------------------------------------------------------

class TestSkew:
    def test_antisymmetric(self):
        v = np.array([1.0, 2.0, 3.0])
        K = _skew(v)
        assert np.allclose(K, -K.T)

    def test_skew_times_v_zero(self):
        v = np.array([1.0, 2.0, 3.0])
        assert np.allclose(_skew(v) @ v, np.zeros(3))

    def test_cross_product_identity(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert np.allclose(_skew(a) @ b, np.cross(a, b))
