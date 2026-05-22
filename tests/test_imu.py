"""
Unit tests for IMU-related modules:
  - lidar_mapping.sensors.ahrs    (MadgwickAHRS, MahonyAHRS)
  - lidar_mapping.sensors.imu     (IMUReading, SerialAHRSDriver CSV parsing)
  - lidar_mapping.mapping.imu_preintegrator  (IMUPreintegrator)
  - Integration: Mapper with IMUPreintegrator transform hint
"""

from __future__ import annotations

import math
import time
import types
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# AHRS filter tests
# ---------------------------------------------------------------------------

from lidar_mapping.sensors.ahrs import (
    MadgwickAHRS,
    MahonyAHRS,
    _quaternion_to_euler_degrees,
)


class TestMadgwickAHRS:
    """Tests for the Madgwick AHRS filter."""

    def test_initial_quaternion_is_identity(self):
        ahrs = MadgwickAHRS(sample_rate=100.0)
        q = ahrs.quaternion
        assert q.shape == (4,)
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-9)

    def test_initial_euler_near_zero(self):
        ahrs = MadgwickAHRS(sample_rate=100.0)
        roll, pitch, yaw = ahrs.euler_degrees
        assert abs(roll) < 1e-9
        assert abs(pitch) < 1e-9
        assert abs(yaw) < 1e-9

    def test_quaternion_remains_unit_after_updates(self):
        """After many 6-DOF updates the quaternion must stay normalised."""
        ahrs = MadgwickAHRS(sample_rate=100.0, beta=0.1)
        gyro = np.array([0.1, 0.05, -0.02])
        accel = np.array([0.0, 0.0, 9.81])

        for _ in range(500):
            ahrs.update_imu(gyro, accel)

        q = ahrs.quaternion
        norm = float(np.linalg.norm(q))
        assert abs(norm - 1.0) < 1e-6

    def test_9dof_update_keeps_unit_quaternion(self):
        ahrs = MadgwickAHRS(sample_rate=100.0)
        gyro = np.array([0.05, -0.03, 0.08])
        accel = np.array([0.1, 0.0, 9.81])
        mag = np.array([25.0, 5.0, -40.0])

        for _ in range(200):
            ahrs.update(gyro, accel, mag)

        norm = float(np.linalg.norm(ahrs.quaternion))
        assert abs(norm - 1.0) < 1e-6

    def test_reset_returns_to_identity(self):
        ahrs = MadgwickAHRS(sample_rate=100.0)
        gyro = np.array([0.3, 0.1, 0.0])
        accel = np.array([0.0, 0.0, 9.81])
        for _ in range(100):
            ahrs.update_imu(gyro, accel)
        # Should have drifted from identity
        ahrs.reset()
        np.testing.assert_allclose(ahrs.quaternion, [1.0, 0.0, 0.0, 0.0], atol=1e-9)

    def test_zero_accel_falls_back_to_gyro_integration(self):
        """update_imu with near-zero accel should not crash."""
        ahrs = MadgwickAHRS(sample_rate=100.0)
        gyro = np.array([0.1, 0.0, 0.0])
        accel = np.zeros(3)  # pathological: zero gravity vector
        # Should not raise
        ahrs.update_imu(gyro, accel)

    def test_zero_mag_falls_back_to_6dof(self):
        """9-DOF update with near-zero magnetometer should not crash."""
        ahrs = MadgwickAHRS(sample_rate=100.0)
        gyro = np.array([0.0, 0.1, 0.0])
        accel = np.array([0.0, 0.0, 9.81])
        mag = np.zeros(3)
        ahrs.update(gyro, accel, mag)  # should not raise

    def test_euler_degrees_range(self):
        """Euler angles should be in the correct range after arbitrary motion."""
        ahrs = MadgwickAHRS(sample_rate=100.0, beta=0.1)
        rng = np.random.default_rng(42)
        for _ in range(300):
            gyro = rng.uniform(-1.0, 1.0, 3)
            accel = np.array([0.0, 0.0, 9.81])
            mag = rng.uniform(-60.0, 60.0, 3)
            ahrs.update(gyro, accel, mag)

        roll, pitch, yaw = ahrs.euler_degrees
        assert -180.0 <= roll <= 180.0
        assert -90.0 <= pitch <= 90.0
        assert -180.0 <= yaw <= 180.0

    def test_gravity_alignment_converges(self):
        """
        Starting from identity, feeding 'flat' gravity should converge to
        near-zero roll and pitch.
        """
        ahrs = MadgwickAHRS(sample_rate=100.0, beta=0.5)
        gyro = np.zeros(3)
        accel = np.array([0.0, 0.0, 9.81])
        mag = np.array([25.0, 0.0, -40.0])

        for _ in range(2000):
            ahrs.update(gyro, accel, mag)

        roll, pitch, _ = ahrs.euler_degrees
        assert abs(roll) < 2.0, f"Expected near-zero roll, got {roll:.2f}°"
        assert abs(pitch) < 2.0, f"Expected near-zero pitch, got {pitch:.2f}°"


class TestMahonyAHRS:
    """Tests for the Mahony AHRS filter."""

    def test_initial_quaternion_is_identity(self):
        ahrs = MahonyAHRS(sample_rate=100.0)
        np.testing.assert_allclose(ahrs.quaternion, [1.0, 0.0, 0.0, 0.0], atol=1e-9)

    def test_unit_quaternion_preserved(self):
        ahrs = MahonyAHRS(sample_rate=100.0, kp=2.0, ki=0.005)
        gyro = np.array([0.2, -0.1, 0.05])
        accel = np.array([0.0, 0.0, 9.81])
        mag = np.array([20.0, 5.0, -45.0])

        for _ in range(300):
            ahrs.update(gyro, accel, mag)

        norm = float(np.linalg.norm(ahrs.quaternion))
        assert abs(norm - 1.0) < 1e-6

    def test_6dof_mode(self):
        ahrs = MahonyAHRS(sample_rate=100.0)
        gyro = np.array([0.05, 0.05, 0.05])
        accel = np.array([0.0, 0.0, 9.81])
        for _ in range(200):
            ahrs.update_imu(gyro, accel)
        norm = float(np.linalg.norm(ahrs.quaternion))
        assert abs(norm - 1.0) < 1e-6

    def test_reset(self):
        ahrs = MahonyAHRS(sample_rate=100.0)
        for _ in range(100):
            ahrs.update_imu(np.array([0.5, 0.0, 0.0]), np.array([0.0, 0.0, 9.81]))
        ahrs.reset()
        np.testing.assert_allclose(ahrs.quaternion, [1.0, 0.0, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(ahrs._integral_fb, [0.0, 0.0, 0.0], atol=1e-9)


class TestQuaternionToEuler:
    """Tests for the shared _quaternion_to_euler_degrees helper."""

    def test_identity_gives_zero_euler(self):
        r, p, y = _quaternion_to_euler_degrees(np.array([1.0, 0.0, 0.0, 0.0]))
        assert abs(r) < 1e-9
        assert abs(p) < 1e-9
        assert abs(y) < 1e-9

    def test_90deg_yaw(self):
        # Quaternion for 90° yaw: q = [cos(45°), 0, 0, sin(45°)]
        angle = math.radians(90.0)
        q = np.array([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])
        _, _, yaw = _quaternion_to_euler_degrees(q)
        assert abs(yaw - 90.0) < 0.001

    def test_45deg_roll(self):
        angle = math.radians(45.0)
        q = np.array([math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0])
        roll, _, _ = _quaternion_to_euler_degrees(q)
        assert abs(roll - 45.0) < 0.001

    def test_gimbal_lock_does_not_raise(self):
        # Pitch = 90° is the gimbal-lock singularity
        angle = math.radians(90.0)
        q = np.array([math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0])
        r, p, y = _quaternion_to_euler_degrees(q)
        # Should not raise; pitch should be ~90°
        assert abs(p - 90.0) < 0.5


# ---------------------------------------------------------------------------
# IMU reading dataclass
# ---------------------------------------------------------------------------

from lidar_mapping.sensors.imu import IMUReading


class TestIMUReading:
    def test_default_fields(self):
        r = IMUReading(
            timestamp=1.0,
            accel_mss=np.array([0.0, 0.0, 9.81]),
            gyro_rads=np.array([0.01, -0.02, 0.005]),
        )
        assert r.mag_ut is None
        assert r.temperature_c is None
        assert r.roll_deg == 0.0
        assert r.yaw_deg == 0.0
        np.testing.assert_array_equal(r.quaternion, [1.0, 0.0, 0.0, 0.0])

    def test_full_fields(self):
        q = np.array([0.9998477, 0.0174524, 0.0, 0.0])
        r = IMUReading(
            timestamp=42.5,
            accel_mss=np.array([0.1, 0.0, 9.81]),
            gyro_rads=np.array([0.0, 0.0, 0.0]),
            mag_ut=np.array([25.0, 0.0, -40.0]),
            temperature_c=22.5,
            roll_deg=2.0,
            pitch_deg=0.5,
            yaw_deg=10.0,
            quaternion=q,
        )
        assert r.temperature_c == 22.5
        assert r.roll_deg == 2.0
        np.testing.assert_array_equal(r.mag_ut, [25.0, 0.0, -40.0])


# ---------------------------------------------------------------------------
# SerialAHRSDriver — CSV and PASHR parsers (no hardware required)
# ---------------------------------------------------------------------------

from lidar_mapping.sensors.imu import SerialAHRSDriver


class TestSerialAHRSParsers:
    """Test the static CSV / PASHR parser methods without serial hardware."""

    def test_csv_6dof(self):
        line = "1.0,0.0,9.81,0.01,-0.02,0.005"
        accel, gyro, mag, temp = SerialAHRSDriver._parse_csv(line)
        np.testing.assert_allclose(accel, [1.0, 0.0, 9.81], atol=1e-6)
        np.testing.assert_allclose(gyro, [0.01, -0.02, 0.005], atol=1e-6)
        assert mag is None
        assert temp is None

    def test_csv_9dof(self):
        line = "0.1,0.2,9.8,0.01,0.02,0.03,25.0,5.0,-40.0"
        accel, gyro, mag, temp = SerialAHRSDriver._parse_csv(line)
        np.testing.assert_allclose(mag, [25.0, 5.0, -40.0], atol=1e-6)

    def test_csv_whitespace_tolerant(self):
        line = " 0.0 , 0.0 , 9.81 , 0.0 , 0.0 , 0.0 "
        accel, gyro, mag, temp = SerialAHRSDriver._parse_csv(line)
        np.testing.assert_allclose(accel[2], 9.81, atol=1e-6)

    def test_csv_too_short_raises(self):
        with pytest.raises(ValueError, match="too short"):
            SerialAHRSDriver._parse_csv("1.0,2.0,3.0")

    def test_pashr_valid(self):
        line = "$PASHR,045.00,2.50,-1.00,0.00,0.10,-0.20,0.05,1*00"
        accel, gyro, mag, temp = SerialAHRSDriver._parse_pashr(line)
        # Returns zeros for raw sensors
        np.testing.assert_array_equal(accel, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(gyro, [0.0, 0.0, 0.0])
        assert mag is None

    def test_pashr_invalid_raises(self):
        with pytest.raises(ValueError, match="Not a PASHR"):
            SerialAHRSDriver._parse_pashr("$GPRMC,123456,A,...")

    def test_pashr_too_few_fields_raises(self):
        with pytest.raises(ValueError, match="Too few fields"):
            SerialAHRSDriver._parse_pashr("$PASHR,045.00*00")

    def test_csv_negative_values(self):
        line = "-1.5,2.3,-9.81,-0.1,0.2,-0.05"
        accel, gyro, mag, temp = SerialAHRSDriver._parse_csv(line)
        np.testing.assert_allclose(accel[0], -1.5, atol=1e-6)
        np.testing.assert_allclose(accel[2], -9.81, atol=1e-6)


# ---------------------------------------------------------------------------
# IMUPreintegrator tests
# ---------------------------------------------------------------------------

from lidar_mapping.mapping.imu_preintegrator import IMUPreintegrator, _skew


def _make_reading(
    timestamp: float,
    gyro: np.ndarray,
    accel: Optional[np.ndarray] = None,
) -> IMUReading:
    if accel is None:
        accel = np.array([0.0, 0.0, 9.81])
    return IMUReading(
        timestamp=timestamp,
        accel_mss=accel,
        gyro_rads=gyro,
    )


class TestIMUPreintegrator:
    def test_consume_empty_returns_identity(self):
        pi = IMUPreintegrator()
        T = pi.consume()
        np.testing.assert_allclose(T, np.eye(4), atol=1e-10)

    def test_consume_too_few_returns_identity(self):
        pi = IMUPreintegrator(min_readings=3)
        pi.push(_make_reading(0.0, np.zeros(3)))
        T = pi.consume()
        np.testing.assert_allclose(T, np.eye(4), atol=1e-10)

    def test_consume_clears_buffer(self):
        pi = IMUPreintegrator()
        for i in range(10):
            pi.push(_make_reading(i * 0.01, np.array([0.1, 0.0, 0.0])))
        pi.consume()
        assert pi.buffered_count == 0

    def test_peek_does_not_clear_buffer(self):
        pi = IMUPreintegrator()
        for i in range(10):
            pi.push(_make_reading(i * 0.01, np.array([0.1, 0.0, 0.0])))
        pi.peek()
        assert pi.buffered_count == 10

    def test_identity_when_gyro_below_noise_threshold(self):
        """Sub-threshold gyro readings should produce identity."""
        pi = IMUPreintegrator(gyro_noise_threshold=0.01)
        for i in range(20):
            pi.push(_make_reading(i * 0.01, np.array([0.001, 0.001, 0.001])))
        T = pi.consume()
        np.testing.assert_allclose(T, np.eye(4), atol=1e-6)

    def test_rotation_is_valid_for_pure_yaw(self):
        """A constant yaw-rate should produce a valid rotation matrix."""
        pi = IMUPreintegrator(gyro_noise_threshold=0.0)
        omega_z = 1.0  # rad/s yaw
        dt = 0.01
        for i in range(30):
            pi.push(_make_reading(i * dt, np.array([0.0, 0.0, omega_z])))

        T = pi.consume()
        R = T[:3, :3]

        # R should be orthogonal
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
        # Determinant should be +1 (proper rotation)
        assert abs(np.linalg.det(R) - 1.0) < 1e-6

    def test_rotation_is_valid_for_pure_yaw_matrix_method(self):
        """Same test using the matrix-integration path."""
        pi = IMUPreintegrator(
            use_quaternion_integration=False, gyro_noise_threshold=0.0
        )
        omega_z = 0.5
        dt = 0.01
        for i in range(30):
            pi.push(_make_reading(i * dt, np.array([0.0, 0.0, omega_z])))

        T = pi.consume()
        R = T[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-5)
        assert abs(np.linalg.det(R) - 1.0) < 1e-5

    def test_transform_hint_angle_matches_expected(self):
        """Integrating a known constant yaw-rate should match analytical result."""
        omega_z = math.pi / 2.0  # 90°/s
        dt = 0.01
        n = 100  # 1 second → 90°
        pi = IMUPreintegrator(gyro_noise_threshold=0.0)

        for i in range(n):
            pi.push(_make_reading(i * dt, np.array([0.0, 0.0, omega_z])))

        T = pi.consume()
        R = T[:3, :3]

        # Extract yaw angle from rotation matrix
        yaw_rad = math.atan2(R[1, 0], R[0, 0])
        yaw_deg = math.degrees(yaw_rad)

        # Expect ~90° with some tolerance due to numerical integration
        assert abs(yaw_deg - 90.0) < 5.0, f"Expected ~90°, got {yaw_deg:.2f}°"

    def test_reset(self):
        pi = IMUPreintegrator()
        for i in range(20):
            pi.push(_make_reading(i * 0.01, np.array([0.5, 0.0, 0.0])))
        pi.reset()
        assert pi.buffered_count == 0

    def test_max_queue_respected(self):
        pi = IMUPreintegrator(max_readings=10)
        for i in range(50):
            pi.push(_make_reading(i * 0.01, np.zeros(3)))
        assert pi.buffered_count == 10

    def test_bad_timestamps_skipped(self):
        """Readings with zero or negative dt should not corrupt the result."""
        pi = IMUPreintegrator(gyro_noise_threshold=0.0)
        # Insert readings with non-monotonic timestamps
        pi.push(_make_reading(0.0, np.array([0.0, 0.0, 1.0])))
        pi.push(_make_reading(0.0, np.array([0.0, 0.0, 1.0])))  # Δt=0
        pi.push(_make_reading(-0.01, np.array([0.0, 0.0, 1.0])))  # negative Δt
        pi.push(_make_reading(0.02, np.array([0.0, 0.0, 1.0])))
        # Should not raise
        T = pi.consume()
        R = T[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-5)

    def test_tilt_from_accel_flat(self):
        roll, pitch = IMUPreintegrator.tilt_from_accel(np.array([0.0, 0.0, 9.81]))
        assert abs(roll) < 1e-6
        assert abs(pitch) < 1e-6

    def test_tilt_from_accel_tilted(self):
        # 45° roll: ax=0, ay=g/√2, az=g/√2
        g = 9.81
        accel = np.array([0.0, g / math.sqrt(2), g / math.sqrt(2)])
        roll, pitch = IMUPreintegrator.tilt_from_accel(accel)
        assert abs(math.degrees(roll) - 45.0) < 0.1


# ---------------------------------------------------------------------------
# Skew matrix helper
# ---------------------------------------------------------------------------

class TestSkewMatrix:
    def test_skew_antisymmetric(self):
        v = np.array([1.0, 2.0, 3.0])
        K = _skew(v)
        np.testing.assert_allclose(K + K.T, np.zeros((3, 3)), atol=1e-10)

    def test_skew_cross_product(self):
        """K @ u should equal cross(v, u) for any u."""
        v = np.array([1.0, 2.0, 3.0])
        u = np.array([4.0, 5.0, 6.0])
        K = _skew(v)
        np.testing.assert_allclose(K @ u, np.cross(v, u), atol=1e-10)


# ---------------------------------------------------------------------------
# Integration test: Mapper uses IMUPreintegrator hint
# ---------------------------------------------------------------------------

from lidar_mapping.mapping.mapper import Mapper
from lidar_mapping.processing.registration import RegistrationResult


def _make_mock_icp():
    """Return a fake ICPRegistration that always reports perfect convergence."""
    mock_icp = MagicMock()
    mock_icp.register.return_value = RegistrationResult(
        transform=np.eye(4, dtype=np.float64),
        fitness=1.0,
        inlier_rmse=0.0,
        converged=True,
    )
    return mock_icp


def _identity_filter(pts, *args, **kwargs):
    return pts


def _noop_ground_removal(pts, *args, **kwargs):
    return pts, pts[:0], np.array([0, 0, 1, 0])


class TestMapperWithIMU:
    """Verify that Mapper correctly consumes IMU hints via preintegrator."""

    @staticmethod
    def _make_cloud(n: int = 500, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.uniform(-5.0, 5.0, (n, 3)).astype(np.float32)

    @staticmethod
    def _make_mapper(**kwargs):
        """Create a Mapper with ICP mocked out."""
        mapper = Mapper(voxel_size=0.5, **kwargs)
        mapper._icp = _make_mock_icp()
        return mapper

    def test_mapper_accepts_imu_preintegrator(self):
        """Mapper should initialise without error when a preintegrator is given."""
        preint = IMUPreintegrator()
        mapper = self._make_mapper(imu_preintegrator=preint)
        assert mapper._preintegrator is preint

    @patch("lidar_mapping.mapping.mapper.voxel_downsample", side_effect=_identity_filter)
    @patch("lidar_mapping.mapping.mapper.range_filter", side_effect=_identity_filter)
    @patch("lidar_mapping.mapping.mapper.passthrough_filter", side_effect=_identity_filter)
    def test_first_scan_no_imu_readings(self, *_patches):
        """First scan (no prior) should succeed even with empty preintegrator."""
        preint = IMUPreintegrator()
        mapper = self._make_mapper(imu_preintegrator=preint)
        mapper.add_scan(self._make_cloud())
        assert mapper.scans_processed == 1

    @patch("lidar_mapping.mapping.mapper.voxel_downsample", side_effect=_identity_filter)
    @patch("lidar_mapping.mapping.mapper.range_filter", side_effect=_identity_filter)
    @patch("lidar_mapping.mapping.mapper.passthrough_filter", side_effect=_identity_filter)
    def test_preintegrator_consumed_on_add_scan(self, *_patches):
        """Preintegrator buffer should be drained when add_scan is called."""
        preint = IMUPreintegrator()
        for i in range(20):
            preint.push(_make_reading(i * 0.01, np.array([0.0, 0.0, 0.5])))
        assert preint.buffered_count == 20

        mapper = self._make_mapper(imu_preintegrator=preint)
        # First scan initialises the map; preintegrator is consumed
        mapper.add_scan(self._make_cloud(seed=0))
        assert preint.buffered_count == 0

    @patch("lidar_mapping.mapping.mapper.voxel_downsample", side_effect=_identity_filter)
    @patch("lidar_mapping.mapping.mapper.range_filter", side_effect=_identity_filter)
    @patch("lidar_mapping.mapping.mapper.passthrough_filter", side_effect=_identity_filter)
    def test_explicit_hint_overrides_imu(self, *_patches):
        """An explicit transform_hint passed to add_scan must be used instead of IMU."""
        preint = IMUPreintegrator()
        mapper = self._make_mapper(imu_preintegrator=preint)
        mapper.add_scan(self._make_cloud(seed=0))  # init map

        # Load the preintegrator with a big yaw motion
        for i in range(10):
            preint.push(_make_reading(i * 0.01, np.array([0.0, 0.0, 2.0])))

        explicit_hint = np.eye(4, dtype=np.float64)
        explicit_hint[0, 3] = 0.5  # small translation hint

        # Capture what ICP receives
        received_hints = []
        original_register = mapper._icp.register.side_effect

        def recording_register(source, target, initial_transform=None):
            received_hints.append(
                initial_transform.copy() if initial_transform is not None else None
            )
            return mapper._icp.register.return_value

        mapper._icp.register = MagicMock(side_effect=recording_register)
        mapper._icp.register.return_value = RegistrationResult(
            transform=np.eye(4), fitness=1.0, inlier_rmse=0.0, converged=True
        )

        mapper.add_scan(self._make_cloud(seed=1), transform_hint=explicit_hint)

        assert len(received_hints) == 1
        np.testing.assert_allclose(received_hints[0], explicit_hint, atol=1e-10)

    def test_mapper_without_imu_still_works(self):
        """Mapper with no preintegrator should have no preintegrator set."""
        mapper = self._make_mapper()
        assert mapper._preintegrator is None
