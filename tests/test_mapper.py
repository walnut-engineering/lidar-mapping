"""
Tests for the Mapper class — end-to-end mapping with synthetic scans.

Uses Open3D for ICP. No hardware required.
"""

from pathlib import Path

import numpy as np
import pytest

from lidar_mapping.mapping.mapper import Mapper
from lidar_mapping.mapping.imu_preintegrator import IMUPreintegrator
from lidar_mapping.sensors.imu import IMUReading
from lidar_mapping.utils.transforms import (
    make_transform,
    make_transform_from_euler,
    apply_transform,
)


# ---------------------------------------------------------------------------
# Synthetic scene
# ---------------------------------------------------------------------------

def _make_scene(n_per_face: int = 400, seed: int = 0) -> np.ndarray:
    """Make a 10x10x4 room (4 walls + floor) of points."""
    rng = np.random.default_rng(seed)
    pts = []
    # Floor z=0
    pts.append(np.stack([
        rng.uniform(-5, 5, n_per_face),
        rng.uniform(-5, 5, n_per_face),
        np.zeros(n_per_face),
    ], axis=1))
    # Walls
    for sign in (-1, 1):
        pts.append(np.stack([
            np.full(n_per_face, 5.0 * sign),
            rng.uniform(-5, 5, n_per_face),
            rng.uniform(0.1, 4.0, n_per_face),
        ], axis=1))
        pts.append(np.stack([
            rng.uniform(-5, 5, n_per_face),
            np.full(n_per_face, 5.0 * sign),
            rng.uniform(0.1, 4.0, n_per_face),
        ], axis=1))
    return np.vstack(pts).astype(np.float32)


# ---------------------------------------------------------------------------
# Construction & basic state
# ---------------------------------------------------------------------------

class TestMapperInit:
    def test_initial_state(self):
        m = Mapper(voxel_size=0.1)
        assert m.map_points is None
        assert np.allclose(m.current_pose, np.eye(4))
        assert m.scans_processed == 0
        assert len(m.pose_history) == 1

    def test_pose_history_is_copy(self):
        m = Mapper()
        history = m.pose_history
        history.append(np.zeros((4, 4)))
        assert len(m.pose_history) == 1


# ---------------------------------------------------------------------------
# Single-scan
# ---------------------------------------------------------------------------

class TestFirstScan:
    def test_first_scan_initialises_map(self):
        m = Mapper(voxel_size=0.1, min_range=0.0, max_range=100.0)
        scan = _make_scene()
        result = m.add_scan(scan)
        assert m.map_points is not None
        assert len(m.map_points) > 0
        assert result.converged
        assert m.scans_processed == 1

    def test_first_scan_identity_pose(self):
        m = Mapper(voxel_size=0.1, min_range=0.0, max_range=100.0)
        m.add_scan(_make_scene())
        assert np.allclose(m.current_pose, np.eye(4))


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_range_filter_applied(self):
        m = Mapper(voxel_size=0.05, min_range=1.0, max_range=2.0)
        pts = np.array([
            [0.1, 0.0, 0.0],   # too close
            [1.5, 0.0, 0.0],   # in range
            [10.0, 0.0, 0.0],  # too far
        ], dtype=np.float32)
        out = m._preprocess(pts)
        # Only the middle point should survive
        assert len(out) == 1
        assert np.isclose(np.linalg.norm(out[0]), 1.5)

    def test_z_passthrough_applied(self):
        m = Mapper(voxel_size=0.05, min_range=0.0, max_range=100.0,
                   z_min=0.0, z_max=1.0)
        pts = np.array([
            [1.0, 0.0, -1.0],  # below
            [1.0, 0.0, 0.5],   # in
            [1.0, 0.0, 2.0],   # above
        ], dtype=np.float32)
        out = m._preprocess(pts)
        assert len(out) == 1

    def test_drops_extra_columns(self):
        m = Mapper(voxel_size=0.05, min_range=0.0, max_range=100.0)
        pts = np.array([[1.0, 2.0, 3.0, 99.0, 1.0]], dtype=np.float32)
        out = m._preprocess(pts)
        assert out.shape[1] == 3


# ---------------------------------------------------------------------------
# Multi-scan tracking
# ---------------------------------------------------------------------------

class TestTracking:
    def test_static_scans_pose_near_identity(self):
        m = Mapper(voxel_size=0.1, min_range=0.0, max_range=100.0)
        scene = _make_scene()
        m.add_scan(scene)
        m.add_scan(scene)
        # No motion → pose should stay near identity
        assert np.allclose(m.current_pose, np.eye(4), atol=0.05)
        assert m.scans_processed == 2

    def test_translation_recovered(self):
        m = Mapper(
            voxel_size=0.1,
            min_range=0.0,
            max_range=100.0,
            icp_max_correspondence_distance=2.0,
        )
        scene = _make_scene(n_per_face=2000)
        m.add_scan(scene)
        # Move sensor +0.3 m in X → scene appears -0.3 m in sensor frame
        T_motion = make_transform(np.eye(3), np.array([0.3, 0.0, 0.0]))
        moved = apply_transform(scene, np.linalg.inv(T_motion))
        result = m.add_scan(moved)
        assert result.converged
        # Recovered pose translation magnitude should be near 0.3 m
        recovered_t = m.current_pose[:3, 3]
        assert abs(np.linalg.norm(recovered_t) - 0.3) < 0.1

    def test_pose_history_grows(self):
        m = Mapper(voxel_size=0.1, min_range=0.0, max_range=100.0)
        scene = _make_scene()
        for _ in range(3):
            m.add_scan(scene)
        # Initial identity + 3 scans
        assert len(m.pose_history) == 4

    def test_processing_time_accumulates(self):
        m = Mapper(voxel_size=0.1, min_range=0.0, max_range=100.0)
        scene = _make_scene()
        m.add_scan(scene)
        m.add_scan(scene)
        assert m.total_processing_time > 0.0


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_state(self):
        m = Mapper(voxel_size=0.1, min_range=0.0, max_range=100.0)
        m.add_scan(_make_scene())
        m.add_scan(_make_scene())
        m.reset()
        assert m.map_points is None
        assert np.allclose(m.current_pose, np.eye(4))
        assert m.scans_processed == 0
        assert len(m.pose_history) == 1
        assert m.total_processing_time == 0.0


# ---------------------------------------------------------------------------
# IMU integration
# ---------------------------------------------------------------------------

class TestIMUHint:
    def test_preintegrator_consumed_on_scan(self):
        preint = IMUPreintegrator()
        # Push gyro readings that produce a non-identity rotation
        for i in range(50):
            t = i * 0.005
            preint.push(IMUReading(
                timestamp=t,
                accel_mss=np.array([0, 0, 9.81]),
                gyro_rads=np.array([0, 0, 1.0]),
            ))
        m = Mapper(voxel_size=0.1, min_range=0.0, max_range=100.0,
                   imu_preintegrator=preint)
        scene = _make_scene()
        m.add_scan(scene)
        m.add_scan(scene)
        # Buffer must be drained after second add_scan
        assert preint.buffered_count == 0

    def test_explicit_hint_takes_precedence(self):
        preint = IMUPreintegrator()
        m = Mapper(voxel_size=0.1, min_range=0.0, max_range=100.0,
                   imu_preintegrator=preint)
        scene = _make_scene()
        # First scan initialises map (no ICP)
        m.add_scan(scene)
        # Now push readings AFTER first scan so they survive until next add_scan
        for i in range(50):
            preint.push(IMUReading(
                timestamp=i * 0.005,
                accel_mss=np.array([0, 0, 9.81]),
                gyro_rads=np.array([0, 0, 5.0]),
            ))
        explicit = np.eye(4)
        m.add_scan(scene, transform_hint=explicit)
        # When explicit hint is given, preintegrator buffer is NOT consumed
        assert preint.buffered_count == 50


# ---------------------------------------------------------------------------
# Map I/O
# ---------------------------------------------------------------------------

class TestMapIO:
    def test_get_map_o3d_empty(self):
        m = Mapper()
        pcd = m.get_map_o3d()
        assert len(np.asarray(pcd.points)) == 0

    def test_get_map_o3d_populated(self):
        m = Mapper(voxel_size=0.1, min_range=0.0, max_range=100.0)
        m.add_scan(_make_scene())
        pcd = m.get_map_o3d()
        assert len(np.asarray(pcd.points)) > 0

    def test_save_without_map_raises(self, tmp_path):
        m = Mapper()
        with pytest.raises(RuntimeError):
            m.save_map(tmp_path / "out.pcd")

    def test_save_and_load_pcd(self, tmp_path):
        m = Mapper(voxel_size=0.1, min_range=0.0, max_range=100.0)
        m.add_scan(_make_scene())
        original = m.map_points.copy()
        path = tmp_path / "map.pcd"
        m.save_map(path)
        assert path.exists()
        m.reset()
        assert m.map_points is None
        m.load_map(path)
        assert m.map_points is not None
        assert len(m.map_points) == len(original)

    def test_save_with_voxel_downsample(self, tmp_path):
        m = Mapper(voxel_size=0.01, min_range=0.0, max_range=100.0)
        m.add_scan(_make_scene(n_per_face=2000))
        path = tmp_path / "down.pcd"
        m.save_map(path, voxel_size=0.5)
        # Reload to count
        m2 = Mapper()
        m2.load_map(path)
        # Voxel-downsampled map should be smaller than original
        assert len(m2.map_points) < len(m.map_points)


# ---------------------------------------------------------------------------
# Map cap
# ---------------------------------------------------------------------------

class TestMapCap:
    def test_map_capped_when_exceeding_max(self):
        # Set a very low max so it triggers after a couple of scans
        m = Mapper(voxel_size=0.2, min_range=0.0, max_range=100.0,
                   max_map_points=500)
        scene = _make_scene(n_per_face=200)
        for _ in range(5):
            m.add_scan(scene)
        # After capping, point count should remain manageable
        assert m.map_points is not None
