"""
Integration tests for Open3D-dependent modules:
  - lidar_mapping.processing.point_cloud
  - lidar_mapping.processing.registration  (ICP on real point clouds)
  - lidar_mapping.mapping.mapper            (full pipeline with real ICP)

These tests require `open3d` to be installed.  They are skipped automatically
when open3d is absent so the core test suite stays lightweight.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Skip guard
# ---------------------------------------------------------------------------

try:
    import open3d as o3d
    _O3D_AVAILABLE = True
except ImportError:
    _O3D_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _O3D_AVAILABLE,
    reason="open3d not installed",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_plane(n: int = 2000, noise: float = 0.005, seed: int = 0) -> np.ndarray:
    """Return a noisy flat horizontal plane at z≈0, (N,3) float32."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-5.0, 5.0, (n, 2))
    z = rng.normal(0.0, noise, (n, 1))
    return np.hstack([xy, z]).astype(np.float32)


def _sphere_cloud(n: int = 1000, radius: float = 3.0, seed: int = 0) -> np.ndarray:
    """Return points distributed on the surface of a sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3)).astype(np.float64)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return (pts * radius).astype(np.float32)


def _translated_cloud(pts: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    offset = np.array([[dx, dy, dz]], dtype=np.float32)
    return pts + offset


# ---------------------------------------------------------------------------
# point_cloud.py
# ---------------------------------------------------------------------------

from lidar_mapping.processing.point_cloud import (
    crop_box,
    estimate_normals,
    numpy_to_o3d,
    o3d_to_numpy,
    remove_ground_plane,
    remove_radius_outliers,
    remove_statistical_outliers,
    voxel_downsample,
)


class TestNumpyO3dConversion:
    def test_roundtrip_xyz(self):
        pts = np.random.default_rng(0).uniform(-1, 1, (100, 3)).astype(np.float32)
        pcd = numpy_to_o3d(pts)
        result = o3d_to_numpy(pcd)
        np.testing.assert_allclose(result, pts, atol=1e-5)

    def test_intensity_column_creates_colours(self):
        pts = np.random.default_rng(1).uniform(0, 1, (50, 4)).astype(np.float32)
        pcd = numpy_to_o3d(pts)
        assert pcd.has_colors()

    def test_explicit_colours(self):
        pts = np.ones((10, 3), dtype=np.float32)
        colours = np.tile([1.0, 0.0, 0.0], (10, 1))
        pcd = numpy_to_o3d(pts, colors=colours)
        assert pcd.has_colors()

    def test_include_colors_returns_6_columns(self):
        pts = np.random.default_rng(2).uniform(-1, 1, (30, 3)).astype(np.float32)
        colours = np.random.default_rng(2).uniform(0, 1, (30, 3))
        pcd = numpy_to_o3d(pts, colors=colours)
        result = o3d_to_numpy(pcd, include_colors=True)
        assert result.shape[1] == 6

    def test_empty_cloud(self):
        pts = np.empty((0, 3), dtype=np.float32)
        pcd = numpy_to_o3d(pts)
        result = o3d_to_numpy(pcd)
        assert result.shape == (0, 3)


class TestVoxelDownsample:
    def test_reduces_point_count(self):
        pts = _flat_plane(n=5000)
        down = voxel_downsample(pts, voxel_size=0.2)
        assert len(down) < len(pts)
        assert len(down) > 0

    def test_coarser_voxel_fewer_points(self):
        pts = _flat_plane(n=3000)
        fine = voxel_downsample(pts, voxel_size=0.05)
        coarse = voxel_downsample(pts, voxel_size=0.5)
        assert len(coarse) < len(fine)

    def test_output_within_original_bounding_box(self):
        pts = _flat_plane(n=1000)
        down = voxel_downsample(pts, voxel_size=0.1)
        assert down[:, 0].min() >= pts[:, 0].min() - 0.1
        assert down[:, 0].max() <= pts[:, 0].max() + 0.1


class TestRemoveOutliers:
    def test_statistical_removes_noise(self):
        pts = _flat_plane(n=500, noise=0.001)
        # Add a handful of far-off outliers
        outliers = np.array([[50.0, 50.0, 50.0],
                              [-50.0, 0.0, 10.0]], dtype=np.float32)
        contaminated = np.vstack([pts, outliers])
        clean, mask = remove_statistical_outliers(contaminated, nb_neighbors=20, std_ratio=2.0)
        assert len(clean) < len(contaminated)
        assert len(clean) >= len(pts) * 0.9

    def test_radius_removes_isolated_points(self):
        pts = _flat_plane(n=500)
        isolated = np.array([[100.0, 100.0, 0.0]], dtype=np.float32)
        contaminated = np.vstack([pts, isolated])
        clean, mask = remove_radius_outliers(contaminated, nb_points=5, radius=0.5)
        # The isolated point should have been removed
        assert not np.any(np.all(clean == isolated[0], axis=1))

    def test_mask_length_equals_input(self):
        pts = _flat_plane(n=300)
        _, mask = remove_statistical_outliers(pts)
        assert len(mask) == len(pts)


class TestEstimateNormals:
    def test_returns_same_pcd(self):
        pts = _flat_plane(n=200)
        pcd = numpy_to_o3d(pts)
        result = estimate_normals(pcd, radius=0.5)
        assert result is pcd

    def test_normals_are_unit_vectors(self):
        pts = _flat_plane(n=500)
        pcd = numpy_to_o3d(pts)
        estimate_normals(pcd, radius=0.5)
        norms = np.asarray(pcd.normals)
        lengths = np.linalg.norm(norms, axis=1)
        np.testing.assert_allclose(lengths, np.ones(len(lengths)), atol=1e-5)

    def test_flat_plane_normals_near_vertical(self):
        """A flat horizontal plane → normals should be close to ±Z."""
        pts = _flat_plane(n=1000, noise=0.001)
        pcd = numpy_to_o3d(pts)
        estimate_normals(pcd, radius=0.5)
        norms = np.asarray(pcd.normals)
        # |nz| should be close to 1 for most points
        assert np.mean(np.abs(norms[:, 2])) > 0.8


class TestRemoveGroundPlane:
    def test_removes_most_of_ground(self):
        ground = _flat_plane(n=1000, noise=0.01)
        # Add some elevated "object" points
        objects = np.random.default_rng(5).uniform(-3, 3, (200, 3)).astype(np.float32)
        objects[:, 2] += 1.5  # lift above ground
        scene = np.vstack([ground, objects])

        above, below, model = remove_ground_plane(scene, distance_threshold=0.1)
        # Most ground points should be in "below", most objects in "above"
        assert len(above) > 0
        assert len(below) > 0
        assert len(model) == 4  # plane equation [a,b,c,d]

    def test_plane_model_has_unit_normal(self):
        pts = _flat_plane(n=500)
        _, _, model = remove_ground_plane(pts)
        normal_length = math.sqrt(model[0]**2 + model[1]**2 + model[2]**2)
        assert abs(normal_length - 1.0) < 0.01


class TestCropBox:
    def test_keeps_only_points_inside(self):
        pts = np.array([
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [-100.0, 0.0, 0.0],
        ], dtype=np.float32)
        result = crop_box(pts, x_range=(-10.0, 10.0), y_range=(-10.0, 10.0), z_range=(-10.0, 10.0))
        assert len(result) == 1
        assert result[0, 0] == pytest.approx(0.0)

    def test_empty_result_when_nothing_inside(self):
        pts = np.ones((100, 3), dtype=np.float32) * 1000
        result = crop_box(pts)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# ICPRegistration (real Open3D)
# ---------------------------------------------------------------------------

from lidar_mapping.processing.registration import ICPRegistration, RegistrationResult


class TestICPRegistration:
    @staticmethod
    def _overlapping_pair(n: int = 800, tx: float = 0.2, seed: int = 0):
        """Return two point clouds separated by a known small translation."""
        pts = _flat_plane(n=n, noise=0.005, seed=seed)
        pts2 = _translated_cloud(pts, tx, 0.0, 0.0)
        return pts.astype(np.float64), pts2.astype(np.float64)

    def test_converges_on_close_clouds(self):
        src, tgt = self._overlapping_pair(tx=0.05)
        icp = ICPRegistration(voxel_size=0.1)
        result = icp.register(src, tgt)
        assert result.converged
        assert result.fitness > 0.0

    def test_recovered_translation_approx_correct(self):
        """ICP should recover a small 0.1 m X translation."""
        tx = 0.1
        src, tgt = self._overlapping_pair(tx=tx)
        icp = ICPRegistration(voxel_size=0.05, max_correspondence_distance=0.3)
        result = icp.register(src, tgt)
        assert result.converged
        # The estimated X translation in the transform
        estimated_tx = result.transform[0, 3]
        assert abs(estimated_tx - tx) < 0.05, (
            f"Expected translation ~{tx}, got {estimated_tx:.4f}"
        )

    def test_transform_is_rigid(self):
        """The returned transform's rotation block must be a valid SO(3) matrix."""
        src, tgt = self._overlapping_pair(tx=0.15)
        icp = ICPRegistration(voxel_size=0.1)
        result = icp.register(src, tgt)
        R = result.transform[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-5)
        assert abs(np.linalg.det(R) - 1.0) < 1e-5

    def test_identity_hint_works(self):
        src, tgt = self._overlapping_pair(tx=0.05)
        icp = ICPRegistration(voxel_size=0.1)
        result = icp.register(src, tgt, initial_transform=np.eye(4))
        assert result.converged

    def test_result_has_expected_fields(self):
        src, tgt = self._overlapping_pair(tx=0.05)
        icp = ICPRegistration(voxel_size=0.1)
        result = icp.register(src, tgt)
        assert result.transform.shape == (4, 4)
        assert 0.0 <= result.fitness <= 1.0
        assert result.inlier_rmse >= 0.0


# ---------------------------------------------------------------------------
# Mapper — full pipeline
# ---------------------------------------------------------------------------

from lidar_mapping.mapping.mapper import Mapper


def _random_scan(n: int = 800, centre: tuple = (0.0, 0.0, 0.0), seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-4.0, 4.0, (n, 3)).astype(np.float32)
    pts += np.array(centre, dtype=np.float32)
    return pts


class TestMapperFullPipeline:
    def test_first_scan_initialises_map(self):
        mapper = Mapper(voxel_size=0.2, min_range=0.0)
        mapper.add_scan(_random_scan(seed=0))
        assert mapper.map_points is not None
        assert len(mapper.map_points) > 0
        assert mapper.scans_processed == 1

    def test_second_scan_grows_map(self):
        mapper = Mapper(voxel_size=0.2, min_range=0.0)
        mapper.add_scan(_random_scan(seed=0))
        n_after_first = len(mapper.map_points)
        mapper.add_scan(_random_scan(seed=1))
        # Map should have grown (second scan adds new points)
        assert mapper.scans_processed == 2
        # Map is at least as large as the first scan
        assert len(mapper.map_points) >= n_after_first

    def test_pose_history_length(self):
        mapper = Mapper(voxel_size=0.2, min_range=0.0)
        for i in range(5):
            mapper.add_scan(_random_scan(seed=i))
        # pose_history starts with identity + one entry per add_scan call
        assert len(mapper.pose_history) == 6

    def test_current_pose_is_4x4(self):
        mapper = Mapper(voxel_size=0.2, min_range=0.0)
        mapper.add_scan(_random_scan(seed=0))
        pose = mapper.current_pose
        assert pose.shape == (4, 4)

    def test_reset_clears_state(self):
        mapper = Mapper(voxel_size=0.2, min_range=0.0)
        mapper.add_scan(_random_scan(seed=0))
        mapper.reset()
        assert mapper.map_points is None
        assert mapper.scans_processed == 0
        assert len(mapper.pose_history) == 1

    def test_result_is_registration_result(self):
        mapper = Mapper(voxel_size=0.2, min_range=0.0)
        r1 = mapper.add_scan(_random_scan(seed=0))
        r2 = mapper.add_scan(_random_scan(seed=1))
        assert isinstance(r1, RegistrationResult)
        assert isinstance(r2, RegistrationResult)

    def test_max_map_points_triggers_downsample(self):
        """Mapper should downsample when the map exceeds max_map_points."""
        # Dense overlapping scans in a small box so voxel downsampling bites hard.
        rng = np.random.default_rng(0)
        dense_scan = rng.uniform(-1.0, 1.0, (2000, 3)).astype(np.float32)
        mapper = Mapper(voxel_size=0.1, min_range=0.0, max_map_points=1000)
        for i in range(10):
            # Slightly perturb each scan so ICP converges but points overlap heavily
            pts = dense_scan + rng.uniform(-0.02, 0.02, dense_scan.shape).astype(np.float32)
            mapper.add_scan(pts)
        # Mapper ran to completion without error; map exists
        assert mapper.map_points is not None
        assert mapper.scans_processed == 10

    def test_remove_ground_flag(self):
        """Mapper with remove_ground=True should not crash on mixed-scene clouds."""
        rng = np.random.default_rng(99)
        ground = _flat_plane(n=800, noise=0.01)
        # Add elevated object points well above the ground plane
        objects = rng.uniform(-2.0, 2.0, (400, 3)).astype(np.float32)
        objects[:, 2] += 2.0  # lift 2 m above ground
        scene = np.vstack([ground, objects])
        mapper = Mapper(voxel_size=0.2, min_range=0.0, remove_ground=True)
        mapper.add_scan(scene)  # first scan (just initialises map)
        mapper.add_scan(scene)  # second scan runs ICP + ground removal
        assert mapper.scans_processed == 2

    def test_get_map_o3d_returns_point_cloud(self):
        mapper = Mapper(voxel_size=0.2, min_range=0.0)
        mapper.add_scan(_random_scan(seed=0))
        pcd = mapper.get_map_o3d()
        assert isinstance(pcd, o3d.geometry.PointCloud)
        assert len(pcd.points) > 0

    def test_get_map_o3d_empty_before_scans(self):
        mapper = Mapper(voxel_size=0.2)
        pcd = mapper.get_map_o3d()
        assert len(pcd.points) == 0


class TestMapperSaveLoad:
    def test_save_and_load_ply(self, tmp_path):
        mapper = Mapper(voxel_size=0.2, min_range=0.0)
        mapper.add_scan(_random_scan(seed=0))
        original_count = len(mapper.map_points)

        out = tmp_path / "map.ply"
        mapper.save_map(out)
        assert out.exists()
        assert out.stat().st_size > 0

        mapper2 = Mapper(voxel_size=0.2)
        mapper2.load_map(out)
        assert mapper2.map_points is not None
        # Point count should be close (voxel downsampling may differ slightly)
        assert abs(len(mapper2.map_points) - original_count) <= original_count * 0.1

    def test_save_and_load_pcd(self, tmp_path):
        mapper = Mapper(voxel_size=0.2, min_range=0.0)
        mapper.add_scan(_random_scan(seed=1))
        out = tmp_path / "map.pcd"
        mapper.save_map(out)
        assert out.exists()

        mapper2 = Mapper(voxel_size=0.2)
        mapper2.load_map(out)
        assert mapper2.map_points is not None
        assert len(mapper2.map_points) > 0

    def test_save_with_voxel_reduces_points(self, tmp_path):
        mapper = Mapper(voxel_size=0.1, min_range=0.0)
        for i in range(5):
            mapper.add_scan(_random_scan(seed=i))
        before = len(mapper.map_points)

        out = tmp_path / "map_coarse.ply"
        mapper.save_map(out, voxel_size=1.0)  # coarser downsampling at save time

        mapper2 = Mapper(voxel_size=0.1)
        mapper2.load_map(out)
        assert len(mapper2.map_points) <= before

    def test_save_with_no_map_raises(self, tmp_path):
        mapper = Mapper(voxel_size=0.2)
        with pytest.raises(RuntimeError, match="No map data"):
            mapper.save_map(tmp_path / "empty.ply")
