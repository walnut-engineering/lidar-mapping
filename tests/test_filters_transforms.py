"""
Unit tests for:
  - lidar_mapping.processing.filters
  - lidar_mapping.utils.transforms
  - lidar_mapping.processing.registration  (no-Open3D paths)

All tests are pure-numpy and require no optional dependencies.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

from lidar_mapping.processing.filters import (
    compute_point_density,
    farthest_point_sample,
    intensity_filter,
    passthrough_filter,
    random_downsample,
    range_filter,
)


def _cloud(n: int = 200, seed: int = 0) -> np.ndarray:
    """Return a reproducible (N, 3) float32 cloud centred near the origin."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-10.0, 10.0, (n, 3)).astype(np.float32)


class TestRangeFilter:
    def test_keeps_points_in_range(self):
        pts = np.array([[1.0, 0.0, 0.0], [5.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
        result = range_filter(pts, min_range=0.5, max_range=10.0)
        assert len(result) == 2
        assert 1.0 in result[:, 0]
        assert 5.0 in result[:, 0]

    def test_rejects_below_min(self):
        pts = np.array([[0.1, 0.0, 0.0]])
        result = range_filter(pts, min_range=0.5, max_range=10.0)
        assert len(result) == 0

    def test_rejects_above_max(self):
        pts = np.array([[200.0, 0.0, 0.0]])
        result = range_filter(pts, min_range=0.5, max_range=100.0)
        assert len(result) == 0

    def test_empty_input(self):
        pts = np.empty((0, 3), dtype=np.float32)
        result = range_filter(pts, 0.5, 100.0)
        assert result.shape == (0, 3)

    def test_preserves_extra_columns(self):
        pts = np.array([[1.0, 0.0, 0.0, 99.0]])  # 4-col with intensity
        result = range_filter(pts, 0.5, 5.0)
        assert result.shape[1] == 4


class TestPassthroughFilter:
    def test_z_slice(self):
        pts = np.array([[0.0, 0.0, -5.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 5.0]])
        result = passthrough_filter(pts, axis=2, min_val=-1.0, max_val=1.0)
        assert len(result) == 1
        assert result[0, 2] == pytest.approx(0.0)

    def test_x_axis(self):
        pts = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        result = passthrough_filter(pts, axis=0, min_val=0.0, max_val=10.0)
        assert len(result) == 1

    def test_inclusive_bounds(self):
        pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
        result = passthrough_filter(pts, axis=2, min_val=0.0, max_val=5.0)
        assert len(result) == 2


class TestIntensityFilter:
    def test_filters_low_intensity(self):
        pts = np.array([[0.0, 0.0, 0.0, 0.5],
                        [0.0, 0.0, 0.0, 10.0],
                        [0.0, 0.0, 0.0, 50.0]])
        result = intensity_filter(pts, min_intensity=5.0, intensity_column=3)
        assert len(result) == 2

    def test_no_intensity_column_passthrough(self):
        pts = np.array([[1.0, 2.0, 3.0]])
        result = intensity_filter(pts, min_intensity=0.0, intensity_column=3)
        assert len(result) == 1  # column doesn't exist → passthrough


class TestRandomDownsample:
    def test_reduces_count(self):
        pts = _cloud(500)
        result = random_downsample(pts, n_keep=100)
        assert len(result) == 100

    def test_no_change_if_smaller(self):
        pts = _cloud(50)
        result = random_downsample(pts, n_keep=200)
        assert len(result) == 50

    def test_reproducible_with_seed(self):
        pts = _cloud(500)
        r1 = random_downsample(pts, n_keep=100, seed=7)
        r2 = random_downsample(pts, n_keep=100, seed=7)
        np.testing.assert_array_equal(r1, r2)


class TestFarthestPointSample:
    def test_reduces_count(self):
        pts = _cloud(200)
        result = farthest_point_sample(pts, n_keep=50)
        assert len(result) == 50

    def test_no_change_if_smaller(self):
        pts = _cloud(20)
        result = farthest_point_sample(pts, n_keep=100)
        assert len(result) == 20

    def test_points_are_from_original(self):
        pts = _cloud(100)
        result = farthest_point_sample(pts, n_keep=10)
        # Every selected row must appear somewhere in the original
        for row in result:
            assert np.any(np.all(pts == row, axis=1))


class TestComputePointDensity:
    def test_isolated_point_has_zero_neighbours(self):
        pts = np.array([[0.0, 0.0, 0.0],
                        [100.0, 0.0, 0.0]])
        counts = compute_point_density(pts, radius=1.0)
        assert counts[0] == 0  # no neighbour within 1 m
        assert counts[1] == 0

    def test_clustered_points_have_high_count(self):
        pts = np.zeros((50, 3), dtype=np.float32)  # all at origin
        counts = compute_point_density(pts, radius=0.5)
        assert np.all(counts == 49)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

from lidar_mapping.utils.transforms import (
    apply_transform,
    compose_transforms,
    euler_from_rotation,
    interpolate_transforms,
    invert_transform,
    make_transform,
    make_transform_from_euler,
    rotation_from_euler,
)


class TestRotationFromEuler:
    def test_identity_at_zero(self):
        R = rotation_from_euler(0.0, 0.0, 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_90deg_yaw_z(self):
        R = rotation_from_euler(0.0, 0.0, 90.0)
        # Column 0 (original X) should map to +Y after 90° yaw
        np.testing.assert_allclose(R[:, 0], [0.0, 1.0, 0.0], atol=1e-10)

    def test_radians_mode(self):
        R_deg = rotation_from_euler(45.0, 0.0, 0.0, degrees=True)
        R_rad = rotation_from_euler(math.radians(45.0), 0.0, 0.0, degrees=False)
        np.testing.assert_allclose(R_deg, R_rad, atol=1e-10)

    def test_orthogonality(self):
        R = rotation_from_euler(30.0, 45.0, 60.0)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10


class TestEulerFromRotation:
    def test_roundtrip(self):
        for roll, pitch, yaw in [(30.0, 20.0, 45.0),
                                  (-10.0, 5.0, -90.0),
                                  (0.0, 0.0, 0.0)]:
            R = rotation_from_euler(roll, pitch, yaw)
            r2, p2, y2 = euler_from_rotation(R)
            assert abs(r2 - roll) < 0.001
            assert abs(p2 - pitch) < 0.001
            assert abs(y2 - yaw) < 0.001


class TestMakeTransform:
    def test_identity_parts(self):
        T = make_transform(np.eye(3), np.zeros(3))
        np.testing.assert_allclose(T, np.eye(4), atol=1e-10)

    def test_translation_component(self):
        T = make_transform(np.eye(3), np.array([1.0, 2.0, 3.0]))
        assert T[0, 3] == 1.0
        assert T[1, 3] == 2.0
        assert T[2, 3] == 3.0

    def test_make_transform_from_euler_convenience(self):
        T = make_transform_from_euler(0.0, 0.0, 90.0, tx=1.0, ty=2.0, tz=3.0)
        assert T.shape == (4, 4)
        assert T[0, 3] == pytest.approx(1.0)


class TestApplyTransform:
    def test_identity_does_not_move_points(self):
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        T = np.eye(4)
        result = apply_transform(pts, T)
        np.testing.assert_allclose(result, pts[:, :3], atol=1e-10)

    def test_pure_translation(self):
        pts = np.array([[0.0, 0.0, 0.0]])
        T = make_transform(np.eye(3), np.array([10.0, 0.0, 0.0]))
        result = apply_transform(pts, T)
        np.testing.assert_allclose(result, [[10.0, 0.0, 0.0]], atol=1e-10)

    def test_90deg_yaw_rotation(self):
        pts = np.array([[1.0, 0.0, 0.0]])
        R = rotation_from_euler(0.0, 0.0, 90.0)
        T = make_transform(R, np.zeros(3))
        result = apply_transform(pts, T)
        np.testing.assert_allclose(result, [[0.0, 1.0, 0.0]], atol=1e-10)


class TestComposeTransforms:
    def test_two_identities(self):
        T = compose_transforms(np.eye(4), np.eye(4))
        np.testing.assert_allclose(T, np.eye(4), atol=1e-10)

    def test_translation_accumulates(self):
        t = np.array([1.0, 0.0, 0.0])
        T = make_transform(np.eye(3), t)
        T2 = compose_transforms(T, T)
        assert T2[0, 3] == pytest.approx(2.0)


class TestInvertTransform:
    def test_invert_identity(self):
        T_inv = invert_transform(np.eye(4))
        np.testing.assert_allclose(T_inv, np.eye(4), atol=1e-10)

    def test_T_times_T_inv_is_identity(self):
        T = make_transform_from_euler(30.0, 20.0, 45.0, tx=1.0, ty=-2.0, tz=0.5)
        T_inv = invert_transform(T)
        product = T @ T_inv
        np.testing.assert_allclose(product, np.eye(4), atol=1e-10)


class TestInterpolateTransforms:
    def test_alpha_zero_returns_T1(self):
        T1 = make_transform_from_euler(0.0, 0.0, 0.0, tx=0.0)
        T2 = make_transform_from_euler(0.0, 0.0, 90.0, tx=10.0)
        T = interpolate_transforms(T1, T2, 0.0)
        np.testing.assert_allclose(T, T1, atol=1e-6)

    def test_alpha_one_returns_T2(self):
        T1 = make_transform_from_euler(0.0, 0.0, 0.0, tx=0.0)
        T2 = make_transform_from_euler(0.0, 0.0, 90.0, tx=10.0)
        T = interpolate_transforms(T1, T2, 1.0)
        np.testing.assert_allclose(T[:3, 3], T2[:3, 3], atol=1e-6)

    def test_midpoint_translation(self):
        T1 = make_transform(np.eye(3), np.array([0.0, 0.0, 0.0]))
        T2 = make_transform(np.eye(3), np.array([10.0, 0.0, 0.0]))
        T_mid = interpolate_transforms(T1, T2, 0.5)
        assert T_mid[0, 3] == pytest.approx(5.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Registration — no-Open3D paths
# ---------------------------------------------------------------------------

from lidar_mapping.processing.registration import RegistrationResult


class TestRegistrationResult:
    def test_default_values(self):
        r = RegistrationResult(transform=np.eye(4))
        assert r.fitness == 0.0
        assert r.inlier_rmse == float("inf")
        assert r.converged is False
        assert r.iterations == 0

    def test_custom_values(self):
        T = np.eye(4)
        T[0, 3] = 1.5
        r = RegistrationResult(
            transform=T, fitness=0.9, inlier_rmse=0.02, converged=True, iterations=30
        )
        assert r.fitness == 0.9
        assert r.converged is True
        np.testing.assert_array_equal(r.transform, T)


class TestICPRegistrationNoOpen3D:
    """Verify that ICPRegistration raises ImportError gracefully when open3d
    is absent, or when voxel_size=None is used without a correspondence
    distance and open3d is unavailable."""

    def test_import_error_without_open3d(self, monkeypatch):
        """If open3d is not installed, register() must raise ImportError."""
        import lidar_mapping.processing.registration as reg_mod

        monkeypatch.setattr(reg_mod, "_O3D_AVAILABLE", False)
        icp = reg_mod.ICPRegistration(voxel_size=0.1)
        with pytest.raises(ImportError, match="open3d"):
            icp.register(
                np.random.randn(50, 3).astype(np.float32),
                np.random.randn(50, 3).astype(np.float32),
            )

    def test_voxel_none_correspondence_none_raises_without_open3d(self, monkeypatch):
        """
        voxel_size=None with no explicit max_correspondence_distance would
        compute 3 * None at runtime — verify it raises before calling open3d.
        """
        import lidar_mapping.processing.registration as reg_mod

        monkeypatch.setattr(reg_mod, "_O3D_AVAILABLE", False)
        with pytest.raises(TypeError):
            # 3 * None raises TypeError — caught before we even call register()
            icp = reg_mod.ICPRegistration(voxel_size=None)
            icp.register(
                np.random.randn(50, 3).astype(np.float32),
                np.random.randn(50, 3).astype(np.float32),
            )
