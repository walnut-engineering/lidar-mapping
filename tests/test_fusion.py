"""Tests for the camera-LiDAR fusion module."""

from __future__ import annotations

import numpy as np
import pytest

from lidar_mapping.processing.fusion import (
    CameraIntrinsics,
    ColorizedCloud,
    colorize_points,
    depth_image_from_points,
    project_points,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _intr(w=640, h=480, fx=500.0, fy=500.0):
    return CameraIntrinsics(fx=fx, fy=fy, cx=w / 2, cy=h / 2,
                            width=w, height=h)


# ---------------------------------------------------------------------------
# CameraIntrinsics
# ---------------------------------------------------------------------------

class TestCameraIntrinsics:
    def test_K_matrix(self):
        intr = CameraIntrinsics(fx=100, fy=200, cx=320, cy=240,
                                width=640, height=480)
        np.testing.assert_array_equal(intr.K, [
            [100, 0, 320],
            [0, 200, 240],
            [0, 0, 1],
        ])

    def test_from_matrix(self):
        K = np.array([[100, 0, 50], [0, 200, 75], [0, 0, 1]], dtype=np.float64)
        intr = CameraIntrinsics.from_matrix(K, 100, 150)
        assert intr.fx == 100
        assert intr.fy == 200
        assert intr.cx == 50
        assert intr.cy == 75
        assert intr.width == 100

    def test_invalid_focal_raises(self):
        with pytest.raises(ValueError):
            CameraIntrinsics(fx=0, fy=1, cx=0, cy=0, width=1, height=1)

    def test_invalid_size_raises(self):
        with pytest.raises(ValueError):
            CameraIntrinsics(fx=1, fy=1, cx=0, cy=0, width=0, height=1)

    def test_invalid_distortion_raises(self):
        with pytest.raises(ValueError):
            CameraIntrinsics(fx=1, fy=1, cx=0, cy=0, width=1, height=1,
                             distortion=np.zeros(4))

    def test_invalid_K_shape(self):
        with pytest.raises(ValueError):
            CameraIntrinsics.from_matrix(np.eye(4), 100, 100)


# ---------------------------------------------------------------------------
# project_points
# ---------------------------------------------------------------------------

class TestProjectPoints:
    def test_center_axis_lands_at_principal_point(self):
        intr = _intr()
        pts = np.array([[0.0, 0.0, 5.0]])
        uv, valid = project_points(pts, intr)
        assert valid[0]
        np.testing.assert_allclose(uv[0], [320.0, 240.0])

    def test_behind_camera_invalid(self):
        intr = _intr()
        pts = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
        uv, valid = project_points(pts, intr)
        assert not valid.any()
        assert np.isnan(uv).all()

    def test_out_of_frame_invalid(self):
        intr = _intr()
        # Way off to the right
        pts = np.array([[100.0, 0.0, 1.0]])
        uv, valid = project_points(pts, intr)
        assert not valid[0]

    def test_known_pixel(self):
        # fx=500, point at X=1, Z=10 → u = cx + 500*0.1 = cx + 50
        intr = _intr()
        pts = np.array([[1.0, 0.0, 10.0]])
        uv, valid = project_points(pts, intr)
        assert valid[0]
        assert uv[0, 0] == pytest.approx(intr.cx + 50.0)
        assert uv[0, 1] == pytest.approx(intr.cy)

    def test_distortion_changes_projection(self):
        d = np.array([0.1, 0.0, 0.0, 0.0, 0.0])  # mild radial only
        intr = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240,
                                width=640, height=480, distortion=d)
        intr_nd = _intr()
        pts = np.array([[1.0, 0.0, 5.0]])
        uv_d, _ = project_points(pts, intr)
        uv_nd, _ = project_points(pts, intr_nd)
        assert uv_d[0, 0] != uv_nd[0, 0]

    def test_shape_validation(self):
        intr = _intr()
        with pytest.raises(ValueError):
            project_points(np.array([1.0, 2.0, 3.0]), intr)


# ---------------------------------------------------------------------------
# colorize_points
# ---------------------------------------------------------------------------

class TestColorize:
    def test_returns_correct_shape(self):
        intr = _intr(w=10, h=10, fx=10, fy=10)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[5, 5] = [255, 0, 0]
        pts = np.array([[0.0, 0.0, 1.0]])  # projects to principal point (5, 5)
        out = colorize_points(pts, img, intr)
        assert isinstance(out, ColorizedCloud)
        assert out.points.shape == (1, 3)
        assert out.colors.shape == (1, 3)
        assert out.mask.shape == (1,)
        assert out.depths.shape == (1,)

    def test_nearest_samples_correct_color(self):
        intr = _intr(w=10, h=10, fx=10, fy=10)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[5, 5] = [11, 22, 33]
        pts = np.array([[0.0, 0.0, 1.0]])
        out = colorize_points(pts, img, intr, sampling="nearest")
        np.testing.assert_array_equal(out.colors[0], [11, 22, 33])

    def test_bilinear_sampling(self):
        intr = _intr(w=10, h=10, fx=10, fy=10)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[5, 5] = [100, 100, 100]
        img[5, 6] = [200, 200, 200]
        # Project to u≈5.5, v=5  → bilinear should give 150
        pts = np.array([[0.05, 0.0, 1.0]])  # x/z = 0.05 → u = 5 + 0.5 = 5.5
        out = colorize_points(pts, img, intr, sampling="bilinear")
        np.testing.assert_allclose(out.colors[0], [150, 150, 150], atol=1)

    def test_with_extrinsic(self):
        # LiDAR point at (1, 0, 0); camera frame offset so it ends up in
        # front of the camera at Z=1.  Use a transform that maps
        # LiDAR (1,0,0) → camera (0,0,1).
        intr = _intr(w=10, h=10, fx=10, fy=10)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[5, 5] = [50, 60, 70]

        # T maps lidar→cam: rotate so lidar +X becomes cam +Z
        T = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        pts = np.array([[1.0, 0.0, 0.0]])
        out = colorize_points(pts, img, intr, T_cam_from_points=T)
        assert out.mask[0]
        np.testing.assert_array_equal(out.colors[0], [50, 60, 70])

    def test_mask_excludes_behind_and_oob(self):
        intr = _intr(w=10, h=10, fx=10, fy=10)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        pts = np.array([
            [0.0, 0.0, 1.0],     # OK
            [0.0, 0.0, -1.0],    # behind
            [100.0, 0.0, 1.0],   # far right
        ])
        out = colorize_points(pts, img, intr)
        np.testing.assert_array_equal(out.mask, [True, False, False])
        assert out.points.shape == (1, 3)

    def test_no_valid_returns_empty(self):
        intr = _intr(w=10, h=10, fx=10, fy=10)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        pts = np.array([[0.0, 0.0, -5.0]])
        out = colorize_points(pts, img, intr)
        assert out.points.shape == (0, 3)
        assert out.colors.shape == (0, 3)
        assert not out.mask.any()

    def test_depths_match_camera_z(self):
        intr = _intr(w=10, h=10, fx=10, fy=10)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        pts = np.array([[0.0, 0.0, 3.5], [0.0, 0.0, 7.0]])
        out = colorize_points(pts, img, intr)
        np.testing.assert_allclose(out.depths, [3.5, 7.0])

    def test_image_size_mismatch_raises(self):
        intr = _intr(w=10, h=10, fx=10, fy=10)
        img = np.zeros((20, 20, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            colorize_points(np.array([[0., 0., 1.]]), img, intr)

    def test_invalid_extrinsic_shape(self):
        intr = _intr(w=10, h=10, fx=10, fy=10)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            colorize_points(np.array([[0., 0., 1.]]), img, intr,
                            T_cam_from_points=np.eye(3))

    def test_invalid_sampling_mode(self):
        intr = _intr(w=10, h=10, fx=10, fy=10)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            colorize_points(np.array([[0., 0., 1.]]), img, intr,
                            sampling="cubic")

    def test_image_with_alpha_channel(self):
        intr = _intr(w=10, h=10, fx=10, fy=10)
        img = np.zeros((10, 10, 4), dtype=np.uint8)
        img[5, 5] = [10, 20, 30, 255]
        pts = np.array([[0.0, 0.0, 1.0]])
        out = colorize_points(pts, img, intr)
        np.testing.assert_array_equal(out.colors[0], [10, 20, 30, 255])


# ---------------------------------------------------------------------------
# depth_image_from_points
# ---------------------------------------------------------------------------

class TestDepthImage:
    def test_inf_when_empty(self):
        intr = _intr(w=10, h=10, fx=10, fy=10)
        depth = depth_image_from_points(np.zeros((0, 3)), intr)
        assert depth.shape == (10, 10)
        assert np.isinf(depth).all()

    def test_closest_depth_wins(self):
        intr = _intr(w=10, h=10, fx=10, fy=10)
        # Two points project to (5, 5) at depths 3 and 7
        pts = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 7.0]])
        depth = depth_image_from_points(pts, intr)
        assert depth[5, 5] == pytest.approx(3.0)

    def test_pixels_without_points_are_inf(self):
        intr = _intr(w=10, h=10, fx=10, fy=10)
        pts = np.array([[0.0, 0.0, 5.0]])
        depth = depth_image_from_points(pts, intr)
        # Most pixels are inf
        assert np.isinf(depth[0, 0])
        assert depth[5, 5] == pytest.approx(5.0)

    def test_with_extrinsic(self):
        intr = _intr(w=10, h=10, fx=10, fy=10)
        T = np.eye(4)
        T[2, 3] = 2.0  # shift +2m along camera Z
        pts = np.array([[0.0, 0.0, 1.0]])  # → camera Z=3
        depth = depth_image_from_points(pts, intr, T_cam_from_points=T)
        assert depth[5, 5] == pytest.approx(3.0)
