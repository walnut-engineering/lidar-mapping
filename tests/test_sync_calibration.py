"""Tests for time-sync utilities and extrinsic calibration."""

import math
import numpy as np
import pytest

from lidar_mapping.utils.sync import (
    TimeAlignedBuffer, slerp, interpolate_pose, estimate_clock_offset,
    _rotation_to_quaternion,
)
from lidar_mapping.utils.calibration import (
    Extrinsic, compute_extrinsic_svd, hand_eye_calibration,
    apply_extrinsic_to_points, chain_extrinsics, _rot_to_axis_angle,
)
from lidar_mapping.utils.transforms import (
    make_transform, make_transform_from_euler, apply_transform,
)


# ---------------------------------------------------------------------------
# TimeAlignedBuffer
# ---------------------------------------------------------------------------

class TestTimeAlignedBuffer:
    def test_push_and_len(self):
        b = TimeAlignedBuffer()
        b.push(1.0, "a")
        b.push(2.0, "b")
        assert len(b) == 2

    def test_nearest_returns_closest(self):
        b = TimeAlignedBuffer()
        for t, v in [(0.0, "x"), (1.0, "y"), (2.0, "z")]:
            b.push(t, v)
        assert b.nearest(0.4)[1] == "x"
        assert b.nearest(0.6)[1] == "y"
        assert b.nearest(10.0)[1] == "z"
        assert b.nearest(-5.0)[1] == "x"

    def test_nearest_empty(self):
        assert TimeAlignedBuffer().nearest(0.0) is None

    def test_bracket(self):
        b = TimeAlignedBuffer()
        for t in (0.0, 1.0, 2.0, 3.0):
            b.push(t, t * 10)
        (t0, v0), (t1, v1) = b.bracket(1.5)
        assert t0 == 1.0 and v0 == 10
        assert t1 == 2.0 and v1 == 20

    def test_bracket_out_of_range(self):
        b = TimeAlignedBuffer()
        b.push(0.0, "a")
        b.push(1.0, "b")
        assert b.bracket(5.0) is None
        assert b.bracket(-1.0) is None

    def test_range_query(self):
        b = TimeAlignedBuffer()
        for t in range(10):
            b.push(float(t), t)
        result = b.range(2.0, 5.0)
        assert [v for _, v in result] == [2, 3, 4, 5]

    def test_capacity(self):
        b = TimeAlignedBuffer(max_samples=3)
        for t in range(10):
            b.push(float(t), t)
        assert len(b) == 3
        # Should retain newest three
        assert b.nearest(9.0)[1] == 9

    def test_out_of_order_insert(self):
        b = TimeAlignedBuffer()
        b.push(0.0, "a")
        b.push(2.0, "c")
        b.push(1.0, "b")  # out of order
        assert b.range(0.0, 2.0) == [(0.0, "a"), (1.0, "b"), (2.0, "c")]

    def test_clear(self):
        b = TimeAlignedBuffer()
        b.push(0.0, 1)
        b.clear()
        assert len(b) == 0


# ---------------------------------------------------------------------------
# SLERP / pose interp
# ---------------------------------------------------------------------------

class TestSlerp:
    def test_endpoints(self):
        q0 = np.array([1, 0, 0, 0], dtype=float)
        q1 = np.array([0, 0, 0, 1], dtype=float)
        assert np.allclose(slerp(q0, q1, 0.0), q0)
        assert np.allclose(slerp(q0, q1, 1.0), q1)

    def test_midpoint_normalised(self):
        q0 = np.array([1, 0, 0, 0], dtype=float)
        q1 = np.array([math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)])
        mid = slerp(q0, q1, 0.5)
        assert abs(np.linalg.norm(mid) - 1.0) < 1e-9

    def test_double_cover(self):
        q0 = np.array([1, 0, 0, 0], dtype=float)
        q1 = np.array([-1, 0, 0, 0], dtype=float)  # same rotation
        # SLERP must take the short path
        mid = slerp(q0, q1, 0.5)
        # Could be +/-q0 — either way it's the identity rotation
        assert abs(abs(mid[0]) - 1.0) < 1e-9


class TestInterpolatePose:
    def test_midpoint_translation(self):
        T0 = make_transform(np.eye(3), np.array([0.0, 0, 0]))
        T1 = make_transform(np.eye(3), np.array([2.0, 0, 0]))
        T = interpolate_pose(0.0, T0, 1.0, T1, 0.5)
        assert np.allclose(T[:3, 3], [1.0, 0, 0])

    def test_same_timestamp_returns_pose0(self):
        T0 = make_transform(np.eye(3), np.array([1.0, 2, 3]))
        T1 = make_transform(np.eye(3), np.array([5.0, 6, 7]))
        T = interpolate_pose(1.0, T0, 1.0, T1, 1.0)
        assert np.allclose(T, T0)

    def test_rotation_interp(self):
        T0 = np.eye(4)
        T1 = make_transform_from_euler(0, 0, math.pi / 2,
                                       degrees=False)
        T = interpolate_pose(0.0, T0, 1.0, T1, 0.5)
        # Should be ~45° rotation about Z
        R = T[:3, :3]
        c = math.cos(math.pi / 4)
        s = math.sin(math.pi / 4)
        expected = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        assert np.allclose(R, expected, atol=1e-6)


class TestRotationToQuaternion:
    def test_identity(self):
        q = _rotation_to_quaternion(np.eye(3))
        assert np.allclose(q, [1, 0, 0, 0])

    def test_180_about_x(self):
        R = np.diag([1.0, -1.0, -1.0])
        q = _rotation_to_quaternion(R)
        # quaternion (0, ±1, 0, 0)
        assert abs(q[0]) < 1e-9
        assert abs(abs(q[1]) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Clock offset
# ---------------------------------------------------------------------------

class TestClockOffset:
    def test_median(self):
        pairs = [(0.0, 1.0), (1.0, 2.05), (2.0, 2.95)]
        assert abs(estimate_clock_offset(pairs) - 1.0) < 0.1

    def test_mean(self):
        pairs = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
        assert estimate_clock_offset(pairs, "mean") == 1.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            estimate_clock_offset([])

    def test_unknown_method(self):
        with pytest.raises(ValueError):
            estimate_clock_offset([(0, 1)], method="bogus")


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class TestExtrinsic:
    def test_inverse(self):
        T = make_transform_from_euler(0, 0, 0.5, 1.0, 2.0, 3.0, degrees=False)
        ext = Extrinsic("imu", "lidar", T)
        inv = ext.inverse()
        assert inv.src_frame == "lidar"
        assert inv.dst_frame == "imu"
        assert np.allclose(inv.transform @ T, np.eye(4), atol=1e-9)


class TestComputeExtrinsicSVD:
    def test_identity(self):
        pts = np.random.RandomState(0).rand(10, 3)
        T = compute_extrinsic_svd(pts, pts)
        assert np.allclose(T, np.eye(4), atol=1e-9)

    def test_pure_translation(self):
        rng = np.random.RandomState(1)
        src = rng.rand(20, 3)
        t = np.array([0.5, -1.0, 2.0])
        dst = src + t
        T = compute_extrinsic_svd(src, dst)
        assert np.allclose(T[:3, 3], t, atol=1e-9)
        assert np.allclose(T[:3, :3], np.eye(3), atol=1e-9)

    def test_rotation_plus_translation(self):
        rng = np.random.RandomState(2)
        src = rng.rand(30, 3)
        T_true = make_transform_from_euler(0.1, 0.2, 0.3, 1.0, 2.0, 3.0,
                                            degrees=False)
        dst = apply_transform(src, T_true)
        T = compute_extrinsic_svd(src, dst)
        assert np.allclose(T, T_true, atol=1e-9)

    def test_too_few_points(self):
        with pytest.raises(ValueError):
            compute_extrinsic_svd(np.zeros((2, 3)), np.zeros((2, 3)))

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            compute_extrinsic_svd(np.zeros((3, 3)), np.zeros((4, 3)))


class TestHandEye:
    def _gen_pair(self, X, motions_b):
        # A = X @ B @ inv(X)
        motions_a = []
        for B in motions_b:
            A = X @ B @ np.linalg.inv(X)
            motions_a.append(A)
        return motions_a

    def test_recovers_known_X(self):
        X = make_transform_from_euler(0.05, -0.1, 0.2,
                                      0.1, -0.2, 0.05, degrees=False)
        motions_b = [
            make_transform_from_euler(0.3, 0, 0, 0.5, 0, 0, degrees=False),
            make_transform_from_euler(0, 0.4, 0, 0, 0.3, 0, degrees=False),
            make_transform_from_euler(0, 0, 0.5, 0, 0, 0.2, degrees=False),
            make_transform_from_euler(0.2, 0.3, 0.1, 0.1, 0.2, 0.3, degrees=False),
        ]
        motions_a = self._gen_pair(X, motions_b)
        X_est = hand_eye_calibration(motions_a, motions_b)
        # Rotation should be close
        R_err = X_est[:3, :3].T @ X[:3, :3]
        # Trace of R_err should be near 3 if R_est == R
        assert (np.trace(R_err) - 3.0) > -0.01
        # Translation
        assert np.linalg.norm(X_est[:3, 3] - X[:3, 3]) < 0.05

    def test_too_few_pairs(self):
        with pytest.raises(ValueError):
            hand_eye_calibration([np.eye(4)], [np.eye(4)])

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            hand_eye_calibration([np.eye(4), np.eye(4)], [np.eye(4)])


class TestChain:
    def test_imu_to_camera_via_lidar(self):
        T_il = make_transform_from_euler(0, 0, 0.1, 0.5, 0, 0, degrees=False)
        T_lc = make_transform_from_euler(0, 0.2, 0, 0, 0.3, 0, degrees=False)
        ext_il = Extrinsic("imu", "lidar", T_il)
        ext_lc = Extrinsic("lidar", "camera", T_lc)
        ext_ic = chain_extrinsics(ext_il, ext_lc)
        assert ext_ic.src_frame == "imu"
        assert ext_ic.dst_frame == "camera"
        # Sanity: a point in imu transformed all the way to camera matches
        p_imu = np.array([[1.0, 2.0, 3.0]])
        p_via_chain = apply_extrinsic_to_points(ext_ic, p_imu)
        p_step = apply_extrinsic_to_points(
            ext_lc, apply_extrinsic_to_points(ext_il, p_imu)
        )
        assert np.allclose(p_via_chain, p_step, atol=1e-9)

    def test_mismatch_raises(self):
        a = Extrinsic("imu", "lidar", np.eye(4))
        b = Extrinsic("camera", "world", np.eye(4))
        with pytest.raises(ValueError):
            chain_extrinsics(a, b)


class TestRotToAxisAngle:
    def test_identity(self):
        assert np.allclose(_rot_to_axis_angle(np.eye(3)), 0.0)

    def test_90deg_about_z(self):
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        v = _rot_to_axis_angle(R)
        assert np.allclose(v / np.linalg.norm(v), [0, 0, 1])
        assert abs(np.linalg.norm(v) - math.pi / 2) < 1e-9
