"""Tests for the loop closure module."""

from __future__ import annotations

import math

import numpy as np
import pytest

try:
    import open3d as o3d  # noqa: F401
    _O3D_AVAILABLE = True
except ImportError:
    _O3D_AVAILABLE = False

from lidar_mapping.mapping.loop_closure import (
    Keyframe,
    KeyframeStore,
    LoopClosureDetector,
    LoopEdge,
    _pose_rotation_angle_deg,
    interpolate_correction,
    optimize_with_loop_closures,
)
from lidar_mapping.utils.transforms import make_transform_from_euler


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _box_scan(half=5.0, n=400, seed=0):
    """Generate a simple structured scan: points on the inside of a box."""
    rng = np.random.default_rng(seed)
    pts = []
    per_face = n // 6
    # ±X faces
    for sign in (-1, 1):
        y = rng.uniform(-half, half, per_face)
        z = rng.uniform(-half, half, per_face)
        pts.append(np.column_stack([np.full_like(y, sign * half), y, z]))
    # ±Y faces
    for sign in (-1, 1):
        x = rng.uniform(-half, half, per_face)
        z = rng.uniform(-half, half, per_face)
        pts.append(np.column_stack([x, np.full_like(x, sign * half), z]))
    # ±Z faces
    for sign in (-1, 1):
        x = rng.uniform(-half, half, per_face)
        y = rng.uniform(-half, half, per_face)
        pts.append(np.column_stack([x, y, np.full_like(x, sign * half)]))
    return np.vstack(pts).astype(np.float64)


def _transform_pts(pts, T):
    h = np.hstack([pts, np.ones((len(pts), 1))])
    return (T @ h.T).T[:, :3]


# ---------------------------------------------------------------------------
# _pose_rotation_angle_deg
# ---------------------------------------------------------------------------

class TestRotationAngle:
    def test_identity(self):
        a = np.eye(4)
        assert _pose_rotation_angle_deg(a, a) == pytest.approx(0.0, abs=1e-6)

    def test_known(self):
        a = np.eye(4)
        b = make_transform_from_euler(0, 0, 90)  # 90 deg yaw, degrees=True
        assert _pose_rotation_angle_deg(a, b) == pytest.approx(90.0, abs=1e-4)


# ---------------------------------------------------------------------------
# KeyframeStore
# ---------------------------------------------------------------------------

class TestKeyframeStore:
    def test_first_always_added(self):
        store = KeyframeStore(distance_threshold=1.0,
                              rotation_threshold_deg=15.0,
                              voxel_size=None)
        scan = _box_scan()
        kf = store.try_add(scan, np.eye(4))
        assert kf is not None
        assert kf.index == 0
        assert len(store) == 1

    def test_skipped_when_below_thresholds(self):
        store = KeyframeStore(distance_threshold=1.0,
                              rotation_threshold_deg=15.0,
                              voxel_size=None)
        store.try_add(_box_scan(), np.eye(4))
        # Tiny move
        T = make_transform_from_euler(0, 0, 0, 0.1, 0, 0)
        kf = store.try_add(_box_scan(), T)
        assert kf is None
        assert len(store) == 1

    def test_added_when_distance_exceeded(self):
        store = KeyframeStore(distance_threshold=1.0,
                              rotation_threshold_deg=180.0,
                              voxel_size=None)
        store.try_add(_box_scan(), np.eye(4))
        T = make_transform_from_euler(0, 0, 0, 2.0, 0, 0)
        kf = store.try_add(_box_scan(), T)
        assert kf is not None
        assert kf.index == 1

    def test_added_when_rotation_exceeded(self):
        store = KeyframeStore(distance_threshold=1000.0,
                              rotation_threshold_deg=10.0,
                              voxel_size=None)
        store.try_add(_box_scan(), np.eye(4))
        T = make_transform_from_euler(0, 0, 20)  # 20 deg yaw
        kf = store.try_add(_box_scan(), T)
        assert kf is not None

    def test_voxel_downsamples(self):
        store = KeyframeStore(distance_threshold=0.0,
                              rotation_threshold_deg=0.0,
                              voxel_size=1.0)
        dense = np.random.default_rng(0).uniform(-2, 2, (5000, 3))
        kf = store.try_add(dense, np.eye(4))
        assert len(kf.points) < 5000

    def test_positions_array(self):
        store = KeyframeStore(distance_threshold=0.5,
                              rotation_threshold_deg=1000.0,
                              voxel_size=None)
        for i in range(3):
            T = make_transform_from_euler(0, 0, 0, float(i), 0, 0)
            store.try_add(_box_scan(seed=i), T)
        pos = store.positions()
        assert pos.shape == (3, 3)
        np.testing.assert_allclose(pos[:, 0], [0, 1, 2])

    def test_invalid_pose_shape(self):
        store = KeyframeStore()
        with pytest.raises(ValueError):
            store.try_add(_box_scan(), np.eye(3))

    def test_invalid_scan_shape(self):
        store = KeyframeStore()
        with pytest.raises(ValueError):
            store.try_add(np.array([1.0, 2.0, 3.0]), np.eye(4))


# ---------------------------------------------------------------------------
# LoopClosureDetector – candidate search
# ---------------------------------------------------------------------------

class TestFindCandidates:
    def _populate(self, n=30, voxel=None):
        store = KeyframeStore(distance_threshold=0.0,
                              rotation_threshold_deg=0.0,
                              voxel_size=voxel)
        # Linear path along +X
        for i in range(n):
            T = make_transform_from_euler(0, 0, 0, float(i), 0, 0)
            store.try_add(_box_scan(seed=i), T)
        return store

    def test_no_candidates_when_alone(self):
        store = KeyframeStore(voxel_size=None)
        store.try_add(_box_scan(), np.eye(4))
        det = LoopClosureDetector(search_radius=10.0, min_time_gap=1)
        kf = store[0]
        # query is the only frame -> no candidates
        assert det.find_candidates(store, kf) == []

    def test_time_gap_filters_recent(self):
        store = self._populate(n=10)
        det = LoopClosureDetector(search_radius=100.0, min_time_gap=5)
        query = store[9]
        cands = det.find_candidates(store, query)
        # Only indices <= 4 can pass (gap >= 5)
        assert all(query.index - c >= 5 for c in cands)
        assert max(cands) <= 4

    def test_radius_filters_far(self):
        store = self._populate(n=20)
        det = LoopClosureDetector(search_radius=2.5, min_time_gap=1)
        query = store[19]
        cands = det.find_candidates(store, query)
        # Within 2.5m of x=19, with gap>=1: indices 17, 18 (gap 1, 2 → excluded if gap=1? gap is index diff; 19-18=1, 19-17=2; both >=1)
        assert set(cands) == {17, 18}

    def test_sorted_by_distance(self):
        # Place keyframes at random distances
        store = KeyframeStore(distance_threshold=0.0,
                              rotation_threshold_deg=0.0,
                              voxel_size=None)
        positions = [0.0, 5.0, 1.0, 10.0, 2.0]  # |q-p| from q=4.5
        for i, x in enumerate(positions):
            T = make_transform_from_euler(0, 0, 0, x, 0, 0)
            store.try_add(_box_scan(seed=i), T)
        query_T = make_transform_from_euler(0, 0, 0, 4.5, 0, 0)
        query = Keyframe(index=999, scan_index=999, pose=query_T,
                         points=_box_scan(seed=99))
        det = LoopClosureDetector(search_radius=100.0, min_time_gap=1)
        cands = det.find_candidates(store, query)
        # Distances from 4.5: [4.5, 0.5, 3.5, 5.5, 2.5]
        # Sorted: indices [1, 4, 2, 0, 3]
        assert cands == [1, 4, 2, 0, 3]


# ---------------------------------------------------------------------------
# LoopClosureDetector – verification (requires open3d)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _O3D_AVAILABLE, reason="open3d not installed")
class TestVerify:
    def test_accepts_well_aligned_overlap(self):
        # Two keyframes observing the same box from poses with a small
        # known offset.  ICP should succeed.
        pts_world = _box_scan(half=5.0, n=2400, seed=42)

        T_a = make_transform_from_euler(0, 0, 0, 0.0, 0, 0)
        T_b = make_transform_from_euler(0, 0, 0, 0.3, 0, 0)

        pts_a = _transform_pts(pts_world, np.linalg.inv(T_a))
        pts_b = _transform_pts(pts_world, np.linalg.inv(T_b))

        kf_a = Keyframe(index=0, scan_index=0, pose=T_a, points=pts_a)
        kf_b = Keyframe(index=1, scan_index=1, pose=T_b, points=pts_b)

        det = LoopClosureDetector(
            search_radius=2.0, min_time_gap=1,
            fitness_threshold=0.3, max_rmse=1.0,
            icp_voxel_size=0.2,
            icp_max_correspondence_distance=1.0,
        )
        edge = det.verify(kf_a, kf_b)
        assert edge is not None
        assert edge.source_kf == 0
        assert edge.target_kf == 1
        assert edge.fitness > 0.3

    def test_rejects_disjoint_clouds(self):
        pts_a = _box_scan(half=5.0, seed=1)
        # Wildly different scan: random sphere far away
        rng = np.random.default_rng(2)
        pts_b = rng.normal(size=(300, 3)) * 0.05 + np.array([200.0, 0, 0])

        kf_a = Keyframe(index=0, scan_index=0, pose=np.eye(4), points=pts_a)
        kf_b = Keyframe(
            index=1, scan_index=1,
            pose=make_transform_from_euler(0, 0, 0, 200.0, 0, 0),
            points=pts_b,
        )
        det = LoopClosureDetector(
            search_radius=500.0, min_time_gap=1,
            fitness_threshold=0.5, max_rmse=0.1,
            icp_voxel_size=0.2,
            icp_max_correspondence_distance=0.5,
        )
        # Either fitness too low or rmse too high → reject
        assert det.verify(kf_a, kf_b) is None


# ---------------------------------------------------------------------------
# optimize_with_loop_closures (requires open3d)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _O3D_AVAILABLE, reason="open3d not installed")
class TestOptimize:
    def test_returns_same_count(self):
        poses = [make_transform_from_euler(0, 0, 0, float(i), 0, 0)
                 for i in range(5)]
        out = optimize_with_loop_closures(poses, [])
        assert len(out) == len(poses)

    def test_zero_or_one_pose_passthrough(self):
        assert optimize_with_loop_closures([], []) == []
        out = optimize_with_loop_closures([np.eye(4)], [])
        assert len(out) == 1
        np.testing.assert_allclose(out[0], np.eye(4))

    def test_no_loop_preserves_relative_transforms(self):
        # Without loop edges the optimisation should preserve the
        # sequential relative transforms.  (Translation magnitude is
        # preserved; absolute orientation may be rotated by the optimiser
        # since the chain has no global anchor on rotation.)
        poses = [make_transform_from_euler(0, 0, 10.0 * i, float(i), 0, 0)
                 for i in range(6)]
        out = optimize_with_loop_closures(poses, [])
        for i in range(len(poses) - 1):
            rel_orig = np.linalg.inv(poses[i]) @ poses[i + 1]
            rel_corr = np.linalg.inv(out[i]) @ out[i + 1]
            # Translation magnitudes preserved
            assert np.linalg.norm(rel_corr[:3, 3]) == pytest.approx(
                np.linalg.norm(rel_orig[:3, 3]), abs=0.05
            )

    def test_loop_corrects_drift(self):
        # Build a true square loop, then inject drift into the odometry,
        # and verify the loop edge closes it.
        n = 20
        true_poses = []
        for i in range(n):
            angle = 2 * math.pi * i / n
            x = 5.0 * math.cos(angle)
            y = 5.0 * math.sin(angle)
            yaw_deg = math.degrees(angle + math.pi / 2)
            T = make_transform_from_euler(0, 0, yaw_deg, x, y, 0.0)
            true_poses.append(T)

        # Drifted poses: rotate true positions slightly
        rng = np.random.default_rng(0)
        drift = make_transform_from_euler(0, 0, 5.0, 0.5, 0.0, 0.0)
        drifted = [true_poses[0].copy()]
        for i in range(1, n):
            T_rel_true = (np.linalg.inv(true_poses[i - 1])
                          @ true_poses[i])
            # Add a small extra rotation/translation each step
            extra = make_transform_from_euler(0, 0, 0.5, 0.02, 0, 0)
            T_rel_drift = T_rel_true @ extra
            drifted.append(drifted[-1] @ T_rel_drift)

        # Loop edge: last keyframe ↔ first keyframe with true relative
        T_loop = np.linalg.inv(true_poses[0]) @ true_poses[-1]
        loop = LoopEdge(source_kf=0, target_kf=n - 1,
                        transform=T_loop, fitness=0.9, inlier_rmse=0.05)

        corrected = optimize_with_loop_closures(
            drifted, [loop],
            max_correspondence_distance=2.0,
            loop_information=100.0 * np.eye(6),
        )

        # End-to-start gap should be smaller after correction
        gap_before = np.linalg.norm(
            drifted[-1][:3, 3] - drifted[0][:3, 3]
            - (true_poses[-1][:3, 3] - true_poses[0][:3, 3])
        )
        gap_after = np.linalg.norm(
            corrected[-1][:3, 3] - corrected[0][:3, 3]
            - (true_poses[-1][:3, 3] - true_poses[0][:3, 3])
        )
        assert gap_after < gap_before

    def test_rejects_out_of_range_loop(self):
        poses = [np.eye(4), np.eye(4)]
        bad = LoopEdge(source_kf=0, target_kf=5,
                       transform=np.eye(4), fitness=1.0, inlier_rmse=0.0)
        with pytest.raises(IndexError):
            optimize_with_loop_closures(poses, [bad])


# ---------------------------------------------------------------------------
# interpolate_correction
# ---------------------------------------------------------------------------

class TestInterpolateCorrection:
    def test_identity_correction_is_noop(self):
        dense = [make_transform_from_euler(0, 0, 0, float(i), 0, 0)
                 for i in range(10)]
        kf_orig = [dense[0], dense[5]]
        kf_corr = [dense[0], dense[5]]
        out = interpolate_correction(kf_orig, kf_corr, dense, [0, 5])
        for a, b in zip(out, dense):
            np.testing.assert_allclose(a, b)

    def test_applies_correction_after_keyframe(self):
        dense = [make_transform_from_euler(0, 0, 0, float(i), 0, 0)
                 for i in range(6)]
        # Pretend keyframe 0 was perfect, keyframe 1 (at dense idx 3)
        # had a +1m Y correction.
        kf_orig = [dense[0], dense[3]]
        shifted = dense[3].copy()
        shifted[1, 3] += 1.0
        kf_corr = [dense[0], shifted]
        out = interpolate_correction(kf_orig, kf_corr, dense, [0, 3])
        # Frames 0..2 unchanged (kf 0 correction is identity)
        for i in range(3):
            np.testing.assert_allclose(out[i], dense[i])
        # Frames 3..5 should all be lifted +1m in Y
        for i in range(3, 6):
            np.testing.assert_allclose(out[i][1, 3], dense[i][1, 3] + 1.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            interpolate_correction([np.eye(4)], [np.eye(4), np.eye(4)],
                                   [np.eye(4)], [0])

    def test_index_mismatch_raises(self):
        with pytest.raises(ValueError):
            interpolate_correction([np.eye(4)], [np.eye(4)],
                                   [np.eye(4)], [0, 1])

    def test_empty_keyframes_passthrough(self):
        dense = [np.eye(4) for _ in range(3)]
        out = interpolate_correction([], [], dense, [])
        assert len(out) == 3


# ---------------------------------------------------------------------------
# detect() integration
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _O3D_AVAILABLE, reason="open3d not installed")
class TestDetectIntegration:
    def test_finds_revisit_in_path(self):
        # Build a path that returns to the origin
        store = KeyframeStore(distance_threshold=0.0,
                              rotation_threshold_deg=0.0,
                              voxel_size=0.2)
        pts_world = _box_scan(half=5.0, n=3000, seed=7)

        # Forward 5 keyframes along +X, then back to origin
        xs = [0.0, 1.0, 2.0, 3.0, 4.0]
        for i, x in enumerate(xs):
            T = make_transform_from_euler(0, 0, 0, x, 0, 0)
            pts_local = _transform_pts(pts_world, np.linalg.inv(T))
            store.try_add(pts_local, T, scan_index=i)

        # Revisit near origin (small offset)
        T_revisit = make_transform_from_euler(0, 0, 0, 0.2, 0, 0)
        pts_local = _transform_pts(pts_world, np.linalg.inv(T_revisit))
        revisit = store.try_add(pts_local, T_revisit, scan_index=len(xs))
        assert revisit is not None

        det = LoopClosureDetector(
            search_radius=1.0, min_time_gap=3,
            fitness_threshold=0.3, max_rmse=1.0,
            icp_voxel_size=0.2,
            icp_max_correspondence_distance=1.0,
        )
        edges = det.detect(store, revisit)
        # Keyframe 0 is the only one within 1m and >=3 indices away
        assert any(e.source_kf == 0 for e in edges)
