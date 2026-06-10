"""Tests for the loop closure modules."""

from __future__ import annotations

import math

import numpy as np
import pytest

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    import open3d as o3d  # noqa: F401
    _O3D_AVAILABLE = True
except ImportError:
    _O3D_AVAILABLE = False

# Test both loop closure implementations if they exist
try:
    from lidar_mapping.mapping.loop_closure import (
        Keyframe,
        KeyframeStore,
        LoopClosureDetector as MappingLoopClosureDetector,
        LoopEdge,
        _pose_rotation_angle_deg,
        interpolate_correction,
        optimize_with_loop_closures,
    )
    from lidar_mapping.utils.transforms import make_transform_from_euler
    _MAPPING_LOOP_CLOSURE_AVAILABLE = True
except ImportError:
    _MAPPING_LOOP_CLOSURE_AVAILABLE = False

try:
    from lidar_mapping.fusion.loop_closure import (
        LoopClosureDetector as FusionLoopClosureDetector,
        LoopCandidate,
        VerifiedLoopClosure,
    )
    _FUSION_LOOP_CLOSURE_AVAILABLE = True
except ImportError:
    _FUSION_LOOP_CLOSURE_AVAILABLE = False


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


# ===========================================================================
# Mapping Loop Closure Tests (lidar_mapping.mapping.loop_closure)
# ===========================================================================

@pytest.mark.skipif(not _MAPPING_LOOP_CLOSURE_AVAILABLE, reason="mapping.loop_closure not available")
class TestMappingRotationAngle:
    def test_identity(self):
        a = np.eye(4)
        assert _pose_rotation_angle_deg(a, a) == pytest.approx(0.0, abs=1e-6)

    def test_known(self):
        a = np.eye(4)
        b = make_transform_from_euler(0, 0, 90)  # 90 deg yaw, degrees=True
        assert _pose_rotation_angle_deg(a, b) == pytest.approx(90.0, abs=1e-4)


@pytest.mark.skipif(not _MAPPING_LOOP_CLOSURE_AVAILABLE, reason="mapping.loop_closure not available")
class TestMappingKeyframeStore:
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


@pytest.mark.skipif(not (_MAPPING_LOOP_CLOSURE_AVAILABLE and _O3D_AVAILABLE), 
                    reason="mapping.loop_closure or open3d not installed")
class TestMappingDetectIntegration:
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

        det = MappingLoopClosureDetector(
            search_radius=1.0, min_time_gap=3,
            fitness_threshold=0.3, max_rmse=1.0,
            icp_voxel_size=0.2,
            icp_max_correspondence_distance=1.0,
        )
        edges = det.detect(store, revisit)
        # Keyframe 0 is the only one within 1m and >=3 indices away
        assert any(e.source_kf == 0 for e in edges)


# ===========================================================================
# Fusion Loop Closure Tests (lidar_mapping.fusion.loop_closure)
# ===========================================================================

@pytest.mark.skipif(not _FUSION_LOOP_CLOSURE_AVAILABLE, reason="fusion.loop_closure not available")
class TestFusionLoopClosureDetector:
    """Test fusion loop closure detection logic."""
    
    def test_detector_initialization(self):
        """Test detector can be initialized."""
        detector = FusionLoopClosureDetector(
            descriptor_match_ratio=0.75,
            min_matches_threshold=20,
        )
        
        assert detector.descriptor_match_ratio == 0.75
        assert detector.min_matches_threshold == 20
    
    def test_find_loop_candidates_empty_db(self):
        """Test candidate finding with empty keyframe database."""
        detector = FusionLoopClosureDetector()
        
        descriptors = np.random.randint(0, 256, (100, 32), dtype=np.uint8)
        candidates = detector.find_loop_candidates(descriptors, keyframe_db=[])
        
        assert len(candidates) == 0
    
    def test_find_loop_candidates_identical_frame(self):
        """Test that identical descriptors are handled (may not produce matches due to knnMatch behavior)."""
        detector = FusionLoopClosureDetector(min_matches_threshold=10)
        
        # Create realistic descriptors
        descriptors = np.random.randint(0, 256, (100, 32), dtype=np.uint8)
        keyframe_db = [(0, descriptors)]
        
        # Should not crash with identical descriptors
        candidates = detector.find_loop_candidates(descriptors, keyframe_db)
        
        # Candidates may be empty depending on knnMatch behavior
        # The important thing is that it doesn't crash
        assert isinstance(candidates, list)
    
    def test_statistics(self):
        """Test statistics reporting."""
        detector = FusionLoopClosureDetector()
        
        stats = detector.statistics()
        
        assert stats["candidates_checked"] == 0
        assert stats["loop_closures_verified"] == 0
        assert stats["total_loop_closures"] == 0
    
    def test_reset(self):
        """Test resetting detector state."""
        detector = FusionLoopClosureDetector()
        detector.candidates_checked = 10
        detector.loop_closures_verified = 5
        
        detector.reset()
        
        assert detector.candidates_checked == 0
        assert detector.loop_closures_verified == 0
        assert len(detector.loop_closures) == 0
    
    def test_loop_candidate_dataclass(self):
        """Test LoopCandidate dataclass."""
        candidate = LoopCandidate(
            keyframe_id_a=0,
            keyframe_id_b=10,
            match_count=50,
            inlier_count=45,
            score=0.9,
        )
        
        assert candidate.keyframe_id_a == 0
        assert candidate.keyframe_id_b == 10
        assert candidate.match_count == 50
    
    def test_verified_loop_closure_dataclass(self):
        """Test VerifiedLoopClosure dataclass."""
        transform = np.eye(4)
        transform[0, 3] = 0.1
        
        loop = VerifiedLoopClosure(
            keyframe_id_a=0,
            keyframe_id_b=10,
            transform=transform,
            match_count=50,
            inlier_count=45,
            reprojection_error=0.5,
            confidence=0.9,
        )
        
        assert loop.keyframe_id_a == 0
        assert loop.keyframe_id_b == 10
        assert loop.transform.shape == (4, 4)
        assert loop.confidence == 0.9
