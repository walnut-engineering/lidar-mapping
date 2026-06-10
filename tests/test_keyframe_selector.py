"""
Unit tests for keyframe selector.
"""

import numpy as np
import pytest

from lidar_mapping.fusion.keyframe_selector import KeyframeSelector, Keyframe


class TestKeyframeSelector:
    """Test keyframe selection logic."""
    
    def test_first_frame_always_keyframe(self):
        """First frame should always be selected as keyframe."""
        selector = KeyframeSelector(motion_threshold_m=0.1)
        pose = np.eye(4)
        
        assert selector.should_be_keyframe(pose) is True
    
    def test_small_translation_no_keyframe(self):
        """Small translation should not trigger keyframe."""
        selector = KeyframeSelector(motion_threshold_m=0.1)
        
        # Add first keyframe
        pose1 = np.eye(4)
        selector._last_keyframe_pose = pose1
        
        # Small translation (0.05 m < 0.1 m threshold)
        pose2 = np.eye(4)
        pose2[0, 3] = 0.05
        
        assert selector.should_be_keyframe(pose2) is False
    
    def test_large_translation_creates_keyframe(self):
        """Large translation should trigger keyframe."""
        selector = KeyframeSelector(motion_threshold_m=0.1)
        
        pose1 = np.eye(4)
        selector._last_keyframe_pose = pose1
        
        pose2 = np.eye(4)
        pose2[0, 3] = 0.15  # 0.15 m > 0.1 m threshold
        
        assert selector.should_be_keyframe(pose2) is True
    
    def test_rotation_creates_keyframe(self):
        """Rotation beyond threshold should trigger keyframe."""
        selector = KeyframeSelector(
            motion_threshold_m=0.1,
            angular_threshold_deg=5.0,
        )
        
        pose1 = np.eye(4)
        selector._last_keyframe_pose = pose1
        
        # Rotation of 10 degrees around z-axis
        angle = np.radians(10.0)
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ])
        pose2 = np.eye(4)
        pose2[:3, :3] = R
        
        assert selector.should_be_keyframe(pose2) is True
    
    def test_force_keyframe(self):
        """Force keyframe flag should override thresholds."""
        selector = KeyframeSelector(motion_threshold_m=10.0)
        
        pose1 = np.eye(4)
        selector._last_keyframe_pose = pose1
        
        pose2 = np.eye(4)  # Identity, would normally fail
        
        assert selector.should_be_keyframe(pose2, force_keyframe=True) is True
    
    def test_add_keyframe(self):
        """Test adding a keyframe to selector."""
        selector = KeyframeSelector()
        
        pose = np.eye(4)
        descriptors = np.random.randint(0, 256, (100, 32), dtype=np.uint8)
        
        kf = selector.add_keyframe(
            pose=pose,
            descriptors=descriptors,
            timestamp=1.0,
        )
        
        assert kf.keyframe_id == 0
        assert len(selector.keyframes) == 1
        assert selector.keyframe_count == 1
        np.testing.assert_array_equal(kf.pose, pose)
    
    def test_sequential_keyframes(self):
        """Test adding multiple keyframes in sequence."""
        selector = KeyframeSelector(motion_threshold_m=0.05)
        
        for i in range(5):
            pose = np.eye(4)
            pose[0, 3] = 0.1 * i  # Progressive movement
            descriptors = np.random.randint(0, 256, (50 + i * 10, 32), dtype=np.uint8)
            
            if selector.should_be_keyframe(pose):
                kf = selector.add_keyframe(pose, descriptors, timestamp=float(i))
                assert kf.keyframe_id == selector.keyframe_count - 1
        
        assert len(selector.keyframes) > 1
    
    def test_keyframe_retrieval(self):
        """Test retrieving keyframes by ID."""
        selector = KeyframeSelector()
        
        pose = np.eye(4)
        descriptors = np.random.randint(0, 256, (100, 32), dtype=np.uint8)
        kf1 = selector.add_keyframe(pose, descriptors, timestamp=0.0)
        
        pose[0, 3] = 0.2
        kf2 = selector.add_keyframe(pose, descriptors, timestamp=1.0)
        
        retrieved = selector.get_keyframe(kf1.keyframe_id)
        assert retrieved is not None
        assert retrieved.keyframe_id == kf1.keyframe_id
        
        retrieved = selector.get_keyframe(kf2.keyframe_id)
        assert retrieved is not None
        assert retrieved.keyframe_id == kf2.keyframe_id
    
    def test_get_recent_keyframes(self):
        """Test retrieving recent keyframes."""
        selector = KeyframeSelector()
        
        for i in range(10):
            pose = np.eye(4)
            pose[0, 3] = 0.1 * i
            descriptors = np.random.randint(0, 256, (50, 32), dtype=np.uint8)
            selector.add_keyframe(pose, descriptors, timestamp=float(i))
        
        recent = selector.get_recent_keyframes(count=3)
        assert len(recent) == 3
        assert recent[-1].keyframe_id == 9
    
    def test_reset(self):
        """Test resetting the selector."""
        selector = KeyframeSelector()
        
        pose = np.eye(4)
        descriptors = np.random.randint(0, 256, (100, 32), dtype=np.uint8)
        selector.add_keyframe(pose, descriptors, timestamp=0.0)
        
        assert len(selector.keyframes) == 1
        
        selector.reset()
        assert len(selector.keyframes) == 0
        assert selector.keyframe_count == 0
    
    def test_statistics(self):
        """Test statistics reporting."""
        selector = KeyframeSelector(
            motion_threshold_m=0.1,
            angular_threshold_deg=5.0,
        )
        
        stats = selector.statistics()
        
        assert stats["total_keyframes"] == 0
        assert stats["next_keyframe_id"] == 0
        assert stats["motion_threshold_m"] == 0.1
        assert abs(stats["angular_threshold_deg"] - 5.0) < 0.01
    
    def test_keyframe_descriptor_validation(self):
        """Test that invalid descriptors raise error."""
        selector = KeyframeSelector()
        
        pose = np.eye(4)
        descriptors_wrong_shape = np.random.randint(0, 256, (100, 16), dtype=np.uint8)
        
        with pytest.raises(ValueError):
            selector.add_keyframe(pose, descriptors_wrong_shape, timestamp=0.0)
    
    def test_motion_combining_translation_and_rotation(self):
        """Test that combined translation and rotation work."""
        selector = KeyframeSelector(
            motion_threshold_m=0.05,
            angular_threshold_deg=5.0,
        )
        
        pose1 = np.eye(4)
        selector._last_keyframe_pose = pose1
        
        # Combined: small translation + small rotation
        pose2 = np.eye(4)
        pose2[0, 3] = 0.02  # 2 cm
        angle = np.radians(2.0)  # 2 degrees
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ])
        pose2[:3, :3] = R
        
        # Neither alone exceeds threshold, but combined should not trigger
        assert selector.should_be_keyframe(pose2) is False
        
        # Now large translation + small rotation
        pose3 = np.eye(4)
        pose3[0, 3] = 0.1  # 10 cm > 5 cm threshold
        pose3[:3, :3] = R
        
        assert selector.should_be_keyframe(pose3) is True
