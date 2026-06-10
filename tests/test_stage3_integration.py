"""
Stage 3 Integration Tests: VisualFrontend + Keyframe + Loop Closure + Pose Graph

Tests the wiring of loop closure components into VisualFrontend.
"""

import logging
import unittest
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np

from lidar_mapping.fusion.keyframe_selector import Keyframe
from lidar_mapping.fusion.pose_graph_backend import Pose4DOF
from lidar_mapping.fusion.pose_helpers import (
    extract_4dof_from_se3,
    pose_4dof_delta,
    se3_from_4dof,
)
from lidar_mapping.fusion.visual_frontend import VisualFrontend
from lidar_mapping.observability.state import FusionState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockSensorHub:
    """Mock sensor hub for testing."""

    def __init__(self):
        self.camera = Mock()
        self.lidar = Mock()
        self.camera_hz = 10.0


class TestStage3Integration(unittest.TestCase):
    """Test VisualFrontend integration with loop closure components."""

    def setUp(self):
        """Set up test fixtures."""
        self.hub = MockSensorHub()
        self.state = FusionState()

    def test_visual_frontend_initialization(self):
        """Test that VisualFrontend initializes with Stage 3 components."""
        frontend = VisualFrontend(self.hub, state=self.state, enable_loop_closure=True)
        
        self.assertIsNotNone(frontend._keyframe_selector)
        self.assertIsNotNone(frontend._loop_detector)
        self.assertIsNotNone(frontend._pose_graph)
        self.assertEqual(frontend.get_keyframe_count(), 0)
        self.assertEqual(frontend.get_pose_graph_size(), (0, 0))
        logger.info("✅ VisualFrontend initialization successful")

    def test_pose_helpers_4dof_conversion(self):
        """Test 4-DOF conversion utilities."""
        # Create a pose with translation and yaw rotation
        x, y, z, yaw = 1.0, 2.0, 3.0, np.pi / 4
        T_se3 = se3_from_4dof(x, y, z, yaw)
        
        # Verify it's a valid SE(3)
        self.assertEqual(T_se3.shape, (4, 4))
        np.testing.assert_array_almost_equal(T_se3[3, :], [0, 0, 0, 1])
        
        # Convert back
        pose_4dof = extract_4dof_from_se3(T_se3)
        self.assertEqual(pose_4dof.shape, (4,))
        
        # Check values (angle wrapping may differ slightly)
        np.testing.assert_almost_equal(pose_4dof[0], x, decimal=6)
        np.testing.assert_almost_equal(pose_4dof[1], y, decimal=6)
        np.testing.assert_almost_equal(pose_4dof[2], z, decimal=6)
        # Yaw should be close (within floating point)
        yaw_diff = abs(pose_4dof[3] - yaw)
        self.assertLess(min(yaw_diff, abs(yaw_diff - 2*np.pi)), 1e-6)
        logger.info("✅ 4-DOF conversion helpers working")

    def test_pose_delta_computation(self):
        """Test delta pose computation."""
        pose_a = np.array([0.0, 0.0, 0.0, 0.0])
        pose_b = np.array([1.0, 2.0, 0.5, np.pi / 4])
        
        delta = pose_4dof_delta(pose_a, pose_b)
        
        np.testing.assert_array_almost_equal(delta, pose_b - pose_a)
        logger.info("✅ Pose delta computation working")

    def test_keyframe_extraction_mock(self):
        """Test keyframe extraction with mocked camera data."""
        frontend = VisualFrontend(self.hub, state=self.state, enable_loop_closure=True)
        
        # Generate synthetic camera frame
        frame_bgr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Detect features
        orb = cv2.ORB_create(nfeatures=500)
        keypoints, descriptors = orb.detectAndCompute(frame_gray, None)
        
        self.assertIsNotNone(keypoints)
        self.assertIsNotNone(descriptors)
        self.assertGreater(len(keypoints), 0)
        self.assertEqual(descriptors.dtype, np.uint8)
        logger.info(f"✅ Synthetic keyframe generation: {len(keypoints)} features")

    def test_pose_graph_accumulation(self):
        """Test that pose graph accumulates keyframes correctly."""
        frontend = VisualFrontend(self.hub, state=self.state, enable_loop_closure=True)
        
        # Simulate sequence of poses moving in a line
        pose_4dof_1 = np.array([0.0, 0.0, 0.0, 0.0])
        pose_4dof_2 = np.array([0.1, 0.0, 0.0, 0.0])
        pose_4dof_3 = np.array([0.2, 0.0, 0.0, 0.0])
        
        # Manually add to pose graph
        frontend._pose_graph.add_pose(0, Pose4DOF(*pose_4dof_1))
        frontend._pose_graph.add_pose(1, Pose4DOF(*pose_4dof_2))
        frontend._pose_graph.add_pose(2, Pose4DOF(*pose_4dof_3))
        
        self.assertEqual(len(frontend._pose_graph.poses), 3)
        
        # Add odometry factors
        info = np.eye(4) * 10.0
        delta_1_2 = pose_4dof_delta(pose_4dof_1, pose_4dof_2)
        delta_2_3 = pose_4dof_delta(pose_4dof_2, pose_4dof_3)
        
        frontend._pose_graph.add_factor(0, 1, delta_1_2, info)
        frontend._pose_graph.add_factor(1, 2, delta_2_3, info)
        
        self.assertEqual(len(frontend._pose_graph.factors), 2)
        logger.info("✅ Pose graph accumulation working")

    def test_pose_graph_optimization_simple(self):
        """Test pose graph optimization on simple trajectory."""
        frontend = VisualFrontend(self.hub, state=self.state, enable_loop_closure=True)
        
        # Create a simple trajectory: 3 poses in a line with slight noise
        poses = [
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([0.1, 0.01, 0.0, 0.05]),  # Slightly noisy
            np.array([0.2, -0.01, 0.0, -0.05]),  # Slightly noisy
        ]
        
        for pose_id, pose in enumerate(poses):
            frontend._pose_graph.add_pose(pose_id, Pose4DOF(*pose))
        
        # Add odometry factors with some error
        info = np.eye(4) * 10.0
        delta_1 = pose_4dof_delta(poses[0], poses[1])
        delta_2 = pose_4dof_delta(poses[1], poses[2])
        
        frontend._pose_graph.add_factor(0, 1, delta_1, info)
        frontend._pose_graph.add_factor(1, 2, delta_2, info)
        
        # Optimize
        frontend._pose_graph.optimize(max_iterations=5)
        
        # Check that we have a trajectory
        traj = frontend.get_pose_graph_trajectory()
        self.assertIsNotNone(traj)
        self.assertEqual(traj.shape[0], 3)  # 3 poses
        logger.info("✅ Pose graph optimization working")

    def test_pose_graph_loop_closure(self):
        """Test pose graph with loop closure constraint."""
        frontend = VisualFrontend(self.hub, state=self.state, enable_loop_closure=True)
        
        # Create a loop: 4 poses returning to starting location
        # Start → Forward → Right → Left → Start
        poses = [
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([0.5, 0.0, 0.0, 0.0]),
            np.array([0.5, 0.5, 0.0, np.pi/2]),
            np.array([0.0, 0.5, 0.0, np.pi]),
        ]
        
        for pose_id, pose in enumerate(poses):
            frontend._pose_graph.add_pose(pose_id, Pose4DOF(*pose))
        
        # Add odometry factors
        info_odom = np.eye(4) * 10.0
        for i in range(len(poses) - 1):
            delta = pose_4dof_delta(poses[i], poses[i+1])
            frontend._pose_graph.add_factor(i, i+1, delta, info_odom)
        
        # Add loop closure: pose 3 back to pose 0
        # Should be close to identity (small drift)
        loop_delta = pose_4dof_delta(poses[3], poses[0])
        info_loop = np.eye(4) * 5.0  # Lower confidence than odometry
        frontend._pose_graph.add_factor(3, 0, loop_delta, info_loop)
        
        self.assertEqual(len(frontend._pose_graph.poses), 4)
        self.assertEqual(len(frontend._pose_graph.factors), 4)
        
        # Optimize
        frontend._pose_graph.optimize(max_iterations=10)
        
        # Check trajectory
        traj = frontend.get_pose_graph_trajectory()
        self.assertIsNotNone(traj)
        self.assertEqual(traj.shape[0], 4)
        logger.info("✅ Loop closure in pose graph working")

    def test_frontend_methods(self):
        """Test VisualFrontend query methods."""
        frontend = VisualFrontend(self.hub, state=self.state, enable_loop_closure=True)
        
        # Initially empty
        self.assertEqual(frontend.get_keyframe_count(), 0)
        self.assertEqual(frontend.get_pose_graph_size(), (0, 0))
        
        # Add some data manually
        pose = Pose4DOF(0.0, 0.0, 0.0, 0.0)
        frontend._pose_graph.add_pose(0, pose)
        
        self.assertEqual(frontend.get_pose_graph_size()[0], 1)
        logger.info("✅ Frontend methods working")

    def test_keyframe_storage(self):
        """Test keyframe storage in VisualFrontend."""
        frontend = VisualFrontend(self.hub, state=self.state, enable_loop_closure=True)
        
        # Create mock keyframe
        pose = np.eye(4)
        pose[:3, 3] = [1.0, 2.0, 3.0]
        
        descriptors = np.random.randint(0, 2, (100, 32), dtype=np.uint8)
        keypoints = np.random.rand(100, 2).astype(np.float32)
        
        kf = Keyframe(
            keyframe_id=0,
            pose=pose,
            timestamp=1.0,
            descriptors=descriptors,
            keypoints=keypoints,
            image_shape=(480, 640),
        )
        
        # Store in frontend
        frontend._keyframes[0] = kf
        
        # Retrieve
        retrieved = frontend._keyframes[0]
        self.assertEqual(retrieved.keyframe_id, 0)
        np.testing.assert_array_equal(retrieved.pose, pose)
        self.assertEqual(retrieved.timestamp, 1.0)
        logger.info("✅ Keyframe storage working")

    def test_loop_closure_disabled(self):
        """Test VisualFrontend with loop closure disabled."""
        frontend = VisualFrontend(
            self.hub, state=self.state, enable_loop_closure=False
        )
        
        self.assertFalse(frontend.enable_loop_closure)
        self.assertEqual(frontend.get_keyframe_count(), 0)
        logger.info("✅ Loop closure disable flag working")


class TestPose4DOFRoundtrip(unittest.TestCase):
    """Test round-trip conversions between SE(3) and 4-DOF."""

    def test_multiple_poses(self):
        """Test round-trip for various poses."""
        test_cases = [
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 2.0, 3.0, np.pi / 4),
            (-1.0, -2.0, -3.0, -np.pi / 3),
            (10.0, -10.0, 5.0, np.pi),
            (0.1, 0.2, 0.3, 0.05),
        ]
        
        for x, y, z, yaw in test_cases:
            # Convert to SE(3)
            T = se3_from_4dof(x, y, z, yaw)
            
            # Convert back
            pose_4dof = extract_4dof_from_se3(T)
            
            # Check
            np.testing.assert_almost_equal(pose_4dof[0], x, decimal=10)
            np.testing.assert_almost_equal(pose_4dof[1], y, decimal=10)
            np.testing.assert_almost_equal(pose_4dof[2], z, decimal=10)
            
            # Yaw wrapping
            yaw_recovered = pose_4dof[3]
            yaw_diff = abs(yaw_recovered - yaw)
            # Allow for 2π wrapping
            yaw_diff = min(yaw_diff, abs(yaw_diff - 2*np.pi))
            self.assertLess(yaw_diff, 1e-10)
        
        logger.info(f"✅ Round-trip tests: {len(test_cases)} poses")


if __name__ == "__main__":
    unittest.main()
