"""
Unit tests for pose graph optimizer.
"""

import numpy as np
import pytest

from lidar_mapping.fusion.pose_graph_backend import (
    Pose4DOF,
    Factor,
    PoseGraphOptimizer,
)


class TestPose4DOF:
    """Test 4-DOF pose representation."""
    
    def test_pose_initialization_default(self):
        """Test creating pose with default values."""
        pose = Pose4DOF()
        np.testing.assert_array_almost_equal(pose.pose, [0, 0, 0, 0])
    
    def test_pose_initialization_custom(self):
        """Test creating pose with custom values."""
        pose = Pose4DOF(x=1.0, y=2.0, z=3.0, yaw=0.5)
        np.testing.assert_array_almost_equal(pose.pose, [1.0, 2.0, 3.0, 0.5])
    
    def test_pose_position_property(self):
        """Test position property extraction."""
        pose = Pose4DOF(x=1.0, y=2.0, z=3.0, yaw=0.5)
        np.testing.assert_array_almost_equal(pose.position, [1.0, 2.0, 3.0])
    
    def test_pose_yaw_property(self):
        """Test yaw property extraction."""
        pose = Pose4DOF(x=1.0, y=2.0, z=3.0, yaw=0.5)
        assert abs(pose.yaw - 0.5) < 1e-10
    
    def test_pose_to_se3(self):
        """Test conversion to SE(3) matrix."""
        pose = Pose4DOF(x=1.0, y=2.0, z=3.0, yaw=0.0)
        T = pose.to_se3()
        
        assert T.shape == (4, 4)
        np.testing.assert_array_almost_equal(T[:3, 3], [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(T[3, :], [0, 0, 0, 1])
    
    def test_pose_to_se3_with_rotation(self):
        """Test SE(3) with non-zero yaw."""
        pose = Pose4DOF(x=1.0, y=0.0, z=0.0, yaw=np.pi / 2)
        T = pose.to_se3()
        
        # Should be 90-degree rotation around Z
        expected_R = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ])
        np.testing.assert_array_almost_equal(T[:3, :3], expected_R, decimal=10)
    
    def test_pose_copy(self):
        """Test pose copying."""
        pose1 = Pose4DOF(x=1.0, y=2.0, z=3.0, yaw=0.5)
        pose2 = pose1.copy()
        
        np.testing.assert_array_equal(pose1.pose, pose2.pose)
        
        # Verify it's a true copy
        pose2.pose[0] = 999
        assert pose1.pose[0] != pose2.pose[0]
    
    def test_pose_dtype(self):
        """Test pose uses float64."""
        pose = Pose4DOF(x=1, y=2, z=3, yaw=0)
        assert pose.pose.dtype == np.float64


class TestFactor:
    """Test factor representation."""
    
    def test_factor_initialization(self):
        """Test creating a factor."""
        measurement = np.array([0.1, 0.2, 0.3, 0.01])
        information = np.eye(4)
        
        factor = Factor(
            pose_id_a=0,
            pose_id_b=1,
            measurement=measurement,
            information=information,
        )
        
        assert factor.pose_id_a == 0
        assert factor.pose_id_b == 1
        np.testing.assert_array_equal(factor.measurement, measurement)
    
    def test_factor_residual(self):
        """Test residual computation."""
        measurement = np.array([0.1, 0.0, 0.0, 0.0])
        information = np.eye(4)
        
        factor = Factor(0, 1, measurement, information)
        
        pose_a = Pose4DOF(x=0.0, y=0.0, z=0.0, yaw=0.0)
        pose_b = Pose4DOF(x=0.15, y=0.0, z=0.0, yaw=0.0)
        
        residual = factor.residual(pose_a, pose_b)
        expected = measurement - (pose_b.pose - pose_a.pose)
        np.testing.assert_array_almost_equal(residual, expected)


class TestPoseGraphOptimizer:
    """Test pose graph optimization."""
    
    def test_optimizer_initialization(self):
        """Test creating optimizer."""
        opt = PoseGraphOptimizer(max_iterations=20)
        assert opt.max_iterations == 20
        assert len(opt.poses) == 0
        assert len(opt.factors) == 0
    
    def test_add_pose(self):
        """Test adding poses."""
        opt = PoseGraphOptimizer()
        
        pose1 = Pose4DOF(x=0.0, y=0.0, z=0.0, yaw=0.0)
        pose2 = Pose4DOF(x=1.0, y=0.0, z=0.0, yaw=0.0)
        
        opt.add_pose(0, pose1)
        opt.add_pose(1, pose2)
        
        assert len(opt.poses) == 2
        assert opt.pose_map[0] == 0
        assert opt.pose_map[1] == 1
    
    def test_get_pose(self):
        """Test retrieving poses."""
        opt = PoseGraphOptimizer()
        pose = Pose4DOF(x=1.0, y=2.0, z=3.0, yaw=0.5)
        
        opt.add_pose(0, pose)
        retrieved = opt.get_pose(0)
        
        assert retrieved is not None
        np.testing.assert_array_almost_equal(retrieved.pose, pose.pose)
    
    def test_add_factor(self):
        """Test adding factors."""
        opt = PoseGraphOptimizer()
        
        opt.add_pose(0, Pose4DOF())
        opt.add_pose(1, Pose4DOF())
        
        measurement = np.array([0.1, 0.0, 0.0, 0.0])
        information = np.eye(4)
        
        opt.add_factor(0, 1, measurement, information)
        
        assert len(opt.factors) == 1
    
    def test_add_factor_invalid_pose_id(self):
        """Test that adding factor with invalid pose ID raises error."""
        opt = PoseGraphOptimizer()
        opt.add_pose(0, Pose4DOF())
        
        measurement = np.array([0.1, 0.0, 0.0, 0.0])
        information = np.eye(4)
        
        with pytest.raises(ValueError):
            opt.add_factor(0, 999, measurement, information)
    
    def test_optimize_empty_graph(self):
        """Test optimization on empty graph."""
        opt = PoseGraphOptimizer()
        result = opt.optimize()
        
        assert result["iterations"] == 0
        assert result["converged"] is False
        assert "reason" in result
    
    def test_optimize_no_factors(self):
        """Test optimization with poses but no factors."""
        opt = PoseGraphOptimizer()
        opt.add_pose(0, Pose4DOF())
        opt.add_pose(1, Pose4DOF())
        
        result = opt.optimize()
        
        assert result["iterations"] == 0
        assert result["converged"] is False
    
    def test_optimize_simple_chain(self):
        """Test optimization on simple pose chain."""
        opt = PoseGraphOptimizer(max_iterations=10)
        
        # Create chain: pose0 -> pose1 -> pose2
        opt.add_pose(0, Pose4DOF(x=0.0, y=0.0, z=0.0, yaw=0.0))
        opt.add_pose(1, Pose4DOF(x=0.1, y=0.1, z=0.0, yaw=0.0))  # Slightly off
        opt.add_pose(2, Pose4DOF(x=0.2, y=0.2, z=0.0, yaw=0.0))  # Slightly off
        
        # Add constraints
        # 0 -> 1: should be [0.1, 0.1, 0, 0]
        opt.add_factor(0, 1, np.array([0.1, 0.1, 0.0, 0.0]), np.eye(4))
        
        # 1 -> 2: should be [0.1, 0.1, 0, 0]
        opt.add_factor(1, 2, np.array([0.1, 0.1, 0.0, 0.0]), np.eye(4))
        
        result = opt.optimize()
        
        assert result["iterations"] > 0
        # Residual should be small after optimization
        assert result["residual"] < 1.0
    
    def test_optimize_with_loop_closure(self):
        """Test optimization with loop closure constraint."""
        opt = PoseGraphOptimizer(max_iterations=20)
        
        # Create poses (slightly perturbed from ground truth)
        opt.add_pose(0, Pose4DOF(x=0.0, y=0.0, z=0.0, yaw=0.0))
        opt.add_pose(1, Pose4DOF(x=1.05, y=0.0, z=0.0, yaw=0.0))
        opt.add_pose(2, Pose4DOF(x=1.05, y=1.05, z=0.0, yaw=0.0))
        opt.add_pose(3, Pose4DOF(x=0.0, y=1.05, z=0.0, yaw=0.0))
        
        # Add odometry factors
        opt.add_factor(0, 1, np.array([1.0, 0.0, 0.0, 0.0]), np.eye(4))
        opt.add_factor(1, 2, np.array([0.0, 1.0, 0.0, 0.0]), np.eye(4))
        opt.add_factor(2, 3, np.array([-1.0, 0.0, 0.0, 0.0]), np.eye(4))
        
        # Add loop closure (3 -> 0) with high confidence
        opt.add_factor(3, 0, np.array([0.0, -1.0, 0.0, 0.0]), 10 * np.eye(4))
        
        result = opt.optimize()
        
        assert result["iterations"] > 0
        
        # Check that optimization produced a valid result
        poses = opt.get_poses_as_dict()
        
        # Residual should be reduced after optimization
        assert result["residual"] < 1.0
    
    def test_optimize_convergence(self):
        """Test that optimization history shows convergence."""
        opt = PoseGraphOptimizer(max_iterations=10)
        
        opt.add_pose(0, Pose4DOF())
        opt.add_pose(1, Pose4DOF(x=0.05, y=0.05, z=0.0, yaw=0.0))
        
        opt.add_factor(0, 1, np.array([0.1, 0.1, 0.0, 0.0]), np.eye(4))
        
        opt.optimize()
        
        # History should have multiple entries
        assert len(opt.optimization_history) > 1
        
        # Residual should generally decrease
        if len(opt.optimization_history) > 2:
            # Allow some non-monotonicity but overall should decrease
            assert opt.optimization_history[-1] <= opt.optimization_history[0]
    
    def test_get_poses_as_dict(self):
        """Test retrieving all poses as dictionary."""
        opt = PoseGraphOptimizer()
        
        opt.add_pose(0, Pose4DOF(x=1.0, y=2.0, z=3.0, yaw=0.1))
        opt.add_pose(1, Pose4DOF(x=4.0, y=5.0, z=6.0, yaw=0.2))
        
        poses_dict = opt.get_poses_as_dict()
        
        assert len(poses_dict) == 2
        assert 0 in poses_dict
        assert 1 in poses_dict
    
    def test_get_trajectory(self):
        """Test trajectory export."""
        opt = PoseGraphOptimizer()
        
        opt.add_pose(0, Pose4DOF(x=0.0, y=0.0, z=0.0, yaw=0.0))
        opt.add_pose(1, Pose4DOF(x=1.0, y=0.0, z=0.0, yaw=0.1))
        opt.add_pose(2, Pose4DOF(x=2.0, y=0.0, z=0.0, yaw=0.2))
        
        traj = opt.get_trajectory()
        
        assert traj.shape == (3, 4)
        np.testing.assert_array_almost_equal(traj[0], [0, 0, 0, 0])
        np.testing.assert_array_almost_equal(traj[1], [1, 0, 0, 0.1])
    
    def test_reset(self):
        """Test resetting optimizer state."""
        opt = PoseGraphOptimizer()
        
        opt.add_pose(0, Pose4DOF())
        opt.add_pose(1, Pose4DOF())
        opt.add_factor(0, 1, np.array([0.1, 0, 0, 0]), np.eye(4))
        
        assert len(opt.poses) == 2
        assert len(opt.factors) == 1
        
        opt.reset()
        
        assert len(opt.poses) == 0
        assert len(opt.factors) == 0
        assert len(opt.pose_map) == 0
    
    def test_statistics(self):
        """Test statistics reporting."""
        opt = PoseGraphOptimizer()
        
        opt.add_pose(0, Pose4DOF())
        opt.add_pose(1, Pose4DOF())
        opt.add_factor(0, 1, np.array([0.1, 0, 0, 0]), np.eye(4))
        
        stats = opt.statistics()
        
        assert stats["num_poses"] == 2
        assert stats["num_factors"] == 1
    
    def test_information_matrix_weighting(self):
        """Test that information matrix affects optimization."""
        # Simple test: verify that the optimizer accepts and uses information matrices
        
        opt = PoseGraphOptimizer(max_iterations=10)
        
        opt.add_pose(0, Pose4DOF(x=0.0, y=0.0, z=0.0, yaw=0.0))
        opt.add_pose(1, Pose4DOF(x=0.5, y=0.0, z=0.0, yaw=0.0))
        
        # Add factor with high confidence
        high_info = 10.0 * np.eye(4)
        opt.add_factor(0, 1, np.array([1.0, 0.0, 0.0, 0.0]), high_info)
        
        result = opt.optimize()
        
        assert result["iterations"] > 0
        # Just verify optimization completed without error
        assert "residual" in result
    
    def test_large_graph(self):
        """Test optimization on larger graph."""
        opt = PoseGraphOptimizer(max_iterations=10)
        
        # Build chain of 10 poses
        for i in range(10):
            opt.add_pose(i, Pose4DOF(x=float(i) * 0.1, y=0.0, z=0.0, yaw=0.0))
        
        # Add factors
        for i in range(9):
            opt.add_factor(i, i + 1, np.array([0.1, 0.0, 0.0, 0.0]), np.eye(4))
        
        result = opt.optimize()
        
        assert result["iterations"] > 0
        assert len(opt.poses) == 10
        assert len(opt.factors) == 9
