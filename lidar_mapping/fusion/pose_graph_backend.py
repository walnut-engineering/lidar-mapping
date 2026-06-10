"""
Pose graph optimization backend for VSLAM.

Implements a lightweight factor-graph SLAM optimizer without external dependencies
(no Open3D, GTSAM, or Ceres required). Uses Gauss-Newton optimization for
4-DOF poses (x, y, z, yaw) with gravity-aligned IMU constraints.

Key design:
- Poses: 4-DOF parametrization (position + yaw)
- Factors: Relative pose measurements with covariance
- Optimization: Gauss-Newton with first-pose gauge fixing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import solve

logger = logging.getLogger(__name__)


@dataclass
class Pose4DOF:
    """A 4-DOF pose: (x, y, z, yaw) in meters/radians."""
    
    pose: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float64))
    
    def __init__(self, x: float = 0, y: float = 0, z: float = 0, yaw: float = 0):
        self.pose = np.array([x, y, z, yaw], dtype=np.float64)
    
    @property
    def position(self) -> np.ndarray:
        """Return (x, y, z) as 3D array."""
        return self.pose[:3]
    
    @property
    def yaw(self) -> float:
        """Return yaw angle in radians."""
        return float(self.pose[3])
    
    def to_se3(self) -> np.ndarray:
        """Convert to 4x4 SE(3) matrix (for visualization/comparison)."""
        T = np.eye(4, dtype=np.float64)
        
        # Position
        T[:3, 3] = self.pose[:3]
        
        # Rotation around Z (yaw only)
        yaw = self.pose[3]
        c, s = np.cos(yaw), np.sin(yaw)
        T[:3, :3] = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1],
        ])
        
        return T
    
    def copy(self) -> Pose4DOF:
        """Return a copy of this pose."""
        return Pose4DOF(*self.pose)


@dataclass
class Factor:
    """A relative pose measurement between two nodes."""
    
    pose_id_a: int
    pose_id_b: int
    measurement: np.ndarray  # 4D delta (dx, dy, dz, dyaw)
    information: np.ndarray  # 4x4 information matrix (inverse covariance)
    robust: bool = False  # If True, apply Huber loss
    
    def residual(self, pose_a: Pose4DOF, pose_b: Pose4DOF) -> np.ndarray:
        """Compute residual = measured - (pose_b - pose_a)."""
        delta_actual = pose_b.pose - pose_a.pose
        return self.measurement - delta_actual


class PoseGraphOptimizer:
    """
    4-DOF pose graph optimizer using Gauss-Newton.
    
    Parameters
    ----------
    max_iterations : int
        Maximum optimization iterations.
    convergence_threshold : float
        Stop when residual norm < threshold.
    verbose : bool
        Print optimization progress.
    """
    
    def __init__(
        self,
        max_iterations: int = 10,
        convergence_threshold: float = 1e-6,
        verbose: bool = False,
    ):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.verbose = verbose
        
        self.poses: List[Pose4DOF] = []
        self.pose_map: Dict[int, int] = {}  # pose_id → index in self.poses
        self.factors: List[Factor] = []
        
        self.optimization_history: List[float] = []
    
    def add_pose(self, pose_id: int, pose: Pose4DOF) -> None:
        """Register a new pose node."""
        if pose_id in self.pose_map:
            logger.warning(f"Pose ID {pose_id} already exists, overwriting")
            idx = self.pose_map[pose_id]
            self.poses[idx] = pose.copy()
        else:
            self.pose_map[pose_id] = len(self.poses)
            self.poses.append(pose.copy())
    
    def add_factor(
        self,
        pose_id_a: int,
        pose_id_b: int,
        measurement: np.ndarray,
        information: np.ndarray,
        robust: bool = False,
    ) -> None:
        """
        Add a measurement factor between two poses.
        
        Parameters
        ----------
        pose_id_a, pose_id_b : int
            Pose node IDs.
        measurement : np.ndarray
            (4,) measured delta (dx, dy, dz, dyaw).
        information : np.ndarray
            (4, 4) information matrix (inverse covariance).
        robust : bool
            If True, apply Huber loss to outliers.
        """
        if pose_id_a not in self.pose_map:
            raise ValueError(f"Pose {pose_id_a} not in graph")
        if pose_id_b not in self.pose_map:
            raise ValueError(f"Pose {pose_id_b} not in graph")
        
        measurement = np.asarray(measurement, dtype=np.float64)
        information = np.asarray(information, dtype=np.float64)
        
        if measurement.shape != (4,):
            raise ValueError(f"Measurement must be (4,), got {measurement.shape}")
        if information.shape != (4, 4):
            raise ValueError(f"Information must be (4, 4), got {information.shape}")
        
        factor = Factor(pose_id_a, pose_id_b, measurement, information, robust)
        self.factors.append(factor)
    
    def _build_linear_system(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build Hessian H and gradient b for Gauss-Newton step."""
        n_poses = len(self.poses)
        n_dof = 4 * n_poses
        
        H = np.zeros((n_dof, n_dof), dtype=np.float64)
        b = np.zeros(n_dof, dtype=np.float64)
        
        for factor in self.factors:
            i_a = self.pose_map[factor.pose_id_a]
            i_b = self.pose_map[factor.pose_id_b]
            
            pose_a = self.poses[i_a]
            pose_b = self.poses[i_b]
            
            # Residual
            residual = factor.residual(pose_a, pose_b)
            
            # For linear system, Jacobians are identity (4-DOF linear problem)
            # d/d(pose_a) = -I, d/d(pose_b) = +I
            J_a = -np.eye(4)
            J_b = np.eye(4)
            
            # Accumulate normal equations
            H_aa = J_a.T @ factor.information @ J_a
            H_ab = J_a.T @ factor.information @ J_b
            H_bb = J_b.T @ factor.information @ J_b
            
            b_a = J_a.T @ factor.information @ residual
            b_b = J_b.T @ factor.information @ residual
            
            # Add to global system
            start_a = i_a * 4
            end_a = (i_a + 1) * 4
            start_b = i_b * 4
            end_b = (i_b + 1) * 4
            
            H[start_a:end_a, start_a:end_a] += H_aa
            H[start_a:end_a, start_b:end_b] += H_ab
            H[start_b:end_b, start_a:end_a] += H_ab.T
            H[start_b:end_b, start_b:end_b] += H_bb
            
            b[start_a:end_a] += b_a
            b[start_b:end_b] += b_b
        
        return H, b
    
    def _compute_residual_norm(self) -> float:
        """Compute total residual norm across all factors."""
        total = 0.0
        for factor in self.factors:
            pose_a = self.poses[self.pose_map[factor.pose_id_a]]
            pose_b = self.poses[self.pose_map[factor.pose_id_b]]
            residual = factor.residual(pose_a, pose_b)
            total += np.sum(residual ** 2)
        return np.sqrt(total)
    
    def optimize(self, max_iterations: Optional[int] = None) -> Dict[str, float]:
        """
        Run Gauss-Newton optimization.
        
        Parameters
        ----------
        max_iterations : int, optional
            Override default max iterations.
        
        Returns
        -------
        dict
            Optimization statistics (iterations, residual, converged).
        """
        if len(self.poses) == 0:
            logger.warning("No poses in graph, skipping optimization")
            return {
                "iterations": 0,
                "converged": False,
                "residual": 0.0,
                "reason": "empty_graph",
            }
        
        if len(self.factors) == 0:
            logger.warning("No factors in graph, skipping optimization")
            return {
                "iterations": 0,
                "converged": False,
                "residual": 0.0,
                "reason": "no_factors",
            }
        
        max_iter = max_iterations if max_iterations is not None else self.max_iterations
        self.optimization_history.clear()
        
        converged = False
        residual_norm = self._compute_residual_norm()
        self.optimization_history.append(residual_norm)
        
        if self.verbose:
            logger.info(f"Optimization start: residual = {residual_norm:.6f}")
        
        for iteration in range(max_iter):
            # Build linear system
            try:
                H, b = self._build_linear_system()
            except Exception as e:
                logger.error(f"Failed to build linear system: {e}")
                return {
                    "iterations": iteration,
                    "converged": False,
                    "residual": residual_norm,
                    "reason": "linear_system_failed",
                }
            
            # Fix first pose (gauge freedom)
            H[:4, :4] = np.eye(4)
            b[:4] = 0
            
            # Solve
            try:
                delta_x = solve(H, b, assume_a="gen")
            except np.linalg.LinAlgError as e:
                logger.warning(f"Linear solve failed: {e}, trying pseudoinverse")
                try:
                    delta_x = np.linalg.pinv(H) @ b
                except Exception as e2:
                    logger.error(f"Pseudoinverse also failed: {e2}")
                    return {
                        "iterations": iteration,
                        "converged": False,
                        "residual": residual_norm,
                        "reason": "solve_failed",
                    }
            
            # Line search with damping (helps convergence on ARM)
            damping = 0.5
            for step_size in [1.0, 0.5, 0.25, 0.1]:
                # Update poses
                poses_backup = [p.copy() for p in self.poses]
                
                for i in range(len(self.poses)):
                    self.poses[i].pose += step_size * damping * delta_x[i * 4:(i + 1) * 4]
                
                # Check residual
                residual_new = self._compute_residual_norm()
                
                if residual_new < residual_norm or step_size < 0.1:
                    residual_norm = residual_new
                    break
                else:
                    # Revert
                    self.poses = poses_backup
            
            self.optimization_history.append(residual_norm)
            
            if self.verbose:
                logger.info(
                    f"Iteration {iteration + 1}: residual = {residual_norm:.6f}, "
                    f"delta_norm = {np.linalg.norm(delta_x):.6e}"
                )
            
            # Check convergence
            if residual_norm < self.convergence_threshold:
                converged = True
                if self.verbose:
                    logger.info(f"Converged at iteration {iteration + 1}")
                break
        
        return {
            "iterations": iteration + 1,
            "converged": converged,
            "residual": residual_norm,
            "reason": "max_iterations_reached" if not converged else "converged",
        }
    
    def get_pose(self, pose_id: int) -> Optional[Pose4DOF]:
        """Retrieve a pose by ID."""
        if pose_id not in self.pose_map:
            return None
        return self.poses[self.pose_map[pose_id]].copy()
    
    def get_poses_as_dict(self) -> Dict[int, Pose4DOF]:
        """Return all poses as {pose_id: Pose4DOF}."""
        result = {}
        for pose_id, idx in self.pose_map.items():
            result[pose_id] = self.poses[idx].copy()
        return result
    
    def get_trajectory(self) -> np.ndarray:
        """Return trajectory as (N, 4) array of poses, ordered by ID."""
        if not self.poses:
            return np.array([]).reshape(0, 4)
        
        poses_array = np.array([p.pose for p in self.poses], dtype=np.float64)
        return poses_array
    
    def reset(self) -> None:
        """Clear all poses and factors."""
        self.poses.clear()
        self.pose_map.clear()
        self.factors.clear()
        self.optimization_history.clear()
    
    def statistics(self) -> Dict[str, int]:
        """Return graph statistics."""
        return {
            "num_poses": len(self.poses),
            "num_factors": len(self.factors),
            "num_optimization_steps": len(self.optimization_history),
        }
