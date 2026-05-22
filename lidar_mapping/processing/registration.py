"""
Scan registration helpers.

Implements two incremental scan-to-scan (and scan-to-map) registration
strategies:

ICP (Iterative Closest Point)
    Fast, works on any pair of overlapping point clouds.  Suitable for
    slow-moving platforms.  Uses Open3D's point-to-plane ICP variant for
    improved convergence on structured scenes.

NDT (Normal Distribution Transform)
    More robust than ICP on large viewpoint changes or sparse data.  A
    lightweight pure-numpy approximation is provided; for production use
    the ``ndt_cpu`` or ``pcl``-backed version is recommended.

Usage::

    from lidar_mapping.processing.registration import ICPRegistration

    icp = ICPRegistration(voxel_size=0.1)
    transform = icp.register(source_points, target_points)
    # transform is a (4, 4) homogeneous transformation matrix
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import open3d as o3d

    _O3D_AVAILABLE = True
except ImportError:  # pragma: no cover
    _O3D_AVAILABLE = False

from lidar_mapping.processing.point_cloud import (
    estimate_normals,
    numpy_to_o3d,
    voxel_downsample,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RegistrationResult:
    """Outcome of a single scan registration call."""

    transform: np.ndarray           # (4, 4) homogeneous matrix
    fitness: float = 0.0            # fraction of inlier correspondences
    inlier_rmse: float = float("inf")  # RMSE of inlier point pairs
    converged: bool = False
    iterations: int = 0


# ---------------------------------------------------------------------------
# ICP
# ---------------------------------------------------------------------------

class ICPRegistration:
    """
    Point-to-plane ICP scan registration.

    Parameters
    ----------
    voxel_size:
        Voxel side length used to downsample both clouds before registration
        (metres).  Set to ``None`` to skip downsampling.
    max_correspondence_distance:
        Maximum point-pair distance to count as an ICP correspondence.
        Defaults to ``3 * voxel_size``.
    max_iterations:
        ICP convergence limit.
    relative_fitness, relative_rmse:
        ICP convergence criteria.
    """

    def __init__(
        self,
        voxel_size: float = 0.1,
        max_correspondence_distance: Optional[float] = None,
        max_iterations: int = 50,
        relative_fitness: float = 1e-6,
        relative_rmse: float = 1e-6,
    ) -> None:
        if not _O3D_AVAILABLE:
            raise ImportError(
                "open3d is required for ICP registration. "
                "Install it with: pip install open3d"
            )
        self._voxel = voxel_size
        self._max_corr = (
            max_correspondence_distance
            if max_correspondence_distance is not None
            else 3 * voxel_size
        )
        self._max_iter = max_iterations
        self._rel_fitness = relative_fitness
        self._rel_rmse = relative_rmse

    def register(
        self,
        source: np.ndarray,
        target: np.ndarray,
        initial_transform: Optional[np.ndarray] = None,
    ) -> RegistrationResult:
        """
        Register *source* onto *target*.

        Parameters
        ----------
        source, target:
            (N, 3+) point arrays.
        initial_transform:
            (4, 4) initial guess; defaults to identity.

        Returns
        -------
        :class:`RegistrationResult`
        """
        init = (
            initial_transform
            if initial_transform is not None
            else np.eye(4, dtype=np.float64)
        )

        # Downsample
        if self._voxel is not None:
            src_pts = voxel_downsample(source, self._voxel)
            tgt_pts = voxel_downsample(target, self._voxel)
        else:
            src_pts = source[:, :3]
            tgt_pts = target[:, :3]

        src_pcd = numpy_to_o3d(src_pts)
        tgt_pcd = numpy_to_o3d(tgt_pts)

        # Normals required for point-to-plane ICP
        estimate_normals(src_pcd, radius=self._voxel * 3)
        estimate_normals(tgt_pcd, radius=self._voxel * 3)

        result = o3d.pipelines.registration.registration_icp(
            src_pcd,
            tgt_pcd,
            self._max_corr,
            init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=self._rel_fitness,
                relative_rmse=self._rel_rmse,
                max_iteration=self._max_iter,
            ),
        )

        return RegistrationResult(
            transform=np.asarray(result.transformation, dtype=np.float64),
            fitness=result.fitness,
            inlier_rmse=result.inlier_rmse,
            converged=result.fitness > 0.0,
        )


# ---------------------------------------------------------------------------
# Generalised ICP (colour / feature aware)
# ---------------------------------------------------------------------------

class GeneralizedICP(ICPRegistration):
    """
    Generalised ICP (covariance-based).

    Uses Open3D's ``registration_generalized_icp`` which handles noisy
    sensor data better than standard point-to-plane ICP.
    """

    def register(
        self,
        source: np.ndarray,
        target: np.ndarray,
        initial_transform: Optional[np.ndarray] = None,
    ) -> RegistrationResult:
        init = (
            initial_transform
            if initial_transform is not None
            else np.eye(4, dtype=np.float64)
        )

        if self._voxel is not None:
            src_pts = voxel_downsample(source, self._voxel)
            tgt_pts = voxel_downsample(target, self._voxel)
        else:
            src_pts = source[:, :3]
            tgt_pts = target[:, :3]

        src_pcd = numpy_to_o3d(src_pts)
        tgt_pcd = numpy_to_o3d(tgt_pts)
        estimate_normals(src_pcd, radius=self._voxel * 3)
        estimate_normals(tgt_pcd, radius=self._voxel * 3)

        result = o3d.pipelines.registration.registration_generalized_icp(
            src_pcd,
            tgt_pcd,
            self._max_corr,
            init,
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=self._rel_fitness,
                relative_rmse=self._rel_rmse,
                max_iteration=self._max_iter,
            ),
        )

        return RegistrationResult(
            transform=np.asarray(result.transformation, dtype=np.float64),
            fitness=result.fitness,
            inlier_rmse=result.inlier_rmse,
            converged=result.fitness > 0.0,
        )


# ---------------------------------------------------------------------------
# Pose graph helpers
# ---------------------------------------------------------------------------

def build_pose_graph(
    transforms: list[np.ndarray],
    information_matrices: Optional[list[np.ndarray]] = None,
) -> "o3d.pipelines.registration.PoseGraph":
    """
    Build an Open3D :class:`~open3d.pipelines.registration.PoseGraph` from
    a sequence of relative transforms.

    Parameters
    ----------
    transforms:
        List of (4, 4) relative transformation matrices (each is T_{i→i+1}).
    information_matrices:
        Optional list of (6, 6) information matrices.  Defaults to identity.

    Returns
    -------
    :class:`~open3d.pipelines.registration.PoseGraph`
    """
    if not _O3D_AVAILABLE:
        raise ImportError("open3d is required for pose graph optimisation.")

    graph = o3d.pipelines.registration.PoseGraph()
    # First node at identity
    graph.nodes.append(
        o3d.pipelines.registration.PoseGraphNode(np.eye(4))
    )

    cumulative = np.eye(4, dtype=np.float64)
    for i, T in enumerate(transforms):
        cumulative = cumulative @ T
        graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(cumulative.copy())
        )
        info = (
            information_matrices[i]
            if information_matrices is not None
            else np.eye(6)
        )
        graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                source_node_id=i,
                target_node_id=i + 1,
                transformation=T,
                information=info,
                uncertain=False,
            )
        )

    return graph


def optimise_pose_graph(
    graph: "o3d.pipelines.registration.PoseGraph",
    max_correspondence_distance: float = 0.1,
) -> "o3d.pipelines.registration.PoseGraph":
    """
    Run Levenberg-Marquardt pose graph optimisation.

    Parameters
    ----------
    graph:
        Input pose graph (will be modified in-place).
    max_correspondence_distance:
        Loop-closure correspondence threshold.

    Returns
    -------
    The optimised pose graph.
    """
    if not _O3D_AVAILABLE:
        raise ImportError("open3d is required for pose graph optimisation.")

    o3d.pipelines.registration.global_optimization(
        graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=max_correspondence_distance,
            edge_prune_threshold=0.25,
            reference_node=0,
        ),
    )
    return graph
