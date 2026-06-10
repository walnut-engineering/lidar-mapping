"""
Tests for scan registration (Generalized ICP) and pose graph building /
optimisation.

All tests run fully offline using synthetic numpy point clouds.
Requires open3d.
"""

from __future__ import annotations

import pytest
import numpy as np

try:
    import open3d as o3d
    _O3D_AVAILABLE = True
except ImportError:
    _O3D_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _O3D_AVAILABLE, reason="open3d not installed"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_cloud(n: int = 2000, z: float = 0.0, seed: int = 0) -> np.ndarray:
    """Random planar cloud in XY at height z."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-5.0, 5.0, (n, 3)).astype(np.float32)
    pts[:, 2] = z
    return pts


def _box_cloud(n: int = 3000, side: float = 4.0, seed: int = 0) -> np.ndarray:
    """Random points on the surface of an axis-aligned box."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-side / 2, side / 2, (n, 3)).astype(np.float32)
    # Snap random faces
    face = rng.integers(0, 3, n)
    sign = rng.choice([-1.0, 1.0], n)
    for i in range(n):
        pts[i, face[i]] = sign[i] * side / 2
    return pts


def _transform_cloud(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4×4 homogeneous transform to an (N,3) cloud."""
    ones = np.ones((len(pts), 1), dtype=pts.dtype)
    h = np.hstack([pts, ones])  # (N, 4)
    return (h @ T[:3, :].T).astype(pts.dtype)


def _small_translation(tx: float = 0.05) -> np.ndarray:
    """Return a 4×4 matrix with a small translation."""
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = tx
    return T


# ===========================================================================
# Generalized ICP
# ===========================================================================

class TestGeneralizedICP:
    """Tests for the GeneralizedICP wrapper."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from lidar_mapping.processing.registration import (
            GeneralizedICP,
            RegistrationResult,
        )
        self.GeneralizedICP = GeneralizedICP
        self.RegistrationResult = RegistrationResult

    def test_identical_clouds_converges(self):
        """Registering a cloud onto itself should converge with near-zero RMSE."""
        cloud = _box_cloud(3000)
        gicp = self.GeneralizedICP(voxel_size=0.2)
        result = gicp.register(cloud, cloud)
        assert result.converged
        assert result.inlier_rmse < 0.05

    def test_small_translation_recovered(self):
        """A small known translation should be found within tolerance."""
        cloud = _box_cloud(3000)
        T_true = _small_translation(0.1)
        moved = _transform_cloud(cloud, T_true)

        gicp = self.GeneralizedICP(voxel_size=0.15)
        # Register original onto moved: transform should be close to T_true
        result = gicp.register(cloud, moved, initial_transform=np.eye(4))
        assert result.converged
        # Check the translation magnitude is recovered (sign may vary by convention)
        t_recovered = result.transform[:3, 3]
        assert abs(np.linalg.norm(t_recovered) - 0.1) < 0.05

    def test_returns_registration_result_type(self):
        cloud = _box_cloud(500)
        gicp = self.GeneralizedICP(voxel_size=0.5)
        result = gicp.register(cloud, cloud)
        assert isinstance(result, self.RegistrationResult)
        assert result.transform.shape == (4, 4)
        assert isinstance(result.fitness, float)
        assert isinstance(result.converged, bool)

    def test_identity_initial_transform_accepted(self):
        """Passing identity as initial transform should not crash."""
        cloud = _box_cloud(500)
        gicp = self.GeneralizedICP(voxel_size=0.3)
        result = gicp.register(cloud, cloud, initial_transform=np.eye(4))
        assert result.transform.shape == (4, 4)

    def test_fitness_between_zero_and_one(self):
        """Fitness is the fraction of inlier correspondences (0–1)."""
        cloud = _box_cloud(1000)
        gicp = self.GeneralizedICP(voxel_size=0.2)
        result = gicp.register(cloud, cloud)
        assert 0.0 <= result.fitness <= 1.0


# ===========================================================================
# ICP (re-verify that existing ICP also works)
# ===========================================================================

class TestICPRegistration:
    """Smoke-tests for the standard ICP path (point-to-plane)."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from lidar_mapping.processing.registration import (
            ICPRegistration,
            RegistrationResult,
        )
        self.ICPRegistration = ICPRegistration
        self.RegistrationResult = RegistrationResult

    def test_identical_clouds_converge(self):
        cloud = _box_cloud(2000)
        icp = self.ICPRegistration(voxel_size=0.2)
        result = icp.register(cloud, cloud)
        assert result.converged
        assert result.inlier_rmse < 0.1

    def test_returns_correct_type(self):
        cloud = _box_cloud(500)
        icp = self.ICPRegistration(voxel_size=0.3)
        result = icp.register(cloud, cloud)
        assert isinstance(result, self.RegistrationResult)
        assert result.transform.shape == (4, 4)


# ===========================================================================
# build_pose_graph
# ===========================================================================

class TestBuildPoseGraph:
    """Tests for the ``build_pose_graph`` helper."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from lidar_mapping.processing.registration import build_pose_graph
        self.build_pose_graph = build_pose_graph

    def test_two_transforms_three_nodes(self):
        """2 identity transforms → 3 nodes, 2 edges."""
        transforms = [np.eye(4), np.eye(4)]
        graph = self.build_pose_graph(transforms)
        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2

    def test_five_transforms_six_nodes(self):
        transforms = [np.eye(4)] * 5
        graph = self.build_pose_graph(transforms)
        assert len(graph.nodes) == 6
        assert len(graph.edges) == 5

    def test_first_node_is_identity(self):
        """The reference node (node 0) should be at the identity pose."""
        transforms = [_small_translation(1.0)] * 3
        graph = self.build_pose_graph(transforms)
        np.testing.assert_allclose(graph.nodes[0].pose, np.eye(4), atol=1e-9)

    def test_second_node_matches_first_transform(self):
        """After one transform, node 1's pose should equal that transform."""
        T = _small_translation(2.5)
        graph = self.build_pose_graph([T])
        np.testing.assert_allclose(graph.nodes[1].pose, T, atol=1e-9)

    def test_cumulative_poses_accumulate(self):
        """Nodes after two translations should accumulate correctly."""
        T = _small_translation(1.0)
        graph = self.build_pose_graph([T, T])
        expected_node2 = T @ T
        np.testing.assert_allclose(
            graph.nodes[2].pose, expected_node2, atol=1e-9
        )

    def test_custom_information_matrix_used(self):
        """Passing a custom 6×6 information matrix should not raise."""
        I6 = np.eye(6) * 100.0
        graph = self.build_pose_graph(
            [np.eye(4)], information_matrices=[I6]
        )
        assert len(graph.edges) == 1

    def test_single_transform(self):
        graph = self.build_pose_graph([np.eye(4)])
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1

    def test_empty_transforms_returns_single_node(self):
        graph = self.build_pose_graph([])
        assert len(graph.nodes) == 1
        assert len(graph.edges) == 0


# ===========================================================================
# optimise_pose_graph
# ===========================================================================

class TestOptimisePoseGraph:
    """Tests for the ``optimise_pose_graph`` helper."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from lidar_mapping.processing.registration import (
            build_pose_graph,
            optimise_pose_graph,
        )
        self.build_pose_graph = build_pose_graph
        self.optimise_pose_graph = optimise_pose_graph

    def test_node_count_preserved(self):
        """Optimisation must not add or remove nodes."""
        transforms = [np.eye(4)] * 4
        graph = self.build_pose_graph(transforms)
        n_before = len(graph.nodes)
        self.optimise_pose_graph(graph, max_correspondence_distance=0.5)
        assert len(graph.nodes) == n_before

    def test_returns_same_graph(self):
        """optimise_pose_graph should return the graph (in-place)."""
        graph = self.build_pose_graph([np.eye(4)])
        returned = self.optimise_pose_graph(graph)
        assert returned is graph

    def test_reference_node_stays_near_identity(self):
        """Node 0 is pinned as the reference node; its pose should remain identity."""
        T = _small_translation(1.0)
        graph = self.build_pose_graph([T, T, T])
        self.optimise_pose_graph(graph, max_correspondence_distance=0.5)
        np.testing.assert_allclose(graph.nodes[0].pose, np.eye(4), atol=0.01)

    def test_optimise_three_node_chain(self):
        """Three-node chain with identity edges stays consistent after opt."""
        graph = self.build_pose_graph([np.eye(4), np.eye(4)])
        self.optimise_pose_graph(graph)
        assert len(graph.nodes) == 3

    def test_custom_max_correspondence(self):
        """Passing a custom distance should not raise."""
        graph = self.build_pose_graph([np.eye(4)])
        self.optimise_pose_graph(graph, max_correspondence_distance=1.0)


# ===========================================================================
# RegistrationResult — non-converged edge identification
# ===========================================================================

class TestNonConvergedEdge:
    """Verify that non-converged results can be identified and skipped."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from lidar_mapping.processing.registration import (
            RegistrationResult,
            build_pose_graph,
        )
        self.RegistrationResult = RegistrationResult
        self.build_pose_graph = build_pose_graph

    def test_non_converged_flag(self):
        result = self.RegistrationResult(
            transform=np.eye(4),
            fitness=0.0,
            inlier_rmse=999.9,
            converged=False,
            iterations=50,
        )
        assert not result.converged

    def test_filter_converged_transforms(self):
        """Filtering out non-converged results before pose graph build."""
        results = [
            self.RegistrationResult(np.eye(4), fitness=0.9, converged=True),
            self.RegistrationResult(np.eye(4), fitness=0.1, converged=False),
            self.RegistrationResult(np.eye(4), fitness=0.8, converged=True),
        ]
        converged_transforms = [r.transform for r in results if r.converged]
        graph = self.build_pose_graph(converged_transforms)
        # 2 converged transforms → 3 nodes
        assert len(graph.nodes) == 3

    def test_empty_after_filter_gives_one_node(self):
        """All non-converged → single-node pose graph."""
        results = [
            self.RegistrationResult(np.eye(4), converged=False),
        ]
        transforms = [r.transform for r in results if r.converged]
        graph = self.build_pose_graph(transforms)
        assert len(graph.nodes) == 1
