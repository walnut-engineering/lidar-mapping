"""
Loop closure detection and pose-graph correction.

A *loop closure* occurs when the platform revisits a previously mapped
area.  Without correction, accumulated odometry drift causes the second
visit to be displaced from the first, producing a "double map".  Detecting
the revisit and adding a constraint between the two keyframes lets a
pose-graph optimiser globally distribute the error and produce a single
consistent map.

This module provides:

* :class:`Keyframe`            – a downsampled point cloud + pose snapshot.
* :class:`KeyframeStore`       – appends keyframes whenever the platform
  moves more than a configurable distance/rotation.
* :class:`LoopClosureDetector` – nearest-neighbour search over keyframe
  positions, verified with ICP.
* :func:`optimize_with_loop_closures` – builds an Open3D pose graph from
  the sequential odometry + detected loop edges, runs Levenberg-Marquardt
  optimisation, returns the corrected poses.

The detector and store are pure numpy; the verifier and optimiser require
``open3d``.

Typical use::

    store = KeyframeStore(distance_threshold=1.0, rotation_threshold_deg=15.0)
    detector = LoopClosureDetector(
        search_radius=3.0,
        min_time_gap=20,
        fitness_threshold=0.6,
    )

    for scan, pose in zip(scans, poses):
        kf = store.try_add(scan, pose)
        if kf is not None:
            candidates = detector.find_candidates(store, kf)
            for cand in candidates:
                edge = detector.verify(store[cand], kf)
                if edge is not None:
                    loop_edges.append((cand, len(store) - 1, edge))

    corrected_poses = optimize_with_loop_closures(poses, store, loop_edges)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional dep
    import open3d as o3d
    _O3D_AVAILABLE = True
except ImportError:  # pragma: no cover
    _O3D_AVAILABLE = False

from lidar_mapping.processing.point_cloud import voxel_downsample
from lidar_mapping.processing.registration import (
    ICPRegistration,
    build_pose_graph,
    optimise_pose_graph,
)
from lidar_mapping.utils.transforms import compose_transforms

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Keyframe:
    """A pose-tagged, downsampled point cloud snapshot."""

    index: int                       # position in the keyframe sequence
    scan_index: int                  # position in the original scan sequence
    pose: np.ndarray                 # (4, 4) world ← keyframe
    points: np.ndarray               # (N, 3) sensor-frame points
    timestamp: float = 0.0


@dataclass
class LoopEdge:
    """A verified loop-closure constraint between two keyframes."""

    source_kf: int                   # earlier keyframe index
    target_kf: int                   # later keyframe index
    transform: np.ndarray            # (4, 4) source ← target
    fitness: float
    inlier_rmse: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pose_translation(pose: np.ndarray) -> np.ndarray:
    return np.asarray(pose, dtype=np.float64)[:3, 3]


def _pose_rotation_angle_deg(p0: np.ndarray, p1: np.ndarray) -> float:
    """Angle (degrees) of relative rotation between two 4×4 poses."""
    R0 = np.asarray(p0, dtype=np.float64)[:3, :3]
    R1 = np.asarray(p1, dtype=np.float64)[:3, :3]
    R = R0.T @ R1
    # Clamp trace to handle floating-point noise outside [-1, 3]
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


# ---------------------------------------------------------------------------
# Keyframe store
# ---------------------------------------------------------------------------

class KeyframeStore:
    """
    Append-only collection of keyframes.

    A new keyframe is added by :meth:`try_add` whenever the supplied pose
    differs from the most recent keyframe by more than ``distance_threshold``
    metres OR ``rotation_threshold_deg`` degrees.  The first call always
    adds a keyframe.

    Parameters
    ----------
    distance_threshold:
        Translation threshold (metres).
    rotation_threshold_deg:
        Rotation threshold (degrees).
    voxel_size:
        Voxel size for keyframe-cloud downsampling.  Set to ``None`` to
        skip downsampling.
    """

    def __init__(
        self,
        distance_threshold: float = 1.0,
        rotation_threshold_deg: float = 15.0,
        voxel_size: Optional[float] = 0.2,
    ) -> None:
        if distance_threshold < 0:
            raise ValueError("distance_threshold must be non-negative")
        if rotation_threshold_deg < 0:
            raise ValueError("rotation_threshold_deg must be non-negative")
        self._dist_thr = float(distance_threshold)
        self._rot_thr = float(rotation_threshold_deg)
        self._voxel = voxel_size
        self._frames: List[Keyframe] = []

    # -- Container ----------------------------------------------------

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int) -> Keyframe:
        return self._frames[idx]

    def __iter__(self):
        return iter(self._frames)

    @property
    def frames(self) -> List[Keyframe]:
        """Return the internal list (do not mutate)."""
        return self._frames

    def positions(self) -> np.ndarray:
        """(K, 3) array of keyframe world-frame positions."""
        if not self._frames:
            return np.zeros((0, 3), dtype=np.float64)
        return np.array([_pose_translation(kf.pose) for kf in self._frames],
                        dtype=np.float64)

    # -- Mutation -----------------------------------------------------

    def try_add(
        self,
        scan: np.ndarray,
        pose: np.ndarray,
        scan_index: Optional[int] = None,
        timestamp: float = 0.0,
    ) -> Optional[Keyframe]:
        """
        Conditionally add a keyframe.

        Returns the new :class:`Keyframe` if it was added, otherwise
        ``None``.
        """
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (4, 4):
            raise ValueError("pose must be a (4, 4) matrix")
        scan = np.asarray(scan)
        if scan.ndim != 2 or scan.shape[1] < 3:
            raise ValueError("scan must be an (N, 3+) array")

        if self._frames:
            last = self._frames[-1]
            dt = float(np.linalg.norm(
                _pose_translation(pose) - _pose_translation(last.pose)
            ))
            dr = _pose_rotation_angle_deg(last.pose, pose)
            if dt < self._dist_thr and dr < self._rot_thr:
                return None

        pts = scan[:, :3].astype(np.float64, copy=False)
        if self._voxel is not None and len(pts) > 0:
            pts = voxel_downsample(pts, self._voxel)

        kf = Keyframe(
            index=len(self._frames),
            scan_index=(scan_index if scan_index is not None
                        else len(self._frames)),
            pose=pose.copy(),
            points=pts.astype(np.float64, copy=False),
            timestamp=float(timestamp),
        )
        self._frames.append(kf)
        return kf

    def clear(self) -> None:
        self._frames.clear()


# ---------------------------------------------------------------------------
# Loop closure detection
# ---------------------------------------------------------------------------

class LoopClosureDetector:
    """
    Detect candidate revisits and verify them with ICP.

    Parameters
    ----------
    search_radius:
        Maximum world-frame distance (metres) between keyframes to be
        considered a candidate pair.
    min_time_gap:
        Minimum index difference between candidate keyframes.  Prevents
        matching adjacent keyframes that are trivially close.
    fitness_threshold:
        Minimum ICP fitness (fraction of inlier correspondences) for a
        candidate to be accepted as a verified loop.
    max_rmse:
        Maximum ICP inlier RMSE for acceptance (metres).
    icp_voxel_size:
        Voxel size used by the verification ICP.
    icp_max_correspondence_distance:
        ICP correspondence threshold (metres).
    """

    def __init__(
        self,
        search_radius: float = 3.0,
        min_time_gap: int = 20,
        fitness_threshold: float = 0.5,
        max_rmse: float = 0.5,
        icp_voxel_size: float = 0.2,
        icp_max_correspondence_distance: float = 1.0,
    ) -> None:
        if search_radius <= 0:
            raise ValueError("search_radius must be positive")
        if min_time_gap < 1:
            raise ValueError("min_time_gap must be >= 1")
        if not 0.0 <= fitness_threshold <= 1.0:
            raise ValueError("fitness_threshold must be in [0, 1]")

        self._radius = float(search_radius)
        self._gap = int(min_time_gap)
        self._fit = float(fitness_threshold)
        self._rmse = float(max_rmse)
        self._icp = ICPRegistration(
            voxel_size=icp_voxel_size,
            max_correspondence_distance=icp_max_correspondence_distance,
        )

    # ------------------------------------------------------------------

    def find_candidates(
        self,
        store: KeyframeStore,
        query: Keyframe,
    ) -> List[int]:
        """
        Return keyframe indices within ``search_radius`` of *query* that
        are at least ``min_time_gap`` keyframes older.
        """
        if len(store) == 0:
            return []
        positions = store.positions()
        q = _pose_translation(query.pose)
        diffs = positions - q
        dists = np.linalg.norm(diffs, axis=1)
        out = []
        for i, d in enumerate(dists):
            if i == query.index:
                continue
            if query.index - i < self._gap:
                continue
            if d <= self._radius:
                out.append(int(i))
        # Sort by distance for deterministic ordering
        out.sort(key=lambda i: dists[i])
        return out

    def verify(
        self,
        candidate: Keyframe,
        query: Keyframe,
    ) -> Optional[LoopEdge]:
        """
        Run ICP between *candidate* and *query* to verify a loop closure.

        Returns a :class:`LoopEdge` with the relative transform
        (candidate ← query) if ICP succeeds and meets the fitness/RMSE
        thresholds, otherwise ``None``.
        """
        # Initial guess: relative pose from odometry
        # T_cq = inv(T_world_c) @ T_world_q
        T_c_inv = np.linalg.inv(candidate.pose)
        init = T_c_inv @ query.pose

        result = self._icp.register(
            source=query.points,
            target=candidate.points,
            initial_transform=init,
        )
        if (result.fitness < self._fit
                or result.inlier_rmse > self._rmse
                or not result.converged):
            logger.debug(
                "Loop reject %d↔%d fitness=%.3f rmse=%.3f",
                candidate.index, query.index,
                result.fitness, result.inlier_rmse,
            )
            return None

        return LoopEdge(
            source_kf=candidate.index,
            target_kf=query.index,
            transform=np.asarray(result.transform, dtype=np.float64),
            fitness=float(result.fitness),
            inlier_rmse=float(result.inlier_rmse),
        )

    def detect(
        self,
        store: KeyframeStore,
        query: Keyframe,
    ) -> List[LoopEdge]:
        """
        Convenience: find_candidates → verify each, return all confirmed
        loop edges for *query*.
        """
        out = []
        for idx in self.find_candidates(store, query):
            edge = self.verify(store[idx], query)
            if edge is not None:
                out.append(edge)
        return out


# ---------------------------------------------------------------------------
# Pose graph optimisation with loop closures
# ---------------------------------------------------------------------------

def optimize_with_loop_closures(
    poses: Sequence[np.ndarray],
    loop_edges: Sequence[LoopEdge],
    max_correspondence_distance: float = 0.1,
    odometry_information: Optional[np.ndarray] = None,
    loop_information: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """
    Build a pose graph from sequential odometry + ``loop_edges`` and
    optimise it.

    Parameters
    ----------
    poses:
        Sequence of (4, 4) world ← node transforms (one per node).  This
        is typically the keyframe pose history.
    loop_edges:
        Verified :class:`LoopEdge` instances.  ``source_kf`` and
        ``target_kf`` must be valid indices into *poses*.
    max_correspondence_distance:
        Passed to the optimiser; controls how aggressively loops correct
        the graph.
    odometry_information, loop_information:
        Optional (6, 6) information matrices.  Defaults to identity for
        odometry and ``10 * identity`` for loop edges (loops are trusted
        more strongly than odometry).

    Returns
    -------
    Corrected list of (4, 4) world ← node poses, one per input node.
    """
    if not _O3D_AVAILABLE:
        raise ImportError("open3d is required for pose-graph optimisation.")
    if len(poses) < 2:
        return [np.asarray(p, dtype=np.float64).copy() for p in poses]

    n = len(poses)
    odom_info = (np.eye(6) if odometry_information is None
                 else np.asarray(odometry_information, dtype=np.float64))
    loop_info = (10.0 * np.eye(6) if loop_information is None
                 else np.asarray(loop_information, dtype=np.float64))

    # Build sequential relative transforms from absolute poses
    relatives: List[np.ndarray] = []
    informations: List[np.ndarray] = []
    for i in range(n - 1):
        T_i_inv = np.linalg.inv(np.asarray(poses[i], dtype=np.float64))
        T_rel = T_i_inv @ np.asarray(poses[i + 1], dtype=np.float64)
        relatives.append(T_rel)
        informations.append(odom_info)

    graph = build_pose_graph(relatives, informations)

    # Append loop closure edges (marked uncertain so the optimiser can
    # down-weight outliers via edge pruning)
    for edge in loop_edges:
        if not (0 <= edge.source_kf < n and 0 <= edge.target_kf < n):
            raise IndexError(
                f"loop edge ({edge.source_kf}, {edge.target_kf}) "
                f"out of range for {n} poses"
            )
        if edge.source_kf == edge.target_kf:
            continue
        graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                source_node_id=int(edge.source_kf),
                target_node_id=int(edge.target_kf),
                transformation=np.asarray(edge.transform, dtype=np.float64),
                information=loop_info,
                uncertain=True,
            )
        )

    optimise_pose_graph(
        graph,
        max_correspondence_distance=max_correspondence_distance,
    )

    # Anchor the optimised graph to the original first pose so absolute
    # poses are recovered (Open3D's reference_node=0 fixes node 0 at
    # identity in graph-space).
    anchor = np.asarray(poses[0], dtype=np.float64)
    return [anchor @ np.asarray(node.pose, dtype=np.float64)
            for node in graph.nodes]


# ---------------------------------------------------------------------------
# Convenience: apply corrected keyframe poses back to a dense pose list
# ---------------------------------------------------------------------------

def interpolate_correction(
    original_keyframe_poses: Sequence[np.ndarray],
    corrected_keyframe_poses: Sequence[np.ndarray],
    original_dense_poses: Sequence[np.ndarray],
    keyframe_indices: Sequence[int],
) -> List[np.ndarray]:
    """
    Propagate per-keyframe corrections to a dense pose trajectory.

    Each dense pose between two consecutive keyframes is updated as::

        T_corrected = T_kf_corrected @ inv(T_kf_original) @ T_dense_original

    where ``T_kf`` is the most recent keyframe at or before that dense
    sample.  Dense poses past the last keyframe inherit its correction.

    Parameters
    ----------
    original_keyframe_poses:
        Keyframe poses before optimisation (one per keyframe).
    corrected_keyframe_poses:
        Keyframe poses after optimisation (same length).
    original_dense_poses:
        Original per-scan trajectory.
    keyframe_indices:
        ``original_dense_poses`` index for each keyframe.  Must be
        monotonically non-decreasing.

    Returns
    -------
    List of (4, 4) corrected dense poses (same length as
    ``original_dense_poses``).
    """
    if len(original_keyframe_poses) != len(corrected_keyframe_poses):
        raise ValueError("keyframe pose lists must have the same length")
    if len(keyframe_indices) != len(original_keyframe_poses):
        raise ValueError("keyframe_indices must match keyframe pose count")

    n = len(original_dense_poses)
    corrected: List[np.ndarray] = [None] * n  # type: ignore[list-item]

    if not original_keyframe_poses:
        return [np.asarray(p, dtype=np.float64).copy()
                for p in original_dense_poses]

    # Precompute per-keyframe correction T_corr = T_kf_corr @ inv(T_kf_orig)
    corrections: List[np.ndarray] = []
    for orig, corr in zip(original_keyframe_poses,
                          corrected_keyframe_poses):
        corrections.append(
            np.asarray(corr, dtype=np.float64)
            @ np.linalg.inv(np.asarray(orig, dtype=np.float64))
        )

    kf_idx = 0
    for i in range(n):
        # Advance kf_idx while next keyframe is still <= current scan
        while (kf_idx + 1 < len(keyframe_indices)
               and keyframe_indices[kf_idx + 1] <= i):
            kf_idx += 1
        T = (corrections[kf_idx]
             @ np.asarray(original_dense_poses[i], dtype=np.float64))
        corrected[i] = T
    return corrected  # type: ignore[return-value]
