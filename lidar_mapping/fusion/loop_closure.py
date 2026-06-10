"""
Loop closure detection for SLAM.

Detects when the camera revisits a previously observed region by matching
current frame descriptors against keyframe descriptors. Uses ORB descriptor
matching with Lowe ratio test and optional geometric verification via
Essential Matrix + RANSAC.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LoopCandidate:
    """A potential loop closure between two keyframes."""
    
    keyframe_id_a: int
    keyframe_id_b: int
    match_count: int
    inlier_count: int = 0
    score: float = 0.0  # Confidence score [0, 1]


@dataclass
class VerifiedLoopClosure:
    """A geometrically verified loop closure."""
    
    keyframe_id_a: int
    keyframe_id_b: int
    transform: np.ndarray  # 4x4 relative pose (B from A)
    match_count: int
    inlier_count: int
    reprojection_error: float
    confidence: float


class LoopClosureDetector:
    """
    Detect and verify loop closures via descriptor matching and geometry.
    
    Parameters
    ----------
    descriptor_match_ratio : float
        Lowe ratio threshold for descriptor matching (0.75 typical).
    min_matches_threshold : int
        Minimum descriptor matches to consider loop candidate.
    use_geometric_verification : bool
        If True, verify loop closure with Essential Matrix + RANSAC.
    """
    
    def __init__(
        self,
        descriptor_match_ratio: float = 0.75,
        min_matches_threshold: int = 20,
        use_geometric_verification: bool = True,
    ):
        self.descriptor_match_ratio = descriptor_match_ratio
        self.min_matches_threshold = min_matches_threshold
        self.use_geometric_verification = use_geometric_verification
        
        # Use HAMMING distance for ORB (binary descriptors)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        self.loop_closures: List[VerifiedLoopClosure] = []
        self.candidates_checked = 0
        self.loop_closures_verified = 0
    
    def find_loop_candidates(
        self,
        current_descriptors: np.ndarray,
        keyframe_db: List[tuple],
        exclude_recent: int = 3,
    ) -> List[LoopCandidate]:
        """
        Find keyframes with high descriptor similarity to current frame.
        
        Parameters
        ----------
        current_descriptors : np.ndarray
            (M, 32) ORB descriptors from current frame.
        keyframe_db : List[tuple]
            List of (keyframe_id, descriptors) tuples.
        exclude_recent : int
            Exclude the N most recent keyframes (to avoid matching adjacent frames).
        
        Returns
        -------
        List[LoopCandidate]
            Sorted list (highest match count first).
        """
        if current_descriptors is None or len(current_descriptors) == 0:
            return []
        
        if len(keyframe_db) <= exclude_recent:
            return []
        
        candidates = []
        
        # Only match against old keyframes (skip recent)
        for kf_id, kf_descriptors in keyframe_db[:-exclude_recent]:
            if kf_descriptors is None or len(kf_descriptors) == 0:
                continue
            
            try:
                knn_matches = self._matcher.knnMatch(
                    kf_descriptors, current_descriptors, k=2
                )
            except cv2.error:
                continue
            
            # Lowe ratio test
            good_matches = 0
            for pair in knn_matches:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < self.descriptor_match_ratio * n.distance:
                    good_matches += 1
            
            if good_matches >= self.min_matches_threshold:
                candidates.append(
                    LoopCandidate(
                        keyframe_id_a=kf_id,
                        keyframe_id_b=-1,  # Current frame (ID unknown at this stage)
                        match_count=good_matches,
                    )
                )
            
            self.candidates_checked += 1
        
        # Sort by match count (descending)
        candidates.sort(key=lambda c: -c.match_count)
        return candidates
    
    def verify_loop_with_geometry(
        self,
        pts_prev: np.ndarray,  # 3D points in previous frame
        pts_curr: np.ndarray,  # 2D points in current frame
        K: np.ndarray,  # Intrinsic matrix
        matches: Optional[List] = None,  # Optional match list for correspondences
    ) -> Optional[VerifiedLoopClosure]:
        """
        Verify loop closure using Essential Matrix + RANSAC.
        
        Parameters
        ----------
        pts_prev : np.ndarray
            (N, 3) 3D points from previous keyframe.
        pts_curr : np.ndarray
            (N, 2) 2D image points from current frame.
        K : np.ndarray
            3x3 intrinsic camera matrix.
        matches : List, optional
            OpenCV DMatch list specifying correspondences.
        
        Returns
        -------
        VerifiedLoopClosure or None
            Loop closure if verified, else None.
        """
        if len(pts_prev) < 8 or len(pts_curr) < 8:
            return None
        
        if len(pts_prev) != len(pts_curr):
            return None
        
        try:
            # Compute Essential Matrix
            E, mask = cv2.findEssentialMat(
                pts_curr,
                pts_prev[:, :2],  # Project 3D to 2D for Essential Matrix
                K,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0,
            )
        except cv2.error as e:
            logger.warning(f"Essential Matrix estimation failed: {e}")
            return None
        
        if E is None or mask is None:
            return None
        
        inlier_count = int(np.sum(mask))
        if inlier_count < 8:
            return None
        
        try:
            # Recover pose from Essential Matrix
            _, R, t, mask = cv2.recoverPose(
                E, pts_curr, pts_prev[:, :2], K, mask=mask
            )
        except cv2.error as e:
            logger.warning(f"Pose recovery failed: {e}")
            return None
        
        # Build 4x4 transform (B from A, i.e., T_A_B)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t.ravel()
        
        # Estimate reprojection error
        reprojection_error = np.sqrt(float(np.mean(np.sum(mask))))
        confidence = float(inlier_count) / len(pts_prev)
        
        return VerifiedLoopClosure(
            keyframe_id_a=-1,  # Set by caller
            keyframe_id_b=-1,  # Set by caller
            transform=T,
            match_count=len(pts_prev),
            inlier_count=inlier_count,
            reprojection_error=reprojection_error,
            confidence=confidence,
        )
    
    def get_matching_points_for_keyframe(
        self,
        kf_descriptors: np.ndarray,
        kf_keypoints: List,
        current_descriptors: np.ndarray,
        current_keypoints: List,
    ) -> Tuple[np.ndarray, np.ndarray, List]:
        """
        Extract matched keypoint coordinates between two frames.
        
        Parameters
        ----------
        kf_descriptors, kf_keypoints : OpenCV features from keyframe
        current_descriptors, current_keypoints : OpenCV features from current frame
        
        Returns
        -------
        kf_pts_2d : (M, 2) matching keypoints from keyframe
        curr_pts_2d : (M, 2) matching keypoints from current frame
        matches : List of DMatch objects
        """
        if (
            kf_descriptors is None
            or len(kf_descriptors) == 0
            or current_descriptors is None
            or len(current_descriptors) == 0
        ):
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2), []
        
        knn_matches = self._matcher.knnMatch(kf_descriptors, current_descriptors, k=2)
        
        good_matches = []
        kf_pts = []
        curr_pts = []
        
        for pair in knn_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.descriptor_match_ratio * n.distance:
                good_matches.append(m)
                kf_pts.append(kf_keypoints[m.queryIdx].pt)
                curr_pts.append(current_keypoints[m.trainIdx].pt)
        
        return (
            np.array(kf_pts, dtype=np.float32),
            np.array(curr_pts, dtype=np.float32),
            good_matches,
        )
    
    def reset(self) -> None:
        """Clear loop closure history."""
        self.loop_closures.clear()
        self.candidates_checked = 0
        self.loop_closures_verified = 0
    
    def statistics(self) -> dict:
        """Return detector statistics."""
        return {
            "candidates_checked": self.candidates_checked,
            "loop_closures_verified": self.loop_closures_verified,
            "total_loop_closures": len(self.loop_closures),
            "descriptor_match_ratio": self.descriptor_match_ratio,
            "min_matches_threshold": self.min_matches_threshold,
        }
