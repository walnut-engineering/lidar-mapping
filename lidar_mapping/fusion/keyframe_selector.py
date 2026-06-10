"""
Keyframe selection for pose-graph SLAM.

Extracts sparse keyframes from the dense visual odometry stream based on
motion thresholds. Keyframes serve as nodes in the pose graph and are used
for loop closure matching.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Keyframe:
    """A selected keyframe for loop closure and pose graph optimization."""
    
    keyframe_id: int
    pose: np.ndarray  # 4x4 SE(3) transform, world-from-camera
    timestamp: float
    descriptors: np.ndarray  # ORB descriptors, (N, 32) uint8
    keypoints: Optional[List[object]] = None  # OpenCV KeyPoint list
    image_shape: Optional[tuple] = None  # (height, width, channels)
    
    def __post_init__(self):
        """Validate keyframe consistency."""
        if self.pose.shape != (4, 4):
            raise ValueError(f"Pose must be 4x4, got {self.pose.shape}")
        if self.descriptors.ndim != 2 or self.descriptors.shape[1] != 32:
            raise ValueError(
                f"Descriptors must be (N, 32), got {self.descriptors.shape}"
            )


class KeyframeSelector:
    """
    Select keyframes from visual odometry stream based on motion threshold.
    
    Parameters
    ----------
    motion_threshold_m : float
        Minimum translation (meters) between keyframes.
    angular_threshold_deg : float
        Minimum rotation (degrees) between keyframes.
    
    Attributes
    ----------
    keyframes : List[Keyframe]
        All selected keyframes.
    keyframe_count : int
        Total keyframes created (ID counter).
    """
    
    def __init__(
        self,
        motion_threshold_m: float = 0.05,
        angular_threshold_deg: float = 2.0,
    ):
        self.motion_threshold = motion_threshold_m
        self.angular_threshold = np.radians(angular_threshold_deg)
        
        self.keyframes: List[Keyframe] = []
        self.keyframe_count = 0
        
        self._last_keyframe_pose: Optional[np.ndarray] = None
        self._frame_count = 0
    
    def should_be_keyframe(
        self,
        current_pose: np.ndarray,
        force_keyframe: bool = False,
    ) -> bool:
        """
        Determine if current pose warrants a new keyframe.
        
        Parameters
        ----------
        current_pose : np.ndarray
            Current 4x4 pose estimate (world-from-camera).
        force_keyframe : bool
            If True, create keyframe regardless of motion.
        
        Returns
        -------
        bool
            True if this frame should become a keyframe.
        """
        if force_keyframe or self._last_keyframe_pose is None:
            return True
        
        # Compute translation distance
        delta_t = current_pose[:3, 3] - self._last_keyframe_pose[:3, 3]
        translation_dist = np.linalg.norm(delta_t)
        
        if translation_dist >= self.motion_threshold:
            return True
        
        # Compute rotation angle
        delta_R = (
            self._last_keyframe_pose[:3, :3].T
            @ current_pose[:3, :3]
        )
        # Angle from trace: theta = arccos((trace(R) - 1) / 2)
        trace = np.trace(delta_R)
        trace_clipped = np.clip(trace, -1.0, 3.0)
        angle_rad = np.arccos((trace_clipped - 1.0) / 2.0)
        
        if angle_rad >= self.angular_threshold:
            return True
        
        return False
    
    def add_keyframe(
        self,
        pose: np.ndarray,
        descriptors: np.ndarray,
        timestamp: float,
        keypoints: Optional[List[object]] = None,
        image_shape: Optional[tuple] = None,
    ) -> Keyframe:
        """
        Register a new keyframe.
        
        Parameters
        ----------
        pose : np.ndarray
            4x4 SE(3) pose matrix.
        descriptors : np.ndarray
            (N, 32) ORB descriptors.
        timestamp : float
            Timestamp in seconds (monotonic).
        keypoints : List, optional
            OpenCV KeyPoint objects.
        image_shape : tuple, optional
            (height, width, channels) of source image.
        
        Returns
        -------
        Keyframe
            Newly created keyframe.
        """
        kf = Keyframe(
            keyframe_id=self.keyframe_count,
            pose=pose.copy(),
            timestamp=timestamp,
            descriptors=descriptors.copy(),
            keypoints=keypoints,
            image_shape=image_shape,
        )
        
        self.keyframes.append(kf)
        self._last_keyframe_pose = pose.copy()
        self.keyframe_count += 1
        
        logger.debug(
            f"Keyframe #{kf.keyframe_id} created at t={timestamp:.3f}s "
            f"with {descriptors.shape[0]} descriptors"
        )
        
        return kf
    
    def get_keyframe(self, keyframe_id: int) -> Optional[Keyframe]:
        """Retrieve keyframe by ID."""
        for kf in self.keyframes:
            if kf.keyframe_id == keyframe_id:
                return kf
        return None
    
    def get_recent_keyframes(self, count: int = 5) -> List[Keyframe]:
        """Return the most recent N keyframes."""
        return self.keyframes[-count:] if len(self.keyframes) > 0 else []
    
    def reset(self) -> None:
        """Clear all keyframes and reset counters."""
        self.keyframes.clear()
        self.keyframe_count = 0
        self._last_keyframe_pose = None
        self._frame_count = 0
    
    def statistics(self) -> dict:
        """Return selector statistics."""
        return {
            "total_keyframes": len(self.keyframes),
            "next_keyframe_id": self.keyframe_count,
            "motion_threshold_m": self.motion_threshold,
            "angular_threshold_deg": np.degrees(self.angular_threshold),
        }
