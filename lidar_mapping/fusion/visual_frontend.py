"""
Visual frontend — ORB features with LiDAR-depth association and PnP.

Per camera frame, this module:
    1. Detects ORB keypoints and descriptors.
    2. Projects the most recent LiDAR point cloud into the image and
       assigns sparse depth to keypoints by nearest-neighbor lookup.
    3. Matches against the previous frame's descriptors (BFMatcher +
       Lowe ratio + RANSAC).
    4. Estimates 6-DoF camera motion via solvePnPRansac on 3D-2D
       correspondences.
    5. Renders a debug overlay (keypoints + matches + projected LiDAR)
       and publishes the latest pose estimate / overlay into
       :class:`FusionState`.

Designed to be robust to missing data:
    * No LiDAR frame yet → keypoints detected but PnP skipped.
    * No previous frame yet → only feature extraction performed.
    * PnP failure → fall back to identity motion this tick.

The visual pose is reported as a delta-pose between consecutive camera
frames and a cumulative camera trajectory. Phase 4 will fuse this with
the IMU-driven LiDAR pose inside the pose-graph backend.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

from lidar_mapping.fusion.calibration import CalibrationConfig
from lidar_mapping.fusion.keyframe_selector import KeyframeSelector, Keyframe
from lidar_mapping.fusion.loop_closure import LoopClosureDetector
from lidar_mapping.fusion.pose_graph_backend import PoseGraphOptimizer, Pose4DOF
from lidar_mapping.fusion.pose_helpers import (
    extract_4dof_from_se3, se3_from_4dof, pose_4dof_delta
)
from lidar_mapping.fusion.sensor_hub import SensorHub
from lidar_mapping.observability.state import FusionState, LoopConstraint, get_state

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def project_lidar_to_image(
    points_lidar: np.ndarray,
    T_cam_lidar: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    min_depth: float = 0.3,
    max_depth: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project an (N,3) LiDAR cloud into image pixels.

    Returns
    -------
    uv : (M, 2) float32 pixel coordinates of points that landed inside
        the image and have positive depth in the camera frame.
    z  : (M,)   float32 depths (camera-frame Z) for those pixels.
    """
    if points_lidar.size == 0:
        return np.empty((0, 2), np.float32), np.empty((0,), np.float32)

    # Transform LiDAR points into camera frame.
    R = T_cam_lidar[:3, :3]
    t = T_cam_lidar[:3, 3]
    pts_cam = points_lidar @ R.T + t  # (N,3)

    z = pts_cam[:, 2]
    valid = (z > min_depth) & (z < max_depth)
    pts_cam = pts_cam[valid]
    z = z[valid]
    if pts_cam.shape[0] == 0:
        return np.empty((0, 2), np.float32), np.empty((0,), np.float32)

    # Pinhole projection.
    uv_h = pts_cam @ K.T
    uv = uv_h[:, :2] / uv_h[:, 2:3]

    in_img = (
        (uv[:, 0] >= 0) & (uv[:, 0] < width) &
        (uv[:, 1] >= 0) & (uv[:, 1] < height)
    )
    return uv[in_img].astype(np.float32), z[in_img].astype(np.float32)


def assign_depth_to_keypoints(
    keypoints: list,
    proj_uv: np.ndarray,
    proj_z: np.ndarray,
    radius_px: float = 8.0,
) -> np.ndarray:
    """For each keypoint return the depth of the nearest projected LiDAR
    pixel within ``radius_px``, or NaN if none.

    Implementation: 2D KD-tree-free brute force with bucketing on a
    coarse pixel grid — fast enough for ~1500 keypoints and ~10k LiDAR
    projections per frame.
    """
    if len(keypoints) == 0 or proj_uv.shape[0] == 0:
        return np.full((len(keypoints),), np.nan, dtype=np.float32)

    kp_xy = np.array([kp.pt for kp in keypoints], dtype=np.float32)  # (K,2)

    # Bucket projected pixels by integer cell of size ~radius.
    cell = max(int(radius_px), 1)
    keys = (proj_uv / cell).astype(np.int32)
    buckets: dict[Tuple[int, int], list[int]] = {}
    for idx, (cx, cy) in enumerate(keys):
        buckets.setdefault((int(cx), int(cy)), []).append(idx)

    out = np.full((kp_xy.shape[0],), np.nan, dtype=np.float32)
    r2 = radius_px * radius_px
    for i, (kx, ky) in enumerate(kp_xy):
        cx, cy = int(kx // cell), int(ky // cell)
        best_d2, best_z = r2, np.nan
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                bucket = buckets.get((cx + dx, cy + dy))
                if not bucket:
                    continue
                for idx in bucket:
                    px, py = proj_uv[idx]
                    d2 = (px - kx) ** 2 + (py - ky) ** 2
                    if d2 < best_d2:
                        best_d2 = d2
                        best_z = proj_z[idx]
        out[i] = best_z
    return out


def draw_overlay(
    bgr: np.ndarray,
    keypoints: list,
    proj_uv: np.ndarray,
    proj_z: np.ndarray,
    matches_xy0: Optional[np.ndarray] = None,
    matches_xy1: Optional[np.ndarray] = None,
    pnp_inliers: Optional[np.ndarray] = None,
    hud: Optional[list] = None,
) -> np.ndarray:
    """Draw the debug overlay onto a BGR copy of the image."""
    img = bgr.copy()
    h, w = img.shape[:2]

    # Projected LiDAR points coloured by depth (near=red → far=blue).
    if proj_uv.shape[0] > 0:
        z_min, z_max = 0.5, 25.0
        zc = np.clip((proj_z - z_min) / (z_max - z_min), 0.0, 1.0)
        for (u, v), c in zip(proj_uv, zc):
            color = (int(255 * c), 64, int(255 * (1 - c)))
            cv2.circle(img, (int(u), int(v)), 1, color, -1, cv2.LINE_AA)

    # Keypoints (small green dots).
    for kp in keypoints:
        cv2.circle(img, (int(kp.pt[0]), int(kp.pt[1])), 2,
                   (0, 255, 0), -1, cv2.LINE_AA)

    # Matches: draw lines from previous frame keypoints to current.
    if matches_xy0 is not None and matches_xy1 is not None:
        inlier_mask = (
            pnp_inliers if pnp_inliers is not None
            else np.ones(len(matches_xy0), dtype=bool)
        )
        for i, ((x0, y0), (x1, y1)) in enumerate(zip(matches_xy0, matches_xy1)):
            color = (0, 255, 255) if inlier_mask[i] else (40, 40, 200)
            cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)),
                     color, 1, cv2.LINE_AA)

    # HUD text.
    if hud:
        y = 20
        for line in hud:
            cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y += 18
    return img


# ---------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------
@dataclass
class _PrevFrame:
    timestamp: float
    keypoints: list
    descriptors: np.ndarray
    depths: np.ndarray  # (K,) NaN where unknown
    gray_shape: Tuple[int, int]


class VisualFrontend:
    """Background ORB + LiDAR-depth + PnP visual odometry worker.

    Reads camera frames from :class:`SensorHub`, the latest LiDAR cloud
    for depth association, and publishes a debug overlay + cumulative
    visual-only pose into the shared :class:`FusionState`.
    """

    def __init__(
        self,
        hub: SensorHub,
        calibration: Optional[CalibrationConfig] = None,
        state: Optional[FusionState] = None,
        n_features: int = 1500,
        match_ratio: float = 0.75,
        min_inliers: int = 12,
        depth_radius_px: float = 6.0,
        enable_loop_closure: bool = True,
    ) -> None:
        self.hub = hub
        self.calib = calibration or CalibrationConfig.default()
        self.state = state or get_state()
        self.n_features = int(n_features)
        self.match_ratio = float(match_ratio)
        self.min_inliers = int(min_inliers)
        self.depth_radius_px = float(depth_radius_px)
        self.enable_loop_closure = bool(enable_loop_closure)

        self._orb = cv2.ORB_create(nfeatures=self.n_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Stage 3: Loop closure components
        self._keyframe_selector = KeyframeSelector(
            motion_threshold_m=0.05,
            angular_threshold_deg=2.0,
        )
        self._loop_detector = LoopClosureDetector()
        self._pose_graph = PoseGraphOptimizer()
        self._keyframe_id_counter = 0
        self._keyframes: dict[int, Keyframe] = {}
        self._prev_keyframe_id: Optional[int] = None
        self._last_optimization_tick = 0

        # Share calibration via FusionState so it can be tuned live.
        if self.state.calibration is None:
            self.state.calibration = self.calib
        else:
            # Pick up whatever was already registered (e.g. by a previous run).
            self.calib = self.state.calibration

        self._prev: Optional[_PrevFrame] = None
        self._cam_pose = np.eye(4)  # cumulative camera pose, world ← cam
        self._last_camera_t: Optional[float] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.frames_processed = 0
        self.pnp_successes = 0
        self.pnp_failures = 0

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="visual-frontend"
        )
        self._thread.start()
        logger.info("VisualFrontend started.")

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def get_keyframe_count(self) -> int:
        """Get number of keyframes extracted."""
        return len(self._keyframes)

    def get_pose_graph_size(self) -> Tuple[int, int]:
        """Get (num_poses, num_factors) in pose graph."""
        return (
            len(self._pose_graph.poses),
            len(self._pose_graph.factors),
        )

    def get_pose_graph_trajectory(self) -> Optional[np.ndarray]:
        """Get optimized trajectory from pose graph."""
        if len(self._pose_graph.poses) == 0:
            return None
        return self._pose_graph.get_trajectory()

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while self._running:
            try:
                self._tick()
            except Exception as exc:  # noqa: BLE001
                logger.exception("VisualFrontend tick error: %s", exc)
                time.sleep(0.05)

    def _tick(self) -> None:
        latest = self.hub.camera.latest()
        if latest is None:
            time.sleep(0.02)
            return
        t_curr, bgr = latest
        if self._last_camera_t is not None and t_curr <= self._last_camera_t:
            time.sleep(0.02)
            return

        # Pick up latest live calibration (mutated via HTTP).
        if self.state.calibration is not None:
            self.calib = self.state.calibration

        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Restrict ORB detection to a central horizontal band. Wide-FOV
        # lenses violate the no-distortion pinhole model near the edges,
        # so PnP rejects features there. LiDAR projection / overlay /
        # depth lookup still use the full image and full K.
        frac = float(getattr(self.calib, "vo_center_fraction", 1.0) or 1.0)
        frac = max(0.05, min(1.0, frac))
        if frac < 0.999:
            band = int(round(w * frac))
            x0 = (w - band) // 2
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[:, x0:x0 + band] = 255
        else:
            mask = None
        keypoints, descriptors = self._orb.detectAndCompute(gray, mask)
        keypoints = list(keypoints) if keypoints is not None else []

        # LiDAR projection for depth (latest frame in hub).
        proj_uv = np.empty((0, 2), np.float32)
        proj_z = np.empty((0,), np.float32)
        depths = np.full((len(keypoints),), np.nan, dtype=np.float32)
        lidar_latest = self.hub.lidar.latest()
        if lidar_latest is not None:
            _, lidar_frame = lidar_latest
            cloud = lidar_frame.to_numpy()[:, :3]
            proj_uv, proj_z = project_lidar_to_image(
                cloud, self.calib.T_cam_lidar, self.calib.K, w, h,
            )
            if len(keypoints) > 0 and proj_uv.shape[0] > 0:
                depths = assign_depth_to_keypoints(
                    keypoints, proj_uv, proj_z, self.depth_radius_px,
                )

        # PnP against previous keyframe.
        matches_xy0 = matches_xy1 = None
        pnp_inliers = None
        delta = np.eye(4)
        n_inliers = 0
        prev = self._prev
        if (
            prev is not None
            and descriptors is not None
            and len(keypoints) >= self.min_inliers
            and prev.descriptors is not None
        ):
            try:
                knn = self._matcher.knnMatch(prev.descriptors, descriptors, k=2)
            except cv2.error:
                knn = []
            good = []
            for pair in knn:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < self.match_ratio * n.distance:
                    good.append(m)

            # 3D (prev) ↔ 2D (curr) with valid depth on prev side.
            pts3d: list = []
            pts2d: list = []
            xy0: list = []
            xy1: list = []
            for m in good:
                d0 = prev.depths[m.queryIdx]
                if not np.isfinite(d0):
                    continue
                # Back-project prev keypoint to 3D in prev camera frame.
                u0, v0 = prev.keypoints[m.queryIdx].pt
                u1, v1 = keypoints[m.trainIdx].pt
                K = self.calib.K
                x = (u0 - K[0, 2]) * d0 / K[0, 0]
                y = (v0 - K[1, 2]) * d0 / K[1, 1]
                pts3d.append([x, y, float(d0)])
                pts2d.append([u1, v1])
                xy0.append([u0, v0])
                xy1.append([u1, v1])

            if len(pts3d) >= self.min_inliers:
                obj = np.array(pts3d, dtype=np.float32)
                img_pts = np.array(pts2d, dtype=np.float32)
                ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                    obj, img_pts, self.calib.K, self.calib.dist,
                    iterationsCount=100, reprojectionError=3.0,
                    flags=cv2.SOLVEPNP_EPNP,
                )
                if ok and inliers is not None and len(inliers) >= self.min_inliers:
                    n_inliers = int(len(inliers))
                    Rcw, _ = cv2.Rodrigues(rvec)
                    # solvePnP returns transform mapping prev-frame 3D
                    # points into the current camera frame:
                    #   p_curr = Rcw @ p_prev + tvec
                    # So delta_curr_prev = [Rcw|tvec; 0 1]
                    delta = np.eye(4)
                    delta[:3, :3] = Rcw
                    delta[:3, 3] = tvec.ravel()
                    # Cumulative: T_world_curr = T_world_prev @ inv(delta)
                    self._cam_pose = self._cam_pose @ np.linalg.inv(delta)
                    self.pnp_successes += 1
                    matches_xy0 = np.array(xy0, dtype=np.float32)
                    matches_xy1 = np.array(xy1, dtype=np.float32)
                    mask = np.zeros(len(xy0), dtype=bool)
                    mask[inliers.ravel()] = True
                    pnp_inliers = mask
                else:
                    self.pnp_failures += 1
                    matches_xy0 = np.array(xy0, dtype=np.float32) if xy0 else None
                    matches_xy1 = np.array(xy1, dtype=np.float32) if xy1 else None

        # --- Stage 3: Keyframe + Loop Closure Integration ---
        if self.enable_loop_closure and descriptors is not None:
            # Convert current pose to 4-DOF
            pose_4dof_curr = extract_4dof_from_se3(self._cam_pose)
            
            # Check if this should be a keyframe
            is_keyframe = self._keyframe_selector.should_be_keyframe(self._cam_pose)
            
            if is_keyframe and len(keypoints) > 0:
                # Create and store keyframe
                keyframe = self._keyframe_selector.add_keyframe(
                    pose=self._cam_pose,
                    descriptors=descriptors,
                    timestamp=t_curr,
                    keypoints=keypoints,
                    image_shape=(h, w),
                )
                kf_id = keyframe.keyframe_id
                self._keyframes[kf_id] = keyframe

                with self.state.lock:
                    self.state.keyframe_count += 1
                
                # Add first keyframe to pose graph
                if self._prev_keyframe_id is None:
                    self._pose_graph.add_pose(kf_id, Pose4DOF(*pose_4dof_curr))
                else:
                    # Add odometry factor from previous keyframe
                    prev_pose_4dof = extract_4dof_from_se3(
                        self._keyframes[self._prev_keyframe_id].pose
                    )
                    delta_4dof = pose_4dof_delta(prev_pose_4dof, pose_4dof_curr)
                    
                    # Create odometry factor with high confidence (VO is good)
                    info_matrix = np.eye(4) * 10.0  # 10x weight on odometry
                    self._pose_graph.add_pose(kf_id, Pose4DOF(*pose_4dof_curr))
                    self._pose_graph.add_factor(
                        self._prev_keyframe_id,
                        kf_id,
                        delta_4dof,
                        info_matrix,
                    )
                    
                    # Check for loop closures
                    keyframe_db = [
                        (kf.keyframe_id, kf.descriptors)
                        for kf in self._keyframe_selector.keyframes
                    ]
                    loop_candidates = self._loop_detector.find_loop_candidates(
                        descriptors,
                        keyframe_db,
                        exclude_recent=3,
                    )
                    
                    for candidate in loop_candidates:
                        prev_kf = self._keyframe_selector.get_keyframe(
                            candidate.keyframe_id_a
                        )
                        if prev_kf is None:
                            continue

                        pts_prev, pts_curr, matches = (
                            self._loop_detector.get_matching_points_for_keyframe(
                                prev_kf.descriptors,
                                prev_kf.keypoints or [],
                                descriptors,
                                keypoints,
                            )
                        )

                        # Verify with geometry (Essential Matrix + RANSAC)
                        verified = self._loop_detector.verify_loop_with_geometry(
                            pts_prev,
                            pts_curr,
                            self.calib.K,
                            matches=matches,
                        )
                        
                        if verified is not None:
                            # Add loop constraint to pose graph
                            loop_delta = extract_4dof_from_se3(verified.transform)
                            
                            # Loop closures get moderate confidence
                            loop_info = np.eye(4) * 5.0
                            self._pose_graph.add_factor(
                                candidate.keyframe_id_a,
                                kf_id,
                                loop_delta,
                                loop_info,
                            )
                            
                            # Track in state
                            with self.state.lock:
                                self.state.loop_constraints.append(LoopConstraint(
                                    keyframe_id_a=candidate.keyframe_id_a,
                                    keyframe_id_b=kf_id,
                                    transform=verified.transform,
                                    match_count=verified.match_count,
                                    inlier_count=verified.inlier_count,
                                    confidence=verified.confidence,
                                    timestamp=t_curr,
                                ))
                                self.state.loop_constraint_count += 1
                            
                            logger.info(
                                f"Loop closure detected: KF{candidate.keyframe_id_a} "
                                f"→ KF{kf_id}, inliers={verified.inlier_count}"
                            )
                
                self._prev_keyframe_id = kf_id
                
                # Optimize pose graph periodically (every 5 keyframes)
                if kf_id % 5 == 0:
                    self._pose_graph.optimize(max_iterations=3)
                    self._last_optimization_tick = self.frames_processed

        # Build overlay.
        hud = [
            f"kp={len(keypoints)}  proj_lidar={proj_uv.shape[0]}  "
            f"depth_kp={int(np.isfinite(depths).sum())}",
            f"pnp_inliers={n_inliers}  ok={self.pnp_successes} fail={self.pnp_failures}",
        ]
        overlay = draw_overlay(
            bgr, keypoints, proj_uv, proj_z,
            matches_xy0=matches_xy0,
            matches_xy1=matches_xy1,
            pnp_inliers=pnp_inliers,
            hud=hud,
        )

        with self.state.lock:
            raw = self.state.latest_camera_bgr
            # In dual-camera mode run_stationary publishes a stitched overlay.
            # Do not clobber it with a single-camera overlay of different size.
            if raw is None or raw.shape == overlay.shape:
                self.state.latest_camera_overlay_bgr = overlay
            self.state.camera_frames_total += 1
            self.state.rates.camera_hz = float(self.hub.camera_hz)

        # Advance state.
        self._prev = _PrevFrame(
            timestamp=t_curr,
            keypoints=keypoints,
            descriptors=descriptors,
            depths=depths,
            gray_shape=gray.shape,
        )
        self._last_camera_t = t_curr
        self.frames_processed += 1
