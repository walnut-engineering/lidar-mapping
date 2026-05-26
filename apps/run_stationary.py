"""
Phase 1 entry point — stationary rotation validation.

Brings up:
  * VLP-16 driver (UDP)
  * WitMotion IMU driver (UART /dev/ttyS1 @ 230400)
  * Camera (cv2.VideoCapture, optional)
  * SensorHub: ingests all three
  * LidarFrontend: IMU-priored scan-to-map ICP → FusionState
  * LivePointCloudViewer: operator-facing 3D + camera view
  * Observability HTTP server on :8765 for agent introspection

Run::
    python3 -m apps.run_stationary

While running, the agent can call:
    GET http://<pi>:8765/snapshot/composite
    GET http://<pi>:8765/state
    GET http://<pi>:8765/pose
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

# Vispy GLFW + X11 setup *before* vispy import
os.environ.setdefault("VISPY_BACKEND", "glfw")
os.environ.setdefault("XDG_SESSION_TYPE", "x11")
if "DISPLAY" not in os.environ:
    os.environ["DISPLAY"] = ":0"
if "WAYLAND_DISPLAY" in os.environ:
    del os.environ["WAYLAND_DISPLAY"]

import cv2
import numpy as np

from lidar_mapping.fusion.calibration import CalibrationConfig
from lidar_mapping.fusion.lidar_frontend import LidarFrontend
from lidar_mapping.fusion.sensor_hub import SensorHub
from lidar_mapping.fusion.visual_frontend import VisualFrontend
from lidar_mapping.observability.server import serve_in_thread
from lidar_mapping.observability.state import get_state
from lidar_mapping.sensors.imu import WitMotionDriver
from lidar_mapping.sensors.vlp16 import VLP16Driver

logger = logging.getLogger("apps.run_stationary")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stationary rotation fusion test")
    p.add_argument("--lidar-host", default="0.0.0.0")
    p.add_argument("--lidar-port", type=int, default=2368)
    p.add_argument("--imu-port", default="/dev/ttyS1")
    p.add_argument("--imu-baud", type=int, default=230400)
    p.add_argument("--camera-index", default="0",
                   help="cv2 VideoCapture spec; integer index, path like /dev/video0, or -1 to disable")
    p.add_argument("--http-port", type=int, default=8765)
    p.add_argument("--no-viewer", action="store_true",
                   help="Run headless (no VisPy window) — useful over SSH")
    p.add_argument("--no-visual", action="store_true",
                   help="Disable the ORB visual frontend")
    p.add_argument("--cam-width", type=int, default=1280)
    p.add_argument("--cam-height", type=int, default=720)
    p.add_argument("--cam-fov", type=float, default=70.0,
                   help="Camera horizontal FOV (deg) used for default intrinsics")
    p.add_argument("--calibration-yaml", type=str, default=None,
                   help="Optional path to saved calibration YAML to preload")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    state = get_state()
    state.running = True

    # ------------------------- sensors -------------------------
    logger.info("Starting VLP-16 driver on %s:%d", args.lidar_host, args.lidar_port)
    lidar = VLP16Driver(host=args.lidar_host, port=args.lidar_port)
    lidar.start()

    logger.info("Starting WitMotion IMU on %s @ %d", args.imu_port, args.imu_baud)
    imu = WitMotionDriver(port=args.imu_port, baudrate=args.imu_baud)
    imu.start()

    cam = None
    cam_spec = args.camera_index
    cam_disabled = (cam_spec == "-1")
    if not cam_disabled:
        # Accept integer index or device path string. Some V4L2 drivers reject
        # opening by integer; opening by path string is more robust.
        logger.info("Opening camera %r", cam_spec)
        try:
            cam = cv2.VideoCapture(int(cam_spec))
            if not cam.isOpened():
                cam.release()
                cam = cv2.VideoCapture(f"/dev/video{int(cam_spec)}")
        except ValueError:
            cam = cv2.VideoCapture(cam_spec)
        if cam.isOpened():
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
        else:
            logger.warning("Camera %r not opened; continuing without", cam_spec)
            cam = None

    # ------------------------- hub -----------------------------
    hub = SensorHub()
    hub.start_lidar_ingest(lidar)
    hub.start_imu_ingest(imu)
    if cam is not None:
        def cam_getter():
            ok, frame = cam.read()
            if not ok:
                return None
            state.set_camera(frame)
            return frame
        hub.start_camera_ingest(cam_getter)

    # ------------------------- frontends -----------------------
    frontend = LidarFrontend(hub=hub, state=state)
    frontend.start()

    visual = None
    if cam is not None and not args.no_visual:
        if args.calibration_yaml:
            calib = CalibrationConfig.load_yaml(args.calibration_yaml)
            logger.info("Loaded calibration from %s", args.calibration_yaml)
        else:
            calib = CalibrationConfig.default(
                width=args.cam_width, height=args.cam_height, fov_deg=args.cam_fov,
            )
        state.calibration = calib
        visual = VisualFrontend(hub=hub, calibration=calib, state=state)
        visual.start()
        logger.info("VisualFrontend running (ORB + LiDAR-depth + PnP)")

    # ------------------------- observability -------------------
    serve_in_thread(host="0.0.0.0", port=args.http_port, state=state)
    logger.info("Observability: http://0.0.0.0:%d/snapshot/composite",
                args.http_port)

    # ------------------------- viewer --------------------------
    if not args.no_viewer:
        from lidar_mapping.visualization.viewer import LivePointCloudViewer

        viewer = LivePointCloudViewer(
            title="Fusion (Phase 1)", width=1280, height=720,
        )

        def fetch():
            with state.lock:
                pts = state.latest_cloud.copy() if state.latest_cloud is not None else None
                bgr = state.latest_camera_bgr
                rgb = bgr[..., ::-1].copy() if bgr is not None else None
                pose = state.current_pose.copy()
            if pts is not None:
                # Visualize in world frame so rotation cancels out when working
                xyz = pts[:, :3] @ pose[:3, :3].T + pose[:3, 3]
                vis = np.hstack([xyz, pts[:, 3:4]]) if pts.shape[1] >= 4 else xyz
                return (vis, rgb)
            return (None, rgb)

        viewer.set_callback(fetch)
        try:
            viewer.start()
        except KeyboardInterrupt:
            pass
        finally:
            _shutdown(state, frontend, visual, hub, lidar, imu, cam)
    else:
        logger.info("Running headless; Ctrl-C to stop.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            _shutdown(state, frontend, visual, hub, lidar, imu, cam)


def _shutdown(state, frontend, visual, hub, lidar, imu, cam) -> None:
    state.running = False
    logger.info("Stopping frontends / hub / sensors")
    if visual is not None:
        visual.stop()
    frontend.stop()
    hub.stop()
    lidar.stop()
    imu.stop()
    if cam is not None:
        cam.release()
    logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
