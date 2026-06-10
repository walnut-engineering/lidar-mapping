"""
Phase 1 entry point — stationary rotation validation.

Brings up:
  * VLP-16 driver (UDP)
  * WitMotion IMU driver (UART /dev/ttyS1 @ 230400)
    * One or more cameras (OpenCV via CameraCapture, optional)
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
import threading

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
from lidar_mapping.fusion.visual_frontend import (
    VisualFrontend,
    draw_overlay,
    project_lidar_to_image,
)
from lidar_mapping.observability.server import serve_in_thread
from lidar_mapping.observability.state import get_state
from lidar_mapping.sensors.camera import CameraCapture, CameraFrame
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
    p.add_argument("--camera-indices", default=None,
                   help="Comma-separated camera specs, e.g. '0,1' or '/dev/video0,/dev/video2'."
                        " Overrides --camera-index when provided.")
    p.add_argument("--camera-yaws-deg", default=None,
                   help="Comma-separated yaw offsets (deg) per camera in LiDAR frame,"
                        " e.g. '-10,10' for outward-facing stereo pair")
    p.add_argument("--primary-camera", type=int, default=0,
                   help="Index into --camera-indices used by visual frontend / hub camera stream")
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


def _parse_camera_specs(args: argparse.Namespace) -> list[str]:
    if args.camera_indices is not None:
        specs = [s.strip() for s in args.camera_indices.split(",") if s.strip()]
    else:
        specs = [args.camera_index.strip()]
    specs = [s for s in specs if s != "-1"]
    return specs


def _coerce_camera_device(spec: str) -> int | str:
    try:
        return int(spec)
    except ValueError:
        return spec


def _parse_float_list(spec: str | None, count: int, default: float = 0.0) -> list[float]:
    if spec is None:
        return [default for _ in range(count)]
    vals = []
    for s in spec.split(","):
        t = s.strip()
        if not t:
            continue
        vals.append(float(t))
    if not vals:
        vals = [default]
    if len(vals) < count:
        vals.extend([default] * (count - len(vals)))
    return vals[:count]


def _apply_lidar_yaw_to_T_cam_lidar(T_cam_lidar: np.ndarray, yaw_deg: float) -> np.ndarray:
    rz = np.deg2rad(float(yaw_deg))
    c, s = np.cos(rz), np.sin(rz)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    T = T_cam_lidar.copy()
    T[:3, :3] = T[:3, :3] @ Rz
    return T


def _stitch_frames_horizontally(frames: list[np.ndarray]) -> np.ndarray | None:
    if not frames:
        return None
    valid = [f for f in frames if f is not None]
    if not valid:
        return None
    base_h = valid[0].shape[0]
    resized: list[np.ndarray] = []
    for f in valid:
        if f.shape[0] != base_h:
            w = int(round(f.shape[1] * (base_h / float(f.shape[0]))))
            f = cv2.resize(f, (w, base_h), interpolation=cv2.INTER_AREA)
        resized.append(f)
    return np.hstack(resized)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    state = get_state()
    state.running = True

    # Shared calibration for visual frontend and dual-camera scene overlays.
    if args.calibration_yaml:
        calib = CalibrationConfig.load_yaml(args.calibration_yaml)
        logger.info("Loaded calibration from %s", args.calibration_yaml)
    else:
        calib = CalibrationConfig.default(
            width=args.cam_width, height=args.cam_height, fov_deg=args.cam_fov,
        )
    state.calibration = calib

    # ------------------------- sensors -------------------------
    logger.info("Starting VLP-16 driver on %s:%d", args.lidar_host, args.lidar_port)
    lidar = VLP16Driver(host=args.lidar_host, port=args.lidar_port)
    lidar.start()

    logger.info("Starting WitMotion IMU on %s @ %d", args.imu_port, args.imu_baud)
    imu = WitMotionDriver(port=args.imu_port, baudrate=args.imu_baud)
    imu.start()

    camera_specs = _parse_camera_specs(args)
    cameras: list[CameraCapture] = []
    if camera_specs:
        logger.info("Opening %d camera(s): %s", len(camera_specs), camera_specs)
    for spec in camera_specs:
        try:
            cam = CameraCapture(
                device_index=_coerce_camera_device(spec),
                width=args.cam_width,
                height=args.cam_height,
                fps=30,
                max_queue=4,
            )
            cam.start()
            cameras.append(cam)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Camera %r not opened; continuing without (%s)", spec, exc)

    # ------------------------- hub -----------------------------
    hub = SensorHub()
    hub.start_lidar_ingest(lidar)
    hub.start_imu_ingest(imu)
    cam_mux_stop = threading.Event()
    cam_mux_thread = None
    monitor_stop = threading.Event()
    monitor_thread = None

    if cameras:
        primary_idx = max(0, min(args.primary_camera, len(cameras) - 1))
        latest_frames: list[CameraFrame | None] = [None for _ in cameras]
        last_primary_idx = {"sent": -1}
        cam_yaws_deg = _parse_float_list(args.camera_yaws_deg, len(cameras), default=0.0)
        cam_Ts = [
            _apply_lidar_yaw_to_T_cam_lidar(calib.T_cam_lidar, y) for y in cam_yaws_deg
        ]
        scene_period_s = 0.5
        next_scene_t = {"t": 0.0}

        def cam_getter_primary():
            frame = latest_frames[primary_idx]
            if frame is None:
                return None
            if frame.frame_index == last_primary_idx["sent"]:
                return None
            last_primary_idx["sent"] = int(frame.frame_index)
            return frame.image

        def cam_state_mux_loop():
            while not cam_mux_stop.is_set():
                try:
                    updated = False
                    for i, cam in enumerate(cameras):
                        f = cam.get_latest_frame()
                        if f is not None:
                            latest_frames[i] = f
                            updated = True
                    if updated:
                        raw_frames = [f.image for f in latest_frames if f is not None]
                        stitched = _stitch_frames_horizontally(raw_frames)
                        if stitched is not None:
                            state.set_camera(stitched)

                            now = time.monotonic()
                            if now >= next_scene_t["t"]:
                                lidar_latest = hub.lidar.latest()
                                cloud_xyz = None
                                if lidar_latest is not None:
                                    _, lidar_frame = lidar_latest
                                    cloud_xyz = lidar_frame.to_numpy()[:, :3]

                                overlays: list[np.ndarray] = []
                                for i, fr in enumerate(latest_frames):
                                    if fr is None:
                                        continue
                                    bgr = fr.image
                                    keypoints = []
                                    proj_uv = np.empty((0, 2), np.float32)
                                    proj_z = np.empty((0,), np.float32)
                                    if cloud_xyz is not None and cloud_xyz.size > 0:
                                        h, w = bgr.shape[:2]
                                        proj_uv, proj_z = project_lidar_to_image(
                                            cloud_xyz,
                                            cam_Ts[i],
                                            calib.K,
                                            w,
                                            h,
                                        )
                                    hud = [
                                        f"cam{i} yaw={cam_yaws_deg[i]:+.1f}deg",
                                        f"proj={proj_uv.shape[0]}",
                                    ]
                                    overlays.append(
                                        draw_overlay(
                                            bgr,
                                            keypoints,
                                            proj_uv,
                                            proj_z,
                                            hud=hud,
                                        )
                                    )

                                stitched_overlay = _stitch_frames_horizontally(overlays)
                                if stitched_overlay is not None:
                                    state.set_camera(stitched, overlay_bgr=stitched_overlay)
                                next_scene_t["t"] = now + scene_period_s
                except Exception as exc:  # noqa: BLE001
                    logger.exception("camera mux loop error: %s", exc)
                time.sleep(0.01)

        hub.start_camera_ingest(cam_getter_primary)
        cam_mux_thread = threading.Thread(
            target=cam_state_mux_loop,
            daemon=True,
            name="camera-state-mux",
        )
        cam_mux_thread.start()

    def sensor_state_monitor_loop():
        last_imu_t = None
        while not monitor_stop.is_set():
            imu_latest = hub.imu.latest()
            if imu_latest is not None:
                t_imu, imu_reading = imu_latest
                if last_imu_t is None or t_imu > last_imu_t:
                    state.set_imu(imu_reading)
                    last_imu_t = t_imu
            with state.lock:
                state.rates.lidar_hz = float(hub.lidar_hz)
                state.rates.imu_hz = float(hub.imu_hz)
                state.rates.camera_hz = float(hub.camera_hz)
            time.sleep(0.02)

    monitor_thread = threading.Thread(
        target=sensor_state_monitor_loop,
        daemon=True,
        name="sensor-state-monitor",
    )
    monitor_thread.start()

    # ------------------------- frontends -----------------------
    frontend = LidarFrontend(hub=hub, state=state)
    frontend.start()

    visual = None
    if cameras and not args.no_visual:
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
            _shutdown(
                state,
                frontend,
                visual,
                hub,
                lidar,
                imu,
                cameras,
                cam_mux_stop,
                cam_mux_thread,
                monitor_stop,
                monitor_thread,
            )
    else:
        logger.info("Running headless; Ctrl-C to stop.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            _shutdown(
                state,
                frontend,
                visual,
                hub,
                lidar,
                imu,
                cameras,
                cam_mux_stop,
                cam_mux_thread,
                monitor_stop,
                monitor_thread,
            )


def _shutdown(
    state,
    frontend,
    visual,
    hub,
    lidar,
    imu,
    cameras,
    cam_mux_stop,
    cam_mux_thread,
    monitor_stop,
    monitor_thread,
) -> None:
    state.running = False
    logger.info("Stopping frontends / hub / sensors")
    monitor_stop.set()
    if monitor_thread is not None:
        monitor_thread.join(timeout=1.0)
    cam_mux_stop.set()
    if cam_mux_thread is not None:
        cam_mux_thread.join(timeout=1.0)
    if visual is not None:
        visual.stop()
    frontend.stop()
    hub.stop()
    lidar.stop()
    imu.stop()
    for cam in cameras:
        try:
            cam.stop()
        except Exception:  # noqa: BLE001
            pass
    logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
