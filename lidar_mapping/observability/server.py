"""
HTTP observability server for the fusion pipeline.

Endpoints (all GET unless noted):
  /healthz                  → 200 plain text
  /state                    → JSON full state summary
  /pose                     → JSON {position, quaternion, euler_deg, timestamp}
  /trajectory?max=N         → JSON list of poses
  /stats                    → JSON rates + counts
  /imu                      → JSON imu state
  /snapshot/camera          → PNG latest camera frame (overlay if available)
  /snapshot/top_down        → PNG XY projection + trajectory
  /snapshot/composite       → PNG side-by-side camera + top-down
  /control/<action>  (POST) → reset | save_map | pause | resume

Designed to be agent-friendly: small JSON payloads, deterministic PNGs.

Run standalone (separate Python process) or embed via ``serve_in_thread``.
"""

from __future__ import annotations

import argparse
import logging
import threading
from typing import Optional

import numpy as np
from flask import Flask, Response, jsonify, request

from lidar_mapping.observability.snapshot import (
    camera_png,
    composite_png,
    top_down_png,
)
from lidar_mapping.observability.state import FusionState, get_state

logger = logging.getLogger(__name__)


def _pose_to_dict(pose: np.ndarray) -> dict:
    R = pose[:3, :3]
    t = pose[:3, 3]
    # quaternion from rotation matrix (w, x, y, z)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    pitch = np.degrees(np.arcsin(np.clip(-R[2, 0], -1, 1)))
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    return {
        "position": t.tolist(),
        "quaternion_wxyz": [float(qw), float(qx), float(qy), float(qz)],
        "euler_deg": {"roll": float(roll), "pitch": float(pitch), "yaw": float(yaw)},
    }


def create_app(state: Optional[FusionState] = None) -> Flask:
    state = state or get_state()
    app = Flask(__name__)
    # Pillow / Flask log spam off
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    @app.get("/healthz")
    def healthz():
        return "ok", 200

    @app.get("/state")
    def full_state():
        with state.lock:
            pose = state.current_pose.copy()
            rates = state.rates
            return jsonify({
                "running": state.running,
                "uptime_s": float(__import__("time").time() - state.started_at),
                "pose": _pose_to_dict(pose),
                "counts": {
                    "map_points": int(state.map_points_count),
                    "keyframes": int(state.keyframe_count),
                    "lidar_frames": int(state.lidar_frames_total),
                    "imu_samples": int(state.imu_samples_total),
                    "camera_frames": int(state.camera_frames_total),
                    "trajectory_len": len(state.trajectory),
                },
                "rates_hz": {
                    "lidar": float(rates.lidar_hz),
                    "imu": float(rates.imu_hz),
                    "camera": float(rates.camera_hz),
                },
                "imu_euler_deg": {
                    "roll": state.imu_roll_deg,
                    "pitch": state.imu_pitch_deg,
                    "yaw": state.imu_yaw_deg,
                },
            })

    @app.get("/pose")
    def pose():
        with state.lock:
            return jsonify(_pose_to_dict(state.current_pose.copy()))

    @app.get("/trajectory")
    def trajectory():
        try:
            max_n = int(request.args.get("max", 500))
        except ValueError:
            max_n = 500
        traj = state.trajectory_snapshot(max_points=max_n)
        return jsonify([
            {"t": float(t), **_pose_to_dict(T)} for t, T in traj
        ])

    @app.get("/stats")
    def stats():
        with state.lock:
            return jsonify({
                "rates_hz": {
                    "lidar": float(state.rates.lidar_hz),
                    "imu": float(state.rates.imu_hz),
                    "camera": float(state.rates.camera_hz),
                },
                "counts": {
                    "map_points": int(state.map_points_count),
                    "keyframes": int(state.keyframe_count),
                    "lidar_frames": int(state.lidar_frames_total),
                    "imu_samples": int(state.imu_samples_total),
                    "camera_frames": int(state.camera_frames_total),
                },
            })

    @app.get("/imu")
    def imu():
        with state.lock:
            r = state.last_imu_reading
            base = {
                "roll_deg": state.imu_roll_deg,
                "pitch_deg": state.imu_pitch_deg,
                "yaw_deg": state.imu_yaw_deg,
            }
            if r is not None:
                base.update({
                    "accel_mss": np.asarray(r.accel_mss).tolist(),
                    "gyro_rads": np.asarray(r.gyro_rads).tolist(),
                    "quaternion_wxyz": np.asarray(r.quaternion).tolist(),
                    "timestamp": float(r.timestamp),
                })
            return jsonify(base)

    @app.get("/snapshot/camera")
    def snap_cam():
        return Response(camera_png(state), mimetype="image/png")

    @app.get("/snapshot/top_down")
    def snap_top():
        try:
            rng = float(request.args.get("range", 30))
        except ValueError:
            rng = 30.0
        return Response(top_down_png(state, range_m=rng), mimetype="image/png")

    @app.get("/snapshot/composite")
    def snap_composite():
        return Response(composite_png(state), mimetype="image/png")

    @app.post("/control/<action>")
    def control(action: str):
        action = action.lower()
        if action == "pause":
            with state.lock:
                state.running = False
            return jsonify({"ok": True, "running": False})
        if action == "resume":
            with state.lock:
                state.running = True
            return jsonify({"ok": True, "running": True})
        if action == "reset":
            with state.lock:
                state.current_pose = np.eye(4)
                state.trajectory.clear()
                state.map_points_count = 0
                state.keyframe_count = 0
            return jsonify({"ok": True})
        return jsonify({"ok": False, "error": f"unknown action: {action}"}), 400

    # ------------------------------------------------------------------
    # Calibration (live-tunable)
    # ------------------------------------------------------------------
    def _calib_to_dict(calib) -> dict:
        return {
            "image_width": int(calib.image_width),
            "image_height": int(calib.image_height),
            "K": np.asarray(calib.K).tolist(),
            "dist": np.asarray(calib.dist).tolist(),
            "T_imu_lidar": np.asarray(calib.T_imu_lidar).tolist(),
            "T_imu_cam": np.asarray(calib.T_imu_cam).tolist(),
            "vo_center_fraction": float(getattr(calib, "vo_center_fraction", 1.0)),
        }

    @app.get("/calibration")
    def get_calibration():
        with state.lock:
            if state.calibration is None:
                return jsonify({"error": "no calibration registered"}), 404
            return jsonify(_calib_to_dict(state.calibration))

    @app.post("/calibration")
    def post_calibration():
        """Partial update. Body JSON keys: fov_deg, fx, fy, cx, cy, K, dist,
        T_imu_cam, T_imu_lidar, cam_x, cam_y, cam_z (translation override),
        cam_roll_deg, cam_pitch_deg, cam_yaw_deg (override camera rotation
        relative to LiDAR-forward canonical mapping).
        """
        from lidar_mapping.fusion.calibration import (
            CalibrationConfig, _R_CAM_FROM_LIDAR_FORWARD, _default_K,
        )
        data = request.get_json(force=True, silent=True) or {}
        with state.lock:
            calib = state.calibration
            if calib is None:
                calib = CalibrationConfig.default()
                state.calibration = calib

            if "image_width" in data:
                calib.image_width = int(data["image_width"])
            if "image_height" in data:
                calib.image_height = int(data["image_height"])
            if "fov_deg" in data:
                calib.K = _default_K(calib.image_width, calib.image_height,
                                     float(data["fov_deg"]))
            if "K" in data:
                calib.K = np.asarray(data["K"], dtype=np.float64)
            for i, key in enumerate(("fx", "fy")):
                if key in data:
                    calib.K[i, i] = float(data[key])
            for i, key in enumerate(("cx", "cy")):
                if key in data:
                    calib.K[i, 2] = float(data[key])
            if "dist" in data:
                calib.dist = np.asarray(data["dist"], dtype=np.float64)
            if "vo_center_fraction" in data:
                calib.vo_center_fraction = float(data["vo_center_fraction"])
            if "T_imu_cam" in data:
                calib.T_imu_cam = np.asarray(data["T_imu_cam"], dtype=np.float64)
            if "T_imu_lidar" in data:
                calib.T_imu_lidar = np.asarray(data["T_imu_lidar"], dtype=np.float64)

            # Convenience: rebuild T_imu_cam from euler tweaks (deg) around the
            # canonical camera-from-lidar-forward axis mapping, with optional
            # translation expressed in LiDAR/IMU frame (meters).
            ang_keys = ("cam_roll_deg", "cam_pitch_deg", "cam_yaw_deg")
            tr_keys = ("cam_x", "cam_y", "cam_z")
            if any(k in data for k in (*ang_keys, *tr_keys)):
                rx = np.deg2rad(float(data.get("cam_roll_deg", 0.0)))
                ry = np.deg2rad(float(data.get("cam_pitch_deg", 0.0)))
                rz = np.deg2rad(float(data.get("cam_yaw_deg", 0.0)))
                cr, sr = np.cos(rx), np.sin(rx)
                cp, sp = np.cos(ry), np.sin(ry)
                cy, sy = np.cos(rz), np.sin(rz)
                Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
                Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
                Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
                # tweak applied in IMU frame after canonical mapping
                R_imu_cam = (Rz @ Ry @ Rx) @ _R_CAM_FROM_LIDAR_FORWARD.T
                T = np.eye(4)
                T[:3, :3] = R_imu_cam
                T[0, 3] = float(data.get("cam_x", 0.0))
                T[1, 3] = float(data.get("cam_y", 0.0))
                T[2, 3] = float(data.get("cam_z", 0.0))
                calib.T_imu_cam = T

            return jsonify(_calib_to_dict(calib))

    @app.post("/calibration/save")
    def save_calibration():
        import yaml  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415
        path = request.args.get("path") or (request.get_json(silent=True) or {}).get("path")
        if not path:
            return jsonify({"ok": False, "error": "missing ?path="}), 400
        with state.lock:
            if state.calibration is None:
                return jsonify({"ok": False, "error": "no calibration"}), 404
            d = _calib_to_dict(state.calibration)
        Path(path).write_text(yaml.safe_dump(d, sort_keys=False))
        return jsonify({"ok": True, "path": path})

    return app


def serve_in_thread(host: str = "0.0.0.0", port: int = 8765,
                    state: Optional[FusionState] = None) -> threading.Thread:
    """Launch the Flask app in a daemon background thread."""
    app = create_app(state)
    th = threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=False,
                               use_reloader=False, threaded=True),
        daemon=True,
        name="obs-http",
    )
    th.start()
    logger.info("Observability server listening on http://%s:%d", host, port)
    return th


def main() -> None:
    parser = argparse.ArgumentParser(description="Fusion observability server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = create_app()
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
