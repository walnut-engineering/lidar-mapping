"""Quick hardware bring-up diagnostics for camera, IMU, and VLP-16.

Runs standalone checks and prints a concise pass/fail summary:
  1. Camera frames from specified OpenCV indices
  2. Parsed IMU readings from WitMotion serial stream
  3. Raw UDP packet ingress on LiDAR data port

Example:
  python3 -m apps.diagnose_sensors --camera-indices 0,2 --imu-port /dev/ttyS1
"""

from __future__ import annotations

import argparse
import socket
import time

from lidar_mapping.sensors.camera import CameraCapture
from lidar_mapping.sensors.imu import WitMotionDriver


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sensor diagnostics")
    p.add_argument("--camera-indices", default="0,2",
                   help="Comma-separated camera indices")
    p.add_argument("--cam-width", type=int, default=1280)
    p.add_argument("--cam-height", type=int, default=720)
    p.add_argument("--imu-port", default="/dev/ttyS1")
    p.add_argument("--imu-baud", type=int, default=230400)
    p.add_argument("--lidar-host", default="0.0.0.0")
    p.add_argument("--lidar-port", type=int, default=2368)
    p.add_argument("--duration", type=float, default=4.0)
    return p.parse_args()


def check_cameras(indices: list[int], width: int, height: int, duration: float) -> dict:
    cams: list[CameraCapture] = []
    per_cam_counts: dict[int, int] = {i: 0 for i in indices}
    per_cam_shapes: dict[int, tuple[int, int, int] | None] = {i: None for i in indices}
    try:
        for idx in indices:
            cam = CameraCapture(device_index=idx, width=width, height=height, fps=30, max_queue=6)
            cam.start()
            cams.append(cam)
        t0 = time.monotonic()
        while time.monotonic() - t0 < duration:
            for idx, cam in zip(indices, cams):
                f = cam.get_latest_frame()
                if f is not None:
                    per_cam_counts[idx] += 1
                    per_cam_shapes[idx] = f.image.shape
            time.sleep(0.01)
    finally:
        for cam in cams:
            try:
                cam.stop()
            except Exception:  # noqa: BLE001
                pass

    return {
        "counts": per_cam_counts,
        "shapes": per_cam_shapes,
        "ok": all(c > 0 for c in per_cam_counts.values()),
    }


def check_imu(port: str, baud: int, duration: float) -> dict:
    imu = WitMotionDriver(port=port, baudrate=baud)
    parsed = 0
    last_rpy = None
    try:
        imu.start()
        t0 = time.monotonic()
        while time.monotonic() - t0 < duration:
            r = imu.get_latest_reading()
            if r is not None:
                parsed += 1
                last_rpy = (float(r.roll_deg), float(r.pitch_deg), float(r.yaw_deg))
            time.sleep(0.005)
    finally:
        imu.stop()

    return {
        "parsed_readings": parsed,
        "last_rpy": last_rpy,
        "ok": parsed > 0,
    }


def check_lidar_udp(host: str, port: int, duration: float) -> dict:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    sock.settimeout(0.2)
    packets = 0
    bytes_total = 0
    start = time.monotonic()
    try:
        while time.monotonic() - start < duration:
            try:
                data, _ = sock.recvfrom(2048)
                packets += 1
                bytes_total += len(data)
            except socket.timeout:
                pass
    finally:
        sock.close()

    return {
        "packets": packets,
        "bytes": bytes_total,
        "ok": packets > 0,
    }


def main() -> int:
    args = parse_args()
    cam_indices = [int(x.strip()) for x in args.camera_indices.split(",") if x.strip()]

    cam = check_cameras(cam_indices, args.cam_width, args.cam_height, args.duration)
    imu = check_imu(args.imu_port, args.imu_baud, args.duration)
    lidar = check_lidar_udp(args.lidar_host, args.lidar_port, args.duration)

    print("=== Sensor Diagnostics ===")
    print("camera:")
    for idx in cam_indices:
        print(f"  /dev/video{idx}: frames={cam['counts'][idx]} shape={cam['shapes'][idx]}")
    print(f"  camera_ok={cam['ok']}")

    print("imu:")
    print(f"  parsed_readings={imu['parsed_readings']} last_rpy={imu['last_rpy']}")
    print(f"  imu_ok={imu['ok']}")

    print("lidar:")
    print(f"  udp_packets={lidar['packets']} bytes={lidar['bytes']}")
    print(f"  lidar_ok={lidar['ok']}")

    all_ok = cam["ok"] and imu["ok"] and lidar["ok"]
    print(f"overall_ok={all_ok}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
