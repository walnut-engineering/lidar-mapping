"""Camera-assisted calibration for HBX mount control."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from lidar_mapping.cli.hbx_inject import DIRECTION_VALUES, run_sequence
from lidar_mapping.sensors.camera import CameraCapture

_CV2 = None


def _get_cv2():
    global _CV2
    if _CV2 is None:
        try:
            import cv2 as _cv2  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "opencv-python is required. Install with: pip install opencv-python"
            ) from exc
        _CV2 = _cv2
    return _CV2


@dataclass
class TrialResult:
    axis: str
    direction: str
    move_seconds: float
    dx_px: float
    dy_px: float
    mag_px: float
    axis_px: float
    axis_deg: Optional[float]
    deg_per_sec: Optional[float]


def _estimate_shift(before_bgr: np.ndarray, after_bgr: np.ndarray) -> tuple[float, float, float]:
    cv2 = _get_cv2()
    before = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2GRAY)
    after = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2GRAY)

    pts0 = cv2.goodFeaturesToTrack(
        before,
        maxCorners=300,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
    )
    if pts0 is None or len(pts0) < 12:
        raise RuntimeError("Not enough visual features for motion estimation")

    pts1, status, _ = cv2.calcOpticalFlowPyrLK(before, after, pts0, None)
    if pts1 is None or status is None:
        raise RuntimeError("Optical flow failed")

    good = status.reshape(-1) == 1
    if int(np.count_nonzero(good)) < 8:
        raise RuntimeError("Too few tracked features")

    dxy = pts1[good] - pts0[good]
    dx = float(np.median(dxy[:, 0]))
    dy = float(np.median(dxy[:, 1]))
    mag = float(np.hypot(dx, dy))
    return dx, dy, mag


def _get_latest_image(cam: CameraCapture, timeout_s: float = 1.5) -> np.ndarray:
    deadline = time.monotonic() + timeout_s
    frame = None
    while time.monotonic() < deadline:
        frame = cam.get_latest_frame()
        if frame is not None:
            return frame.image
        time.sleep(0.01)
    raise RuntimeError("No camera frame available")


def _run_trial(
    *,
    cam: CameraCapture,
    port: str,
    baud: int,
    axis: str,
    direction: str,
    move_seconds: float,
    tx_hz: float,
    settle_seconds: float,
    fov_deg: Optional[float],
) -> TrialResult:
    before = _get_latest_image(cam)

    run_sequence(
        port=port,
        baud=baud,
        axis=axis,
        hold_seconds=0.05,
        pulse_count=1,
        gap_seconds=0.2,
        active_mode=0x04,
        active_value=DIRECTION_VALUES[direction],
        release_mode=0x02,
        release_value=0x0000,
        move_seconds=move_seconds,
        tx_hz=tx_hz,
        dry_run=False,
    )

    time.sleep(max(settle_seconds, 0.0))
    after = _get_latest_image(cam)

    dx, dy, mag = _estimate_shift(before, after)

    # Axis-specific projection: azimuth tends to dominate horizontal image shift,
    # tilt tends to dominate vertical shift.
    axis_px = abs(dx) if axis == "azimuth" else abs(dy)
    if axis_px < 0.01:
        axis_px = mag

    axis_deg = None
    deg_per_sec = None
    if fov_deg is not None and fov_deg > 0:
        w = float(before.shape[1])
        h = float(before.shape[0])
        px_span = w if axis == "azimuth" else h
        axis_deg = axis_px * (fov_deg / px_span)
        if move_seconds > 0:
            deg_per_sec = axis_deg / move_seconds

    return TrialResult(
        axis=axis,
        direction=direction,
        move_seconds=move_seconds,
        dx_px=dx,
        dy_px=dy,
        mag_px=mag,
        axis_px=axis_px,
        axis_deg=axis_deg,
        deg_per_sec=deg_per_sec,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lidar-hbx-calibrate")
    p.add_argument("--port", default="COM36", help="Serial port")
    p.add_argument("--baud", type=int, default=19200, help="Serial baud")
    p.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index")
    p.add_argument("--width", type=int, default=1280, help="Camera width")
    p.add_argument("--height", type=int, default=720, help="Camera height")
    p.add_argument("--fps", type=int, default=30, help="Camera fps")
    p.add_argument(
        "--axis",
        choices=("azimuth", "tilt", "both"),
        default="both",
        help="Axis to calibrate",
    )
    p.add_argument("--move-seconds", type=float, default=4.0, help="Seconds per trial")
    p.add_argument("--tx-hz", type=float, default=25.0, help="Active command stream rate")
    p.add_argument("--settle", type=float, default=0.2, help="Settle time after each move")
    p.add_argument("--trials", type=int, default=3, help="Trials per axis and direction")
    p.add_argument(
        "--fov-deg",
        type=float,
        default=None,
        help="Camera field-of-view in degrees (horizontal for azimuth, vertical for tilt)",
    )
    p.add_argument(
        "--out",
        default="mount_motion_calibration.json",
        help="Output JSON file",
    )
    return p


def _summary(results: list[TrialResult]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for r in results:
        if r.deg_per_sec is None:
            continue
        grouped.setdefault(r.axis, {}).setdefault(r.direction, []).append(r.deg_per_sec)

    out: dict[str, dict[str, float]] = {}
    for axis, dirs in grouped.items():
        out[axis] = {}
        for direction, vals in dirs.items():
            out[axis][direction] = float(statistics.median(vals))
            out[axis][f"{direction}_seconds_per_degree"] = (
                float(1.0 / out[axis][direction]) if out[axis][direction] > 0 else 0.0
            )
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    axes = ("azimuth", "tilt") if args.axis == "both" else (args.axis,)
    directions = ("positive", "negative")

    cam = CameraCapture(
        device_index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )

    results: list[TrialResult] = []
    cam.start()
    try:
        # Warm up a few frames.
        time.sleep(0.5)
        for axis in axes:
            for direction in directions:
                for i in range(args.trials):
                    trial = _run_trial(
                        cam=cam,
                        port=args.port,
                        baud=args.baud,
                        axis=axis,
                        direction=direction,
                        move_seconds=args.move_seconds,
                        tx_hz=args.tx_hz,
                        settle_seconds=args.settle,
                        fov_deg=args.fov_deg,
                    )
                    results.append(trial)
                    print(
                        f"trial {axis}/{direction} {i + 1}/{args.trials}: "
                        f"dx={trial.dx_px:.2f}px dy={trial.dy_px:.2f}px "
                        f"axis={trial.axis_px:.2f}px "
                        f"deg_per_sec={trial.deg_per_sec if trial.deg_per_sec is not None else 'n/a'}"
                    )
    finally:
        cam.stop()

    payload = {
        "timestamp": time.time(),
        "config": {
            "port": args.port,
            "baud": args.baud,
            "camera_index": args.camera_index,
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "move_seconds": args.move_seconds,
            "tx_hz": args.tx_hz,
            "settle": args.settle,
            "trials": args.trials,
            "fov_deg": args.fov_deg,
        },
        "trials": [asdict(r) for r in results],
        "summary": _summary(results),
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote calibration: {out_path}")

    if payload["summary"]:
        print(f"summary: {payload['summary']}")
    else:
        print("summary: no angular conversion (set --fov-deg to compute deg/sec)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
