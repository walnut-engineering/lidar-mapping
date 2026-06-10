"""Replay recorded LiDAR + IMU data, optionally feeding the mapper."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from lidar_mapping.io import VLP16Playback, IMUPlayback

log = logging.getLogger("lidar-playback")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lidar-playback")
    ap.add_argument("--in", dest="inp", type=str, required=True,
                    help="Input recording directory")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--no-lidar", action="store_true")
    ap.add_argument("--no-imu", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    in_dir = Path(args.inp)
    lidar_path = in_dir / "lidar.vlp16"
    imu_path = in_dir / "imu.npz"

    lidar_pb = None
    imu_pb = None
    n_frames = 0
    n_imu = 0
    try:
        if not args.no_lidar and lidar_path.exists():
            lidar_pb = VLP16Playback(str(lidar_path), speed=args.speed, loop=args.loop)
            lidar_pb.start()
        if not args.no_imu and imu_path.exists():
            imu_pb = IMUPlayback(str(imu_path), speed=args.speed, loop=args.loop)
            imu_pb.start()

        last_log = time.monotonic()
        while True:
            did_work = False
            if lidar_pb is not None:
                frame = lidar_pb.get_frame(timeout=0.1)
                if frame is not None:
                    n_frames += 1
                    did_work = True
            if imu_pb is not None:
                while imu_pb.readings_available():
                    if imu_pb.get_reading(timeout=0) is None:
                        break
                    n_imu += 1
                    did_work = True
            now = time.monotonic()
            if now - last_log >= 1.0:
                log.info("Playback: %d frames, %d IMU samples", n_frames, n_imu)
                last_log = now
            if not did_work and not args.loop:
                # Both streams exhausted
                if ((lidar_pb is None or not lidar_pb._thread.is_alive()) and  # type: ignore[attr-defined]
                        (imu_pb is None or not imu_pb._thread.is_alive())):    # type: ignore[attr-defined]
                    break
    except KeyboardInterrupt:
        log.warning("Interrupted.")
    finally:
        if lidar_pb is not None:
            lidar_pb.stop()
        if imu_pb is not None:
            imu_pb.stop()
    log.info("Done: %d LiDAR frames, %d IMU samples", n_frames, n_imu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
