"""Build a map from a recorded session (LiDAR + optional IMU)."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from lidar_mapping.config import KitConfig, load_config, kit_config
from lidar_mapping.io import VLP16Playback, IMUPlayback
from lidar_mapping.mapping.mapper import Mapper
from lidar_mapping.mapping.imu_preintegrator import IMUPreintegrator

log = logging.getLogger("lidar-map")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lidar-map")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--in", dest="inp", type=str, required=True,
                    help="Input recording directory")
    ap.add_argument("--save", type=str, default=None,
                    help="Save resulting map to this file (.pcd/.ply)")
    ap.add_argument("--voxel", type=float, default=None,
                    help="Override voxel size (m)")
    ap.add_argument("--no-imu", action="store_true",
                    help="Disable IMU motion prior")
    ap.add_argument("--speed", type=float, default=10.0,
                    help="Playback speed (default: 10x for faster mapping)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Stop after N LiDAR frames (0 = no limit)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = load_config(args.config) if args.config else kit_config()
    if args.voxel is not None:
        cfg.mapper.voxel_size = args.voxel

    in_dir = Path(args.inp)
    lidar_path = in_dir / "lidar.vlp16"
    imu_path = in_dir / "imu.npz"
    if not lidar_path.exists():
        log.error("LiDAR recording not found: %s", lidar_path)
        return 1

    preint = None
    imu_pb = None
    if not args.no_imu and imu_path.exists():
        preint = IMUPreintegrator()
        imu_pb = IMUPlayback(str(imu_path), speed=args.speed)
        imu_pb.start()
        log.info("IMU motion prior enabled")

    mapper = Mapper(
        voxel_size=cfg.mapper.voxel_size,
        min_range=cfg.mapper.min_range,
        max_range=cfg.mapper.max_range,
        z_min=cfg.mapper.z_min,
        z_max=cfg.mapper.z_max,
        remove_ground=cfg.mapper.remove_ground,
        icp_max_correspondence_distance=cfg.mapper.icp_max_correspondence_distance,
        max_map_points=cfg.mapper.max_map_points,
        imu_preintegrator=preint,
    )

    lidar_pb = VLP16Playback(str(lidar_path), speed=args.speed)
    lidar_pb.start()

    n = 0
    t0 = time.monotonic()
    try:
        while True:
            frame = lidar_pb.get_frame(timeout=1.0)
            if frame is None:
                if not lidar_pb._thread.is_alive():  # type: ignore[attr-defined]
                    break
                continue
            # Drain IMU samples accumulated since last frame
            if preint is not None and imu_pb is not None:
                while imu_pb.readings_available():
                    r = imu_pb.get_reading(timeout=0)
                    if r is None:
                        break
                    preint.push(r)
            arr = frame.to_numpy()
            if len(arr) == 0:
                continue
            result = mapper.add_scan(arr)
            n += 1
            if n % 10 == 0:
                log.info("Processed %d frames (last fitness %.3f)", n, result.fitness)
            if args.max_frames and n >= args.max_frames:
                break
    except KeyboardInterrupt:
        log.warning("Interrupted.")
    finally:
        lidar_pb.stop()
        if imu_pb is not None:
            imu_pb.stop()

    dt = time.monotonic() - t0
    map_pts = mapper.map_points
    n_points = 0 if map_pts is None else len(map_pts)
    log.info("Mapping complete: %d scans, %d points, %.1fs", n, n_points, dt)
    if args.save and map_pts is not None:
        out = Path(args.save)
        mapper.save_map(out)
        log.info("Map saved to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
