"""Record live (or simulated) sensor data to disk."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from lidar_mapping.config import KitConfig, load_config, kit_config
from lidar_mapping.io import VLP16Recorder, IMURecorder

log = logging.getLogger("lidar-record")


def _build_lidar(cfg: KitConfig, simulate: bool):
    if simulate:
        from lidar_mapping.simulation import VLP16Simulator
        return VLP16Simulator()
    from lidar_mapping.sensors.vlp16 import VLP16Driver
    return VLP16Driver(
        host=cfg.lidar.host,
        data_port=cfg.lidar.data_port,
        position_port=cfg.lidar.position_port,
    )


def _build_imu(cfg: KitConfig, simulate: bool):
    if simulate:
        from lidar_mapping.simulation import IMUSimulator
        return IMUSimulator(rate_hz=cfg.imu.rate_hz)
    name = cfg.imu.driver.lower()
    if name == "witmotion":
        from lidar_mapping.sensors.imu import WitMotionDriver
        return WitMotionDriver(port=cfg.imu.port, baud=cfg.imu.baud)
    if name == "mpu9250":
        from lidar_mapping.sensors.imu import MPU9250Driver
        return MPU9250Driver(i2c_bus=cfg.imu.i2c_bus)
    if name == "bno055":
        from lidar_mapping.sensors.imu import BNO055Driver
        return BNO055Driver(i2c_bus=cfg.imu.i2c_bus)
    if name == "lsm9ds1":
        from lidar_mapping.sensors.imu import LSM9DS1Driver
        return LSM9DS1Driver(i2c_bus=cfg.imu.i2c_bus)
    if name == "serial_ahrs":
        from lidar_mapping.sensors.imu import SerialAHRSDriver
        return SerialAHRSDriver(port=cfg.imu.port, baud=cfg.imu.baud)
    raise ValueError(f"Unknown IMU driver: {name}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lidar-record")
    ap.add_argument("--config", type=str, default=None,
                    help="Path to YAML/TOML/JSON kit config")
    ap.add_argument("--out", type=str, required=True,
                    help="Output directory")
    ap.add_argument("--duration", type=float, default=30.0,
                    help="Recording duration in seconds")
    ap.add_argument("--simulate", action="store_true",
                    help="Use synthetic sensors instead of real hardware")
    ap.add_argument("--no-lidar", action="store_true")
    ap.add_argument("--no-imu", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = load_config(args.config) if args.config else kit_config()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    lidar = None
    imu = None
    rec_lidar = None
    rec_imu = None
    try:
        if not args.no_lidar:
            lidar = _build_lidar(cfg, args.simulate)
            rec_lidar = VLP16Recorder(lidar, out_dir / "lidar.vlp16")
            rec_lidar.__enter__()
            lidar.start()
            log.info("LiDAR recording started.")
        if not args.no_imu:
            imu = _build_imu(cfg, args.simulate)
            rec_imu = IMURecorder(imu, out_dir / "imu.npz",
                                  poll_rate_hz=cfg.imu.rate_hz)
            rec_imu.__enter__()
            imu.start()
            log.info("IMU recording started.")

        log.info("Recording for %.1f s ...", args.duration)
        t_end = time.monotonic() + args.duration
        while time.monotonic() < t_end:
            time.sleep(0.2)
    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
    finally:
        if rec_lidar is not None:
            rec_lidar.__exit__(None, None, None)
        if rec_imu is not None:
            rec_imu.__exit__(None, None, None)
        if lidar is not None:
            lidar.stop()
        if imu is not None:
            imu.stop()
    log.info("Recording saved to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
