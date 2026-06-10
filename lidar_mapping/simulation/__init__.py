"""
Synthetic sensor data generation for hardware-free development & testing.

Provides:
    - generate_vlp16_packet() : build a single valid 1206-byte VLP-16 UDP packet
    - VLP16Simulator          : full driver-compatible LiDAR simulator (start/stop/get_frame)
    - generate_imu_reading()  : build a single IMUReading from a motion profile
    - IMUSimulator            : threaded IMU producer matching the driver API
    - SyntheticTrajectory     : helper that produces consistent IMU + LiDAR sensing
                                of a virtual room while a virtual sensor moves
                                along a parametric path.
"""

from __future__ import annotations

import math
import queue
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from lidar_mapping.sensors.imu import IMUReading
from lidar_mapping.sensors.vlp16 import (
    VLP16Frame,
    VLP16PacketParser,
    _VERTICAL_COS,
    _VERTICAL_SIN,
)
from lidar_mapping.utils.transforms import (
    apply_transform,
    make_transform,
    make_transform_from_euler,
)

_PACKET_SIZE = 1206
_BLOCKS = 12
_LASERS = 16
_BLOCK_FLAG = 0xFFEE
_DIST_RES = 0.002  # 2 mm per LSB


# ---------------------------------------------------------------------------
# VLP-16 packet builder
# ---------------------------------------------------------------------------

def generate_vlp16_packet(
    azimuths_deg: np.ndarray,
    distances_m: np.ndarray,
    intensities: Optional[np.ndarray] = None,
    timestamp_us: int = 0,
) -> bytes:
    """
    Build a single valid VLP-16 UDP packet.

    Parameters
    ----------
    azimuths_deg:
        (12,) array of azimuth (degrees) for the first firing of each of the
        12 blocks.  The second firing's azimuth is interpolated by the parser.
    distances_m:
        (12, 32) array of distances (metres) — 2 firings of 16 lasers per block.
        Use 0.0 to encode "no return" (point below 0.1 m is dropped by parser).
    intensities:
        Optional (12, 32) uint8 array; defaults to all 100.
    timestamp_us:
        Packet timestamp (µs past the hour) — embedded at bytes 1200-1203.

    Returns
    -------
    bytes of length 1206.
    """
    azimuths_deg = np.asarray(azimuths_deg, dtype=np.float64)
    distances_m = np.asarray(distances_m, dtype=np.float64)
    if azimuths_deg.shape != (_BLOCKS,):
        raise ValueError(f"azimuths_deg must be shape ({_BLOCKS},)")
    if distances_m.shape != (_BLOCKS, 2 * _LASERS):
        raise ValueError(f"distances_m must be shape ({_BLOCKS}, {2 * _LASERS})")

    if intensities is None:
        intensities = np.full((_BLOCKS, 2 * _LASERS), 100, dtype=np.uint8)
    else:
        intensities = np.asarray(intensities, dtype=np.uint8)
        if intensities.shape != (_BLOCKS, 2 * _LASERS):
            raise ValueError(
                f"intensities must be shape ({_BLOCKS}, {2 * _LASERS})"
            )

    buf = bytearray(_PACKET_SIZE)
    offset = 0
    for b in range(_BLOCKS):
        az_raw = int(round((azimuths_deg[b] % 360.0) * 100.0)) & 0xFFFF
        struct.pack_into("<HH", buf, offset, _BLOCK_FLAG, az_raw)
        offset += 4
        for ch in range(2 * _LASERS):
            dist_raw = int(round(distances_m[b, ch] / _DIST_RES)) & 0xFFFF
            struct.pack_into("<HB", buf, offset, dist_raw, int(intensities[b, ch]))
            offset += 3
    # Timestamp
    struct.pack_into("<I", buf, 1200, int(timestamp_us) & 0xFFFFFFFF)
    # Factory bytes (return mode + product id) — strongest, VLP-16
    buf[1204] = 0x37
    buf[1205] = 0x22
    return bytes(buf)


def build_packets_from_ranges(
    ranges_by_channel: np.ndarray,
    azimuth_start_deg: float = 0.0,
    azimuth_step_deg: float = 0.2,
    timestamp_us: int = 0,
) -> List[bytes]:
    """
    Build a list of VLP-16 packets covering a full 360° sweep.

    Parameters
    ----------
    ranges_by_channel:
        (n_azimuths, 16) array of distances per channel.
    azimuth_start_deg:
        Starting azimuth.
    azimuth_step_deg:
        Angular step between firings (default 0.2° = ~600 RPM equivalent).
    timestamp_us:
        Initial timestamp.

    Returns
    -------
    list of 1206-byte packets.
    """
    ranges_by_channel = np.asarray(ranges_by_channel, dtype=np.float64)
    n_az, n_ch = ranges_by_channel.shape
    if n_ch != _LASERS:
        raise ValueError(f"ranges_by_channel must have {_LASERS} columns")
    # Each packet carries 24 firings (12 blocks × 2 sequences)
    firings_per_packet = 2 * _BLOCKS
    n_packets = (n_az + firings_per_packet - 1) // firings_per_packet
    packets: List[bytes] = []
    for p in range(n_packets):
        az = np.zeros(_BLOCKS, dtype=np.float64)
        dist = np.zeros((_BLOCKS, 2 * _LASERS), dtype=np.float64)
        for b in range(_BLOCKS):
            for s in range(2):
                fire_idx = p * firings_per_packet + b * 2 + s
                if fire_idx >= n_az:
                    continue
                a = azimuth_start_deg + fire_idx * azimuth_step_deg
                if s == 0:
                    az[b] = a
                dist[b, s * _LASERS : (s + 1) * _LASERS] = ranges_by_channel[fire_idx]
        packets.append(
            generate_vlp16_packet(
                az, dist, timestamp_us=timestamp_us + p * 1327
            )
        )
    return packets


# ---------------------------------------------------------------------------
# Scene → ranges
# ---------------------------------------------------------------------------

def cast_rays_into_box(
    half_extent_x: float = 5.0,
    half_extent_y: float = 5.0,
    z_floor: float = 0.0,
    z_ceiling: float = 4.0,
    sensor_xyz: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    azimuths_deg: Optional[np.ndarray] = None,
    max_range_m: float = 80.0,
) -> np.ndarray:
    """
    Ray-cast from the sensor into an axis-aligned room (box).

    Returns (n_azimuths, 16) ranges in metres (0.0 if no hit within range).
    Used by VLP16Simulator to generate realistic packets of a virtual room.
    """
    if azimuths_deg is None:
        azimuths_deg = np.arange(0.0, 360.0, 0.2)
    sx, sy, sz = sensor_xyz
    n_az = len(azimuths_deg)
    ranges = np.zeros((n_az, _LASERS), dtype=np.float64)

    for i, az in enumerate(azimuths_deg):
        az_rad = math.radians(az)
        cos_az, sin_az = math.cos(az_rad), math.sin(az_rad)
        for ch in range(_LASERS):
            vc, vs = _VERTICAL_COS[ch], _VERTICAL_SIN[ch]
            # Direction in sensor frame == world frame here (no rotation)
            dx = vc * sin_az
            dy = vc * cos_az
            dz = vs
            # Slab intersection with the axis-aligned box
            t_candidates: List[float] = []
            for plane_pos, axis_dir, axis_origin in (
                (half_extent_x, dx, sx),
                (-half_extent_x, dx, sx),
                (half_extent_y, dy, sy),
                (-half_extent_y, dy, sy),
                (z_ceiling, dz, sz),
                (z_floor, dz, sz),
            ):
                if abs(axis_dir) < 1e-9:
                    continue
                t = (plane_pos - axis_origin) / axis_dir
                if t > 0.0:
                    t_candidates.append(t)
            if not t_candidates:
                continue
            t_min = min(t_candidates)
            if t_min < max_range_m:
                ranges[i, ch] = t_min
    return ranges


# ---------------------------------------------------------------------------
# VLP-16 Simulator (driver-compatible)
# ---------------------------------------------------------------------------

class VLP16Simulator:
    """
    Drop-in replacement for :class:`VLP16Driver` that synthesises packets
    from a virtual scene.  Implements start / stop / get_frame and the
    ``_packet_callback`` hook used by :class:`VLP16Recorder`.

    Parameters
    ----------
    scene_ranges:
        Optional (n_azimuths, 16) array of ranges per azimuth.  If ``None``
        a default 10×10×4 room is generated.
    rpm:
        Simulated rotation speed (default 600 RPM = 10 Hz).
    azimuth_step_deg:
        Angular resolution (default 0.2°).
    """

    def __init__(
        self,
        scene_ranges: Optional[np.ndarray] = None,
        rpm: float = 600.0,
        azimuth_step_deg: float = 0.2,
    ) -> None:
        if scene_ranges is None:
            scene_ranges = cast_rays_into_box()
        self._ranges = np.asarray(scene_ranges, dtype=np.float64)
        self._rpm = rpm
        self._az_step = azimuth_step_deg
        self._parser = VLP16PacketParser()
        # Match VLP16Driver: callback receives only the raw bytes
        self._packet_callback: Optional[Callable[[bytes], None]] = None
        self._frame_queue: "queue.Queue[VLP16Frame]" = queue.Queue(maxsize=10)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # -- driver-compatible API --------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None

    def get_frame(self, timeout: float = 1.0) -> Optional[VLP16Frame]:
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # -- internal ---------------------------------------------------------
    def _run(self) -> None:
        period = 60.0 / self._rpm  # seconds per revolution
        firings_per_packet = 2 * _BLOCKS
        packet_period = period * (firings_per_packet * self._az_step) / 360.0
        ts_us = 0
        while not self._stop.is_set():
            t0 = time.monotonic()
            packets = build_packets_from_ranges(
                self._ranges,
                azimuth_step_deg=self._az_step,
                timestamp_us=ts_us,
            )
            points = []
            for raw in packets:
                if self._stop.is_set():
                    return
                if self._packet_callback is not None:
                    self._packet_callback(raw)
                pts, _ = self._parser.parse(raw)
                points.extend(pts)
                # Sleep to simulate packet arrival cadence
                time.sleep(packet_period)
            frame = VLP16Frame(points=points, timestamp_us=ts_us)
            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                pass
            ts_us += int((time.monotonic() - t0) * 1e6)


# ---------------------------------------------------------------------------
# IMU Simulator
# ---------------------------------------------------------------------------

def generate_imu_reading(
    t: float,
    gyro_rads: np.ndarray = np.zeros(3),
    accel_mss: np.ndarray = np.array([0.0, 0.0, 9.80665]),
    mag_ut: Optional[np.ndarray] = None,
    quaternion: Optional[np.ndarray] = None,
) -> IMUReading:
    """Convenience builder for a single synthetic IMUReading."""
    if quaternion is None:
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    return IMUReading(
        timestamp=float(t),
        accel_mss=np.asarray(accel_mss, dtype=np.float64),
        gyro_rads=np.asarray(gyro_rads, dtype=np.float64),
        mag_ut=mag_ut,
        quaternion=np.asarray(quaternion, dtype=np.float64),
    )


@dataclass
class MotionProfile:
    """
    Parametric motion profile for synthetic IMU generation.

    Attributes
    ----------
    gyro_fn:
        Callable t → (3,) angular velocity in rad/s
    accel_fn:
        Callable t → (3,) linear acceleration in m/s² (gravity not added)
    add_gravity:
        If True, append [0,0,9.80665] to accel_fn output.
    noise_gyro_std:
        Std-dev (rad/s) of Gaussian noise injected on gyro.
    noise_accel_std:
        Std-dev (m/s²) of Gaussian noise injected on accel.
    """

    gyro_fn: Callable[[float], np.ndarray] = field(
        default_factory=lambda: (lambda t: np.zeros(3))
    )
    accel_fn: Callable[[float], np.ndarray] = field(
        default_factory=lambda: (lambda t: np.zeros(3))
    )
    add_gravity: bool = True
    noise_gyro_std: float = 0.0
    noise_accel_std: float = 0.0
    seed: int = 0


class IMUSimulator:
    """
    Threaded IMU producer with the same `get_reading`/`readings_available`
    interface as the real drivers.

    Parameters
    ----------
    profile:
        :class:`MotionProfile` describing motion.
    rate_hz:
        Output sample rate.
    """

    def __init__(
        self,
        profile: Optional[MotionProfile] = None,
        rate_hz: float = 100.0,
    ) -> None:
        self._profile = profile or MotionProfile()
        self._rate = rate_hz
        self._queue: "queue.Queue[IMUReading]" = queue.Queue(maxsize=2000)
        self._latest: Optional[IMUReading] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._rng = np.random.default_rng(self._profile.seed)

    # -- driver-compatible API --------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None

    def get_reading(self, timeout: float = 1.0) -> Optional[IMUReading]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_latest_reading(self) -> Optional[IMUReading]:
        return self._latest

    def readings_available(self) -> bool:
        return not self._queue.empty()

    # -- internal ---------------------------------------------------------
    def _run(self) -> None:
        dt = 1.0 / self._rate
        t0 = time.monotonic()
        next_t = t0
        while not self._stop.is_set():
            now = time.monotonic()
            if now < next_t:
                time.sleep(max(0.0, next_t - now))
            t_rel = next_t - t0
            gyro = np.asarray(self._profile.gyro_fn(t_rel), dtype=np.float64)
            accel = np.asarray(self._profile.accel_fn(t_rel), dtype=np.float64)
            if self._profile.add_gravity:
                accel = accel + np.array([0.0, 0.0, 9.80665])
            if self._profile.noise_gyro_std > 0:
                gyro = gyro + self._rng.normal(0, self._profile.noise_gyro_std, 3)
            if self._profile.noise_accel_std > 0:
                accel = accel + self._rng.normal(0, self._profile.noise_accel_std, 3)
            r = IMUReading(
                timestamp=next_t,
                accel_mss=accel,
                gyro_rads=gyro,
            )
            self._latest = r
            try:
                self._queue.put_nowait(r)
            except queue.Full:
                # Drop oldest
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(r)
                except queue.Empty:
                    pass
            next_t += dt
