"""
VLP-16 LiDAR sensor driver.

Parses Velodyne VLP-16 UDP data packets (factory default: host port 2368)
and converts raw firing data into Cartesian point clouds.

Packet structure reference:
  Velodyne VLP-16 User Manual, Rev F – Appendix B, Data Packet Format.

Usage example::

    from lidar_mapping.sensors.vlp16 import VLP16Driver

    driver = VLP16Driver(host="0.0.0.0", port=2368)
    driver.start()
    try:
        while True:
            frame = driver.get_frame(timeout=2.0)
            if frame is not None:
                points = frame.to_numpy()   # (N, 4) array: x, y, z, intensity
    finally:
        driver.stop()
"""

from __future__ import annotations

import math
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# VLP-16 constants
# ---------------------------------------------------------------------------

_DATA_PORT: int = 2368           # default UDP data port
_POSITION_PORT: int = 8308       # GPS/position port (not used here)
_PACKET_SIZE: int = 1206         # bytes per data packet
_BLOCKS_PER_PACKET: int = 12
_CHANNELS_PER_BLOCK: int = 32    # 2 firing sequences × 16 lasers
_LASERS: int = 16

# Vertical angles (degrees) for each of the 16 laser channels, channel 0..15
_VERTICAL_ANGLES_DEG: Tuple[float, ...] = (
    -15.0, 1.0, -13.0, 3.0, -11.0, 5.0, -9.0, 7.0,
    -7.0, 9.0, -5.0, 11.0, -3.0, 13.0, -1.0, 15.0,
)
_VERTICAL_COS = tuple(math.cos(math.radians(a)) for a in _VERTICAL_ANGLES_DEG)
_VERTICAL_SIN = tuple(math.sin(math.radians(a)) for a in _VERTICAL_ANGLES_DEG)

# Timing offsets (µs) for each firing group within a block
_FIRING_CYCLE_US: float = 55.296  # duration of one full block (both sequences)
_SINGLE_FIRING_US: float = 2.304  # single laser firing duration

# Distance resolution: raw units → metres
_DISTANCE_RESOLUTION: float = 0.002  # 2 mm per LSB

# Block header magic word
_BLOCK_FLAG: int = 0xEEFF


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VLP16Point:
    """A single resolved 3-D point from a VLP-16 firing."""

    x: float
    y: float
    z: float
    intensity: float
    channel: int          # laser channel 0..15
    azimuth_deg: float    # horizontal angle at time of firing
    timestamp_us: float   # packet timestamp + firing offset (µs)
    distance_m: float     # measured range in metres


@dataclass
class VLP16Frame:
    """
    One complete 360° sweep from the VLP-16.

    A frame is accumulated until the azimuth wraps past 0° again.
    """

    points: List[VLP16Point] = field(default_factory=list)
    timestamp_us: float = 0.0       # timestamp of first packet in this frame

    def to_numpy(self) -> np.ndarray:
        """
        Return a (N, 4) float32 array with columns [x, y, z, intensity].

        Useful for passing directly to Open3D or numpy-based pipelines.
        """
        if not self.points:
            return np.empty((0, 4), dtype=np.float32)
        arr = np.array(
            [(p.x, p.y, p.z, p.intensity) for p in self.points],
            dtype=np.float32,
        )
        return arr

    def __len__(self) -> int:
        return len(self.points)


# ---------------------------------------------------------------------------
# Packet parser
# ---------------------------------------------------------------------------

class VLP16PacketParser:
    """
    Low-level VLP-16 UDP packet parser.

    Takes a raw 1206-byte UDP payload and returns a list of :class:`VLP16Point`
    objects together with the packet's GPS timestamp (µs).
    """

    def parse(self, raw: bytes) -> Tuple[List[VLP16Point], float]:
        """
        Parse one VLP-16 data packet.

        Parameters
        ----------
        raw:
            Raw UDP payload (must be exactly 1206 bytes).

        Returns
        -------
        points:
            All resolved 3-D points from this packet.
        timestamp_us:
            GPS timestamp embedded in the packet (µs past the hour).

        Raises
        ------
        ValueError:
            If the payload length is incorrect.
        """
        if len(raw) != _PACKET_SIZE:
            raise ValueError(
                f"VLP-16 packet must be {_PACKET_SIZE} bytes; got {len(raw)}"
            )

        # Bytes 1200..1203: GPS timestamp (µs, little-endian uint32)
        timestamp_us: float = struct.unpack_from("<I", raw, 1200)[0]

        points: List[VLP16Point] = []
        offset = 0

        prev_azimuth: Optional[float] = None

        for block_idx in range(_BLOCKS_PER_PACKET):
            flag, azimuth_raw = struct.unpack_from("<HH", raw, offset)
            offset += 4

            if flag != _BLOCK_FLAG:
                # Skip malformed block
                offset += 96  # 32 channels × 3 bytes
                continue

            azimuth_deg = azimuth_raw / 100.0  # hundredths of a degree

            # Interpolate azimuth for second firing sequence within the block
            if prev_azimuth is not None:
                delta = azimuth_deg - prev_azimuth
                if delta < 0:
                    delta += 360.0
                az_second = (azimuth_deg + delta / 2.0) % 360.0
            else:
                az_second = azimuth_deg

            prev_azimuth = azimuth_deg

            for seq in range(2):  # two firing sequences of 16 lasers each
                az = azimuth_deg if seq == 0 else az_second
                az_rad = math.radians(az)
                cos_az = math.cos(az_rad)
                sin_az = math.sin(az_rad)

                # Timing offset for this firing
                t_offset = (
                    block_idx * _FIRING_CYCLE_US
                    + seq * _SINGLE_FIRING_US * _LASERS
                )

                for ch in range(_LASERS):
                    dist_raw, intensity = struct.unpack_from(
                        "<HB", raw, offset
                    )
                    offset += 3

                    dist_m = dist_raw * _DISTANCE_RESOLUTION
                    if dist_m < 0.1:
                        # Below minimum range — skip (returns 0 for no return)
                        continue

                    r_xy = dist_m * _VERTICAL_COS[ch]
                    x = r_xy * sin_az
                    y = r_xy * cos_az
                    z = dist_m * _VERTICAL_SIN[ch]

                    points.append(
                        VLP16Point(
                            x=x,
                            y=y,
                            z=z,
                            intensity=float(intensity),
                            channel=ch,
                            azimuth_deg=az,
                            timestamp_us=timestamp_us + t_offset,
                            distance_m=dist_m,
                        )
                    )

        return points, timestamp_us


# ---------------------------------------------------------------------------
# High-level driver
# ---------------------------------------------------------------------------

class VLP16Driver:
    """
    High-level VLP-16 driver.

    Opens a UDP socket, receives packets in a background thread, accumulates
    complete 360° frames, and exposes them via :meth:`get_frame`.

    Parameters
    ----------
    host:
        Local IP address to bind (``"0.0.0.0"`` for all interfaces).
    port:
        UDP port to listen on (factory default: 2368).
    rpm:
        Sensor spin rate in revolutions per minute.  Used only for
        per-revolution timing calculations; the driver detects frame
        boundaries from azimuth wraparound regardless of this setting.
    max_queue:
        Maximum number of complete frames to buffer.  Older frames are
        discarded when the queue is full.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = _DATA_PORT,
        rpm: int = 600,
        max_queue: int = 10,
    ) -> None:
        self._host = host
        self._port = port
        self._rpm = rpm
        self._max_queue = max_queue
        self._parser = VLP16PacketParser()

        self._socket: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        self._lock = threading.Lock()
        self._frames: List[VLP16Frame] = []
        self._current_frame = VLP16Frame()
        self._last_azimuth: Optional[float] = None

        self.packets_received: int = 0
        self.frames_completed: int = 0
        self.parse_errors: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the UDP socket and start the background receiver thread."""
        if self._running:
            return
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.settimeout(1.0)
        self._socket.bind((self._host, self._port))
        self._running = True
        self._thread = threading.Thread(
            target=self._receive_loop, daemon=True, name="vlp16-recv"
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the receiver thread and close the socket."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def get_frame(self, timeout: float = 1.0) -> Optional[VLP16Frame]:
        """
        Block until a complete frame is available, then return it.

        Parameters
        ----------
        timeout:
            Maximum seconds to wait.

        Returns
        -------
        :class:`VLP16Frame` or ``None`` if the timeout elapsed.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._frames:
                    return self._frames.pop(0)
            time.sleep(0.005)
        return None

    def frames_available(self) -> int:
        """Return the number of buffered complete frames."""
        with self._lock:
            return len(self._frames)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _receive_loop(self) -> None:
        assert self._socket is not None
        while self._running:
            try:
                raw, _ = self._socket.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break

            self.packets_received += 1
            try:
                points, ts = self._parser.parse(raw)
            except ValueError:
                self.parse_errors += 1
                continue

            if not self._current_frame.timestamp_us:
                self._current_frame.timestamp_us = ts

            for point in points:
                az = point.azimuth_deg
                if (
                    self._last_azimuth is not None
                    and az < self._last_azimuth
                    and self._last_azimuth > 270.0
                    and az < 90.0
                ):
                    # Azimuth wrapped — frame complete
                    with self._lock:
                        self._frames.append(self._current_frame)
                        if len(self._frames) > self._max_queue:
                            self._frames.pop(0)
                    self.frames_completed += 1
                    self._current_frame = VLP16Frame(timestamp_us=ts)

                self._current_frame.points.append(point)
                self._last_azimuth = az
