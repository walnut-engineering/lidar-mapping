"""
Sensor data playback from recorded files.

:class:`VLP16Playback` and :class:`IMUPlayback` implement the same
``get_frame()`` / ``get_reading()`` interfaces as the live hardware
drivers, so the rest of the pipeline works without any changes.
"""

from __future__ import annotations

import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# VLP-16 playback
# ---------------------------------------------------------------------------

class VLP16Playback:
    """
    Replay a VLP-16 recording created by :class:`~lidar_mapping.io.recorder.VLP16Recorder`.

    Implements the same ``start()`` / ``stop()`` / ``get_frame()`` interface
    as :class:`~lidar_mapping.sensors.vlp16.VLP16Driver`, so it can be used
    as a drop-in replacement in offline processing pipelines.

    Parameters
    ----------
    path:
        Path to the ``.vlp16`` binary file.
    speed:
        Playback speed multiplier.  ``1.0`` replays at original speed;
        ``0.0`` replays as fast as possible.
    loop:
        If ``True``, restart from the beginning when the file ends.
    """

    def __init__(
        self,
        path: Union[str, Path],
        speed: float = 1.0,
        loop: bool = False,
    ) -> None:
        self._path = Path(path)
        self._speed = max(speed, 0.0)
        self._loop = loop
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._frames: list = []  # VLP16Frame objects
        self.frames_played: int = 0

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "VLP16Playback":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the replay thread."""
        self._running = True
        self._frames.clear()
        self.frames_played = 0
        self._thread = threading.Thread(
            target=self._replay_loop, daemon=True, name="vlp16-playback"
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the replay thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    # ------------------------------------------------------------------
    # Frame access (same interface as VLP16Driver)
    # ------------------------------------------------------------------

    def get_frame(self, timeout: float = 2.0):
        """
        Block until a frame is available, then return it.

        Returns ``None`` if the timeout elapses.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._frames:
                    return self._frames.pop(0)
            time.sleep(0.005)
        return None

    # ------------------------------------------------------------------
    # Internal replay
    # ------------------------------------------------------------------

    def _replay_loop(self) -> None:
        from lidar_mapping.sensors.vlp16 import VLP16PacketParser, VLP16Frame

        parser = VLP16PacketParser()

        while self._running:
            try:
                packets = list(_iter_vlp16_file(self._path))
            except (FileNotFoundError, OSError) as exc:
                import logging
                logging.getLogger(__name__).error(
                    "VLP16Playback: cannot read %s: %s", self._path, exc
                )
                return

            if not packets:
                return

            t0_file = packets[0][0]
            t0_wall = time.monotonic()

            current_frame = VLP16Frame()
            last_azimuth: Optional[float] = None

            for ts_file, raw in packets:
                if not self._running:
                    return

                if self._speed > 0.0:
                    # Sleep to maintain original timing (scaled by speed)
                    elapsed_file = ts_file - t0_file
                    elapsed_wall = time.monotonic() - t0_wall
                    wait = elapsed_file / self._speed - elapsed_wall
                    if wait > 0:
                        time.sleep(wait)

                try:
                    points, ts_us = parser.parse(raw)
                except ValueError:
                    continue

                if not current_frame.timestamp_us:
                    current_frame.timestamp_us = ts_us

                for point in points:
                    az = point.azimuth_deg
                    if (
                        last_azimuth is not None
                        and az < last_azimuth
                        and last_azimuth > 270.0
                        and az < 90.0
                    ):
                        with self._lock:
                            self._frames.append(current_frame)
                            if len(self._frames) > 64:
                                self._frames.pop(0)
                        self.frames_played += 1
                        current_frame = VLP16Frame(timestamp_us=ts_us)
                    current_frame.points.append(point)
                    last_azimuth = az

            # Flush any partial frame at end of file
            if current_frame.points:
                with self._lock:
                    self._frames.append(current_frame)
                self.frames_played += 1

            if not self._loop:
                break

    @staticmethod
    def iter_raw_packets(path: Union[str, Path]) -> Iterator[tuple[float, bytes]]:
        """Yield ``(timestamp, raw_bytes)`` from a ``.vlp16`` file."""
        yield from _iter_vlp16_file(path)


def _iter_vlp16_file(path: Path) -> Iterator[tuple[float, bytes]]:
    """Read all ``(timestamp, payload)`` records from a .vlp16 file."""
    with open(path, "rb") as f:
        while True:
            header = f.read(10)  # 8-byte float64 + 2-byte uint16
            if not header:
                break
            if len(header) < 10:
                break
            ts, length = struct.unpack(">dH", header)
            payload = f.read(length)
            if len(payload) < length:
                break
            yield ts, payload


# ---------------------------------------------------------------------------
# IMU playback
# ---------------------------------------------------------------------------

class IMUPlayback:
    """
    Replay an IMU recording created by :class:`~lidar_mapping.io.recorder.IMURecorder`.

    Implements the same ``get_reading()`` / ``get_latest_reading()`` /
    ``readings_available()`` interface as
    :class:`~lidar_mapping.sensors.imu.BaseIMUDriver`.

    Parameters
    ----------
    path:
        Path to the ``.npz`` archive.
    speed:
        Playback speed multiplier.  ``1.0`` replays at original speed;
        ``0.0`` replays as fast as possible.
    loop:
        If ``True``, restart from the beginning when the file ends.
    """

    def __init__(
        self,
        path: Union[str, Path],
        speed: float = 1.0,
        loop: bool = False,
    ) -> None:
        self._path = Path(path)
        self._speed = max(speed, 0.0)
        self._loop = loop
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._readings: list = []
        self.samples_played: int = 0

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "IMUPlayback":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Load the archive and start the replay thread."""
        self._running = True
        self._readings.clear()
        self.samples_played = 0
        self._thread = threading.Thread(
            target=self._replay_loop, daemon=True, name="imu-playback"
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the replay thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    # ------------------------------------------------------------------
    # Reading access (same interface as BaseIMUDriver)
    # ------------------------------------------------------------------

    def get_reading(self, timeout: float = 1.0):
        """Block until a reading is available, or return ``None``."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._readings:
                    return self._readings.pop(0)
            time.sleep(0.002)
        return None

    def get_latest_reading(self):
        """Return the most-recent reading without blocking, or ``None``."""
        with self._lock:
            if self._readings:
                reading = self._readings[-1]
                self._readings.clear()
                return reading
        return None

    def readings_available(self) -> int:
        """Return the number of buffered readings."""
        with self._lock:
            return len(self._readings)

    # ------------------------------------------------------------------
    # Internal replay
    # ------------------------------------------------------------------

    def _replay_loop(self) -> None:
        from lidar_mapping.sensors.imu import IMUReading

        data = np.load(str(self._path))
        timestamps  = data["timestamp"]
        accels      = data["accel"]
        gyros       = data["gyro"]
        mags        = data["mag"]
        temps       = data["temperature"]
        quats       = data["quaternion"]
        rolls       = data["roll"]
        pitches     = data["pitch"]
        yaws        = data["yaw"]
        n = len(timestamps)

        if n == 0:
            return  # nothing to replay

        while self._running:
            t0_file = float(timestamps[0])
            t0_wall = time.monotonic()

            for i in range(n):
                if not self._running:
                    return

                if self._speed > 0.0:
                    elapsed_file = float(timestamps[i]) - t0_file
                    elapsed_wall = time.monotonic() - t0_wall
                    wait = elapsed_file / self._speed - elapsed_wall
                    if wait > 0:
                        time.sleep(wait)

                mag_row = mags[i]
                mag = (
                    None if np.all(np.isnan(mag_row))
                    else mag_row.copy()
                )
                temp = (
                    None if np.isnan(temps[i])
                    else float(temps[i])
                )

                reading = IMUReading(
                    timestamp=float(timestamps[i]),
                    accel_mss=accels[i].copy(),
                    gyro_rads=gyros[i].copy(),
                    mag_ut=mag,
                    temperature_c=temp,
                    roll_deg=float(rolls[i]),
                    pitch_deg=float(pitches[i]),
                    yaw_deg=float(yaws[i]),
                    quaternion=quats[i].copy(),
                )

                with self._lock:
                    self._readings.append(reading)

                self.samples_played += 1

            if not self._loop:
                break
