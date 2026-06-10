"""
Sensor data recorders.

File formats
------------
**VLP-16 binary** (``.vlp16``):
    A sequence of variable-length records::

        [8 bytes: float64 timestamp (seconds, monotonic)]
        [2 bytes: uint16  payload length in bytes]
        [N bytes: raw UDP payload]

    Records are written in arrival order with no padding.  The file can be
    replayed with :class:`~lidar_mapping.io.playback.VLP16Playback`.

**IMU NumPy archive** (``.npz``):
    A compressed NumPy archive with the following arrays (one row per
    reading, in time order):

    * ``timestamp``  — float64, seconds
    * ``accel``      — float64 (N, 3), m/s²
    * ``gyro``       — float64 (N, 3), rad/s
    * ``mag``        — float64 (N, 3) or all-NaN if device has no magnetometer
    * ``temperature``— float64 (N,), °C or NaN
    * ``quaternion`` — float64 (N, 4), [w, x, y, z]
    * ``roll``       — float64 (N,), degrees
    * ``pitch``      — float64 (N,), degrees
    * ``yaw``        — float64 (N,), degrees
"""

from __future__ import annotations

import io
import struct
import threading
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# VLP-16 recorder
# ---------------------------------------------------------------------------

class VLP16Recorder:
    """
    Record raw VLP-16 UDP packets from a live :class:`~lidar_mapping.sensors.vlp16.VLP16Driver`
    to a binary file.

    Can be used as a context manager::

        from lidar_mapping.sensors.vlp16 import VLP16Driver
        from lidar_mapping.io import VLP16Recorder

        driver = VLP16Driver()
        with VLP16Recorder(driver, "session.vlp16"):
            driver.start()
            time.sleep(10)
            driver.stop()

    Or manually::

        recorder = VLP16Recorder(driver, "session.vlp16")
        recorder.start()
        ...
        recorder.stop()

    Parameters
    ----------
    driver:
        A :class:`~lidar_mapping.sensors.vlp16.VLP16Driver` instance.
        The recorder hooks into the driver's internal packet callback.
    path:
        Output file path.  Any parent directories are created automatically.
    """

    def __init__(self, driver, path: Union[str, Path]) -> None:
        self._driver = driver
        self._path = Path(path)
        self._file: Optional[io.BufferedWriter] = None
        self._lock = threading.Lock()
        self._packets_written: int = 0

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "VLP16Recorder":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the output file and attach the packet hook."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "wb")  # noqa: WPS515
        self._packets_written = 0
        # Monkey-patch the driver's _on_packet method if it exists,
        # otherwise wrap the internal _handle_packet callback
        self._orig_callback = getattr(self._driver, "_packet_callback", None)
        self._driver._packet_callback = self._record_packet

    def stop(self) -> None:
        """Detach the packet hook and close the file."""
        if hasattr(self._driver, "_packet_callback"):
            self._driver._packet_callback = self._orig_callback
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None

    # ------------------------------------------------------------------
    # Packet handler
    # ------------------------------------------------------------------

    def _record_packet(self, raw: bytes) -> None:
        """Write one UDP payload to the file."""
        ts = time.monotonic()
        header = struct.pack(">dH", ts, len(raw))
        with self._lock:
            if self._file is not None:
                self._file.write(header)
                self._file.write(raw)
                self._packets_written += 1
        # Also forward to original callback if there was one
        if self._orig_callback is not None:
            self._orig_callback(raw)

    @property
    def packets_written(self) -> int:
        """Number of packets written so far."""
        return self._packets_written


# ---------------------------------------------------------------------------
# IMU recorder
# ---------------------------------------------------------------------------

class IMURecorder:
    """
    Record :class:`~lidar_mapping.sensors.imu.IMUReading` samples from a live
    IMU driver to a compressed NumPy ``.npz`` archive.

    Can be used as a context manager::

        from lidar_mapping.sensors.imu import WitMotionDriver
        from lidar_mapping.io import IMURecorder

        imu = WitMotionDriver(port="/dev/ttyUSB0", baudrate=115200)
        with IMURecorder(imu, "imu_session.npz"):
            imu.start()
            time.sleep(30)
            imu.stop()

    Parameters
    ----------
    driver:
        Any :class:`~lidar_mapping.sensors.imu.BaseIMUDriver` instance.
    path:
        Output ``.npz`` file path.
    poll_rate_hz:
        How often per second to drain the driver's reading buffer.
    """

    def __init__(
        self,
        driver,
        path: Union[str, Path],
        poll_rate_hz: float = 200.0,
    ) -> None:
        self._driver = driver
        self._path = Path(path)
        self._poll_rate = poll_rate_hz
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Accumulators
        self._timestamps: list[float] = []
        self._accels: list[np.ndarray] = []
        self._gyros:  list[np.ndarray] = []
        self._mags:   list[np.ndarray] = []
        self._temps:  list[float] = []
        self._quats:  list[np.ndarray] = []
        self._rolls:  list[float] = []
        self._pitches: list[float] = []
        self._yaws:   list[float] = []

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "IMURecorder":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background polling thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="imu-recorder"
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop polling and save all accumulated readings to disk."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._save()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        period = 1.0 / self._poll_rate
        while self._running:
            reading = self._driver.get_latest_reading()
            if reading is not None:
                self._timestamps.append(reading.timestamp)
                self._accels.append(reading.accel_mss)
                self._gyros.append(reading.gyro_rads)
                self._mags.append(
                    reading.mag_ut if reading.mag_ut is not None
                    else np.full(3, np.nan)
                )
                self._temps.append(
                    reading.temperature_c
                    if reading.temperature_c is not None
                    else np.nan
                )
                self._quats.append(reading.quaternion)
                self._rolls.append(reading.roll_deg)
                self._pitches.append(reading.pitch_deg)
                self._yaws.append(reading.yaw_deg)
            time.sleep(period)

    def _save(self) -> None:
        if not self._timestamps:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(self._path),
            timestamp=np.array(self._timestamps, dtype=np.float64),
            accel=np.array(self._accels,   dtype=np.float64),
            gyro=np.array(self._gyros,    dtype=np.float64),
            mag=np.array(self._mags,     dtype=np.float64),
            temperature=np.array(self._temps,  dtype=np.float64),
            quaternion=np.array(self._quats,  dtype=np.float64),
            roll=np.array(self._rolls,  dtype=np.float64),
            pitch=np.array(self._pitches, dtype=np.float64),
            yaw=np.array(self._yaws,   dtype=np.float64),
        )

    @property
    def samples_recorded(self) -> int:
        """Number of samples accumulated so far."""
        return len(self._timestamps)
