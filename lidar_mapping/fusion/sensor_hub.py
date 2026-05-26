"""
Time-synchronized sensor hub.

Buffers timestamped readings from the LiDAR, IMU, and camera drivers and
exposes lookup helpers used by the fusion frontends. Readings are pushed
in by lightweight ingest threads that drain each driver's internal queue.

The hub is intentionally lock-light: each sensor stream has its own deque
+ lock so a slow consumer of one sensor does not stall the others.
"""

from __future__ import annotations

import bisect
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class _Buffer:
    """A timestamp-sorted ring buffer keyed by ``time.monotonic()``."""
    times: Deque[float]
    values: Deque[object]
    lock: threading.Lock

    @classmethod
    def make(cls, maxlen: int) -> "_Buffer":
        return cls(times=deque(maxlen=maxlen),
                   values=deque(maxlen=maxlen),
                   lock=threading.Lock())

    def push(self, t: float, v: object) -> None:
        with self.lock:
            self.times.append(t)
            self.values.append(v)

    def latest(self) -> Optional[tuple[float, object]]:
        with self.lock:
            if not self.times:
                return None
            return self.times[-1], self.values[-1]

    def range(self, t0: float, t1: float) -> list[tuple[float, object]]:
        """Return all (t, v) with t0 <= t <= t1."""
        with self.lock:
            ts = list(self.times)
            vs = list(self.values)
        lo = bisect.bisect_left(ts, t0)
        hi = bisect.bisect_right(ts, t1)
        return list(zip(ts[lo:hi], vs[lo:hi]))

    def nearest(self, t: float) -> Optional[tuple[float, object]]:
        with self.lock:
            ts = list(self.times)
            vs = list(self.values)
        if not ts:
            return None
        i = bisect.bisect_left(ts, t)
        if i == 0:
            return ts[0], vs[0]
        if i >= len(ts):
            return ts[-1], vs[-1]
        if abs(ts[i] - t) < abs(ts[i - 1] - t):
            return ts[i], vs[i]
        return ts[i - 1], vs[i - 1]


class SensorHub:
    """
    Centralized timestamped buffer for LiDAR/IMU/Camera streams.

    Drivers are polled in dedicated daemon threads via the ``ingest_*``
    methods. Each push is timestamped with ``time.monotonic()`` at ingest
    so all streams share a common clock.
    """

    def __init__(self, lidar_buf: int = 64, imu_buf: int = 4096, cam_buf: int = 16):
        self.lidar = _Buffer.make(lidar_buf)
        self.imu = _Buffer.make(imu_buf)
        self.camera = _Buffer.make(cam_buf)
        self._running = False
        self._threads: list[threading.Thread] = []

        # Rate trackers
        self._last_lidar_rate_t = time.monotonic()
        self._last_imu_rate_t = time.monotonic()
        self._last_cam_rate_t = time.monotonic()
        self._lidar_count_since = 0
        self._imu_count_since = 0
        self._cam_count_since = 0
        self.lidar_hz = 0.0
        self.imu_hz = 0.0
        self.camera_hz = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start_lidar_ingest(self, driver, poll_period: float = 0.005) -> None:
        self._start_thread(self._lidar_loop, (driver, poll_period), "hub-lidar")

    def start_imu_ingest(self, driver, poll_period: float = 0.001) -> None:
        self._start_thread(self._imu_loop, (driver, poll_period), "hub-imu")

    def start_camera_ingest(self, getter: Callable[[], Optional[np.ndarray]],
                            poll_period: float = 0.02) -> None:
        """``getter`` returns a BGR ndarray or None."""
        self._start_thread(self._cam_loop, (getter, poll_period), "hub-cam")

    def stop(self) -> None:
        self._running = False
        for th in self._threads:
            th.join(timeout=1.0)
        self._threads.clear()

    def _start_thread(self, target, args, name) -> None:
        self._running = True
        th = threading.Thread(target=target, args=args, daemon=True, name=name)
        th.start()
        self._threads.append(th)

    # ------------------------------------------------------------------
    # Ingest loops
    # ------------------------------------------------------------------
    def _lidar_loop(self, driver, poll_period: float) -> None:
        while self._running:
            try:
                if driver.frames_available() > 0:
                    frame = driver.get_frame(timeout=0.01)
                    if frame is not None:
                        self.lidar.push(time.monotonic(), frame)
                        self._lidar_count_since += 1
                else:
                    time.sleep(poll_period)
            except Exception as exc:  # noqa: BLE001
                logger.debug("lidar ingest: %s", exc)
                time.sleep(poll_period)
            self._update_rate("lidar")

    def _imu_loop(self, driver, poll_period: float) -> None:
        while self._running:
            try:
                # Drain all available readings in one pass
                drained_any = False
                while driver.readings_available() if hasattr(driver, 'readings_available') else True:
                    r = driver.get_reading(timeout=0) if hasattr(driver, 'get_reading') \
                        else driver.get_latest_reading()
                    if r is None:
                        break
                    self.imu.push(time.monotonic(), r)
                    self._imu_count_since += 1
                    drained_any = True
                    if not hasattr(driver, 'readings_available'):
                        break  # WitMotionDriver: get_latest clears; one per loop
                if not drained_any:
                    time.sleep(poll_period)
            except Exception as exc:  # noqa: BLE001
                logger.debug("imu ingest: %s", exc)
                time.sleep(poll_period)
            self._update_rate("imu")

    def _cam_loop(self, getter, poll_period: float) -> None:
        while self._running:
            try:
                frame = getter()
                if frame is not None:
                    self.camera.push(time.monotonic(), frame)
                    self._cam_count_since += 1
                else:
                    time.sleep(poll_period)
            except Exception as exc:  # noqa: BLE001
                logger.debug("cam ingest: %s", exc)
                time.sleep(poll_period)
            self._update_rate("camera")

    def _update_rate(self, which: str) -> None:
        now = time.monotonic()
        if which == "lidar":
            dt = now - self._last_lidar_rate_t
            if dt >= 1.0:
                self.lidar_hz = self._lidar_count_since / dt
                self._lidar_count_since = 0
                self._last_lidar_rate_t = now
        elif which == "imu":
            dt = now - self._last_imu_rate_t
            if dt >= 1.0:
                self.imu_hz = self._imu_count_since / dt
                self._imu_count_since = 0
                self._last_imu_rate_t = now
        elif which == "camera":
            dt = now - self._last_cam_rate_t
            if dt >= 1.0:
                self.camera_hz = self._cam_count_since / dt
                self._cam_count_since = 0
                self._last_cam_rate_t = now

    # ------------------------------------------------------------------
    # Time-sync query helpers
    # ------------------------------------------------------------------
    def imu_between(self, t0: float, t1: float) -> list:
        """Return IMU readings (values only) with t0 <= t <= t1."""
        return [v for _, v in self.imu.range(t0, t1)]

    def imu_nearest(self, t: float):
        """Return the IMU reading closest to ``t`` (value only) or None."""
        r = self.imu.nearest(t)
        return r[1] if r else None

    def latest_lidar(self):
        r = self.lidar.latest()
        return r[1] if r else None

    def latest_camera(self):
        r = self.camera.latest()
        return r[1] if r else None
