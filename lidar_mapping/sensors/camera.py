"""
Camera capture module.

Supports USB / CSI cameras via OpenCV and provides optional
depth-map fusion with VLP-16 point clouds.

Usage example::

    from lidar_mapping.sensors.camera import CameraCapture

    cam = CameraCapture(device_index=0, width=1280, height=720, fps=30)
    cam.start()
    try:
        while True:
            frame = cam.get_frame(timeout=1.0)
            if frame is not None:
                # frame.image  -> (H, W, 3) uint8 BGR array
                # frame.timestamp -> monotonic time in seconds
    finally:
        cam.stop()
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CV2_AVAILABLE = False


@dataclass
class CameraFrame:
    """A single captured camera frame."""

    image: np.ndarray       # (H, W, 3) uint8 BGR array
    timestamp: float        # monotonic capture time (seconds)
    frame_index: int        # sequential frame counter


class CameraCapture:
    """
    Threaded camera capture using OpenCV.

    Grabs frames in a background thread so the main thread never blocks
    on slow camera I/O.

    Parameters
    ----------
    device_index:
        OpenCV device index (0 for first USB camera, or a GStreamer pipeline
        string for CSI cameras on Raspberry Pi).
    width, height:
        Requested capture resolution.  The camera may silently round to the
        nearest supported resolution.
    fps:
        Requested frame rate.
    max_queue:
        Maximum number of frames to buffer before dropping the oldest.
    """

    def __init__(
        self,
        device_index: int | str = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        max_queue: int = 5,
    ) -> None:
        if not _CV2_AVAILABLE:
            raise ImportError(
                "opencv-python is required for CameraCapture. "
                "Install it with: pip install opencv-python"
            )
        self._device_index = device_index
        self._width = width
        self._height = height
        self._fps = fps
        self._max_queue = max_queue

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        self._lock = threading.Lock()
        self._frames: list[CameraFrame] = []
        self._frame_counter: int = 0

        self.frames_captured: int = 0
        self.frames_dropped: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the camera and start the capture thread."""
        if self._running:
            return
        self._cap = cv2.VideoCapture(self._device_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera device {self._device_index!r}"
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="camera-capture"
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the capture thread and release the camera."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def get_frame(self, timeout: float = 1.0) -> Optional[CameraFrame]:
        """
        Block until a frame is available, then return it.

        Parameters
        ----------
        timeout:
            Maximum seconds to wait.

        Returns
        -------
        :class:`CameraFrame` or ``None`` if the timeout elapsed.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._frames:
                    return self._frames.pop(0)
            time.sleep(0.005)
        return None

    def get_latest_frame(self) -> Optional[CameraFrame]:
        """Return the most recent frame without blocking, or ``None``."""
        with self._lock:
            if self._frames:
                frame = self._frames[-1]
                self._frames.clear()
                return frame
        return None

    def frames_available(self) -> int:
        """Return the number of buffered frames."""
        with self._lock:
            return len(self._frames)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def resolution(self) -> Tuple[int, int]:
        """Return the actual capture resolution as ``(width, height)``."""
        if self._cap is None:
            return (self._width, self._height)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        assert self._cap is not None
        while self._running:
            ret, img = self._cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            ts = time.monotonic()
            self._frame_counter += 1
            frame = CameraFrame(
                image=img,
                timestamp=ts,
                frame_index=self._frame_counter,
            )
            with self._lock:
                self._frames.append(frame)
                if len(self._frames) > self._max_queue:
                    self._frames.pop(0)
                    self.frames_dropped += 1
            self.frames_captured += 1
