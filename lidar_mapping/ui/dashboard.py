"""
Touchscreen dashboard for the official 7" Raspberry Pi display (800×480).

A lightweight Tk-based UI that runs alongside the mapping pipeline.
Designed for finger-friendly tap targets (≥60 px) and high-contrast colours.

Layout (800×480):
    ┌──────────────────────────────────────────────────────┐
    │  STATUS BAR  (lidar/imu/cam state, frame counts)     │
    ├────────────────────────────┬─────────────────────────┤
    │                            │  [START]   [STOP]       │
    │  Live map / camera preview │  [SAVE MAP]             │
    │  (matplotlib or canvas)    │  [SCREENSHOT]           │
    │                            │  [QUIT]                 │
    └────────────────────────────┴─────────────────────────┘

The UI talks to a :class:`MappingSession` (engine) that owns the sensors and
mapper.  This module imports Tk lazily so the rest of the package works in
headless environments.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from lidar_mapping.config import KitConfig, kit_config

log = logging.getLogger(__name__)

# Tap-friendly minimum height (px)
_TAP_H = 60
_BG = "#101820"
_FG = "#F0F4F8"
_ACCENT = "#FFB81C"
_OK = "#4CAF50"
_WARN = "#FF9800"
_ERR = "#E53935"


# ---------------------------------------------------------------------------
# Mapping engine — runs sensors + mapper in background threads
# ---------------------------------------------------------------------------

class MappingSession:
    """
    Encapsulates the sensors, recorder, and mapper.

    Designed so the UI can poll cheap status without holding the GIL on
    sensor work, and so the same session can be driven from a CLI or REPL.
    """

    def __init__(
        self,
        config: Optional[KitConfig] = None,
        simulate: bool = False,
    ) -> None:
        self.config = config or kit_config()
        self.simulate = simulate
        self._mapper = None
        self._lidar = None
        self._imu = None
        self._cam = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # status
        self.frames_processed = 0
        self.imu_samples = 0
        self.lidar_running = False
        self.imu_running = False
        self.camera_running = False
        self.last_error: Optional[str] = None
        self.started_at: Optional[float] = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self.last_error = None
        self.frames_processed = 0
        self.imu_samples = 0
        self.started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._thread = None
        for s in (self._lidar, self._imu, self._cam):
            if s is not None and hasattr(s, "stop"):
                try:
                    s.stop()
                except Exception:  # pragma: no cover
                    pass
        self.lidar_running = False
        self.imu_running = False
        self.camera_running = False

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def save_map(self, path: str | Path) -> None:
        if self._mapper is None or self._mapper.map_points is None:
            raise RuntimeError("No map to save")
        self._mapper.save_map(path)

    @property
    def map_size(self) -> int:
        if self._mapper is None or self._mapper.map_points is None:
            return 0
        return len(self._mapper.map_points)

    # ------------------------------------------------------------------
    def _run(self) -> None:
        try:
            from lidar_mapping.mapping.mapper import Mapper
            from lidar_mapping.mapping.imu_preintegrator import IMUPreintegrator

            preint = IMUPreintegrator() if self.config.imu.enabled else None
            self._mapper = Mapper(
                voxel_size=self.config.mapper.voxel_size,
                min_range=self.config.mapper.min_range,
                max_range=self.config.mapper.max_range,
                z_min=self.config.mapper.z_min,
                z_max=self.config.mapper.z_max,
                remove_ground=self.config.mapper.remove_ground,
                imu_preintegrator=preint,
            )

            if self.config.lidar.enabled:
                self._lidar = self._build_lidar()
                self._lidar.start()
                self.lidar_running = True
            if self.config.imu.enabled:
                self._imu = self._build_imu()
                self._imu.start()
                self.imu_running = True

            while not self._stop.is_set():
                if self._lidar is not None:
                    frame = self._lidar.get_frame(timeout=0.2)
                    if frame is None:
                        continue
                    # Drain IMU
                    if self._imu is not None and preint is not None:
                        while self._imu.readings_available():
                            r = self._imu.get_reading(timeout=0)
                            if r is None:
                                break
                            preint.push(r)
                            self.imu_samples += 1
                    arr = frame.to_numpy()
                    if len(arr):
                        self._mapper.add_scan(arr)
                        self.frames_processed += 1
                else:
                    time.sleep(0.1)
        except Exception as exc:  # pragma: no cover (interactive)
            self.last_error = str(exc)
            log.exception("Mapping session crashed: %s", exc)

    def _build_lidar(self):
        if self.simulate:
            from lidar_mapping.simulation import VLP16Simulator
            return VLP16Simulator()
        from lidar_mapping.sensors.vlp16 import VLP16Driver
        return VLP16Driver(
            host=self.config.lidar.host,
            data_port=self.config.lidar.data_port,
            position_port=self.config.lidar.position_port,
        )

    def _build_imu(self):
        if self.simulate:
            from lidar_mapping.simulation import IMUSimulator
            return IMUSimulator(rate_hz=self.config.imu.rate_hz)
        name = self.config.imu.driver.lower()
        if name == "witmotion":
            from lidar_mapping.sensors.imu import WitMotionDriver
            return WitMotionDriver(port=self.config.imu.port,
                                   baud=self.config.imu.baud)
        if name == "mpu9250":
            from lidar_mapping.sensors.imu import MPU9250Driver
            return MPU9250Driver(i2c_bus=self.config.imu.i2c_bus)
        from lidar_mapping.sensors.imu import SerialAHRSDriver
        return SerialAHRSDriver(port=self.config.imu.port,
                                baud=self.config.imu.baud)


# ---------------------------------------------------------------------------
# Tk dashboard
# ---------------------------------------------------------------------------

class TouchscreenDashboard:
    """
    Tkinter dashboard tuned for the 7" Pi touchscreen.

    Run with::

        from lidar_mapping.ui.dashboard import TouchscreenDashboard
        TouchscreenDashboard(simulate=True).run()
    """

    def __init__(
        self,
        session: Optional[MappingSession] = None,
        config: Optional[KitConfig] = None,
        simulate: bool = False,
    ) -> None:
        self.config = config or kit_config()
        self.session = session or MappingSession(self.config, simulate=simulate)

    def run(self) -> None:  # pragma: no cover (interactive)
        import tkinter as tk
        from tkinter import ttk, filedialog

        root = tk.Tk()
        root.title("lidar-mapping")
        w, h = self.config.ui.width, self.config.ui.height
        root.geometry(f"{w}x{h}")
        root.configure(bg=_BG)
        if self.config.ui.fullscreen:
            root.attributes("-fullscreen", True)

        # Layout
        status = tk.Label(
            root, text="", bg=_BG, fg=_FG, anchor="w",
            font=("Helvetica", 12), padx=10,
        )
        status.place(x=0, y=0, width=w, height=40)

        # Map preview area
        preview = tk.Canvas(root, bg="black", highlightthickness=0)
        preview.place(x=0, y=40, width=w - 200, height=h - 40)

        # Side button bar
        side = tk.Frame(root, bg=_BG)
        side.place(x=w - 200, y=40, width=200, height=h - 40)

        def make_btn(text, command, color=_ACCENT):
            return tk.Button(
                side, text=text, command=command,
                bg=color, fg="black",
                font=("Helvetica", 13, "bold"),
                activebackground=_FG,
                relief="flat",
            )

        btn_start = make_btn("START", self.session.start, _OK)
        btn_stop = make_btn("STOP", self.session.stop, _WARN)
        btn_save = make_btn("SAVE MAP", lambda: self._save(filedialog, root))
        btn_quit = make_btn("QUIT", root.destroy, _ERR)

        for i, b in enumerate((btn_start, btn_stop, btn_save, btn_quit)):
            b.place(x=10, y=10 + i * (_TAP_H + 10), width=180, height=_TAP_H)

        # Live update loop
        period_ms = int(1000 / max(1, self.config.ui.fps))

        def tick():
            self._render_status(status)
            self._render_preview(preview)
            root.after(period_ms, tick)

        root.after(period_ms, tick)
        root.mainloop()
        # ensure background threads stop on close
        self.session.stop()

    # ------------------------------------------------------------------
    def _render_status(self, lbl):
        s = self.session
        bits = []
        bits.append("[●] LIDAR" if s.lidar_running else "[○] LIDAR")
        bits.append("[●] IMU" if s.imu_running else "[○] IMU")
        bits.append(f"frames={s.frames_processed}")
        bits.append(f"imu={s.imu_samples}")
        bits.append(f"map={s.map_size}")
        if s.last_error:
            bits.append(f"ERR: {s.last_error}")
        lbl.configure(text="  ".join(bits))

    def _render_preview(self, canvas):
        """Simple top-down scatter of the current map onto the canvas."""
        import numpy as np
        canvas.delete("pts")
        pts = self.session._mapper.map_points if self.session._mapper else None
        if pts is None or len(pts) == 0:
            return
        w = int(canvas.winfo_width())
        h = int(canvas.winfo_height())
        if w <= 1 or h <= 1:
            return
        # Downsample for cheap rendering
        if len(pts) > 5000:
            idx = np.random.default_rng(0).choice(len(pts), 5000, replace=False)
            pts = pts[idx]
        xs = pts[:, 0]
        ys = pts[:, 1]
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())
        if x_max == x_min or y_max == y_min:
            return
        sx = (w - 20) / (x_max - x_min)
        sy = (h - 20) / (y_max - y_min)
        scale = min(sx, sy)
        cx = w / 2 - (x_min + x_max) / 2 * scale
        cy = h / 2 + (y_min + y_max) / 2 * scale
        for x, y in zip(xs, ys):
            px = cx + x * scale
            py = cy - y * scale
            canvas.create_oval(px - 1, py - 1, px + 1, py + 1,
                               fill=_ACCENT, outline="", tags="pts")

    def _save(self, filedialog, root):  # pragma: no cover (interactive)
        path = filedialog.asksaveasfilename(
            parent=root,
            defaultextension=".pcd",
            filetypes=[("PCD", "*.pcd"), ("PLY", "*.ply"), ("XYZ", "*.xyz")],
        )
        if path:
            try:
                self.session.save_map(path)
            except Exception as exc:
                self.session.last_error = f"Save failed: {exc}"


def main(argv=None) -> int:  # pragma: no cover (interactive)
    import argparse
    ap = argparse.ArgumentParser(prog="lidar-ui")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--fullscreen", action="store_true")
    args = ap.parse_args(argv)
    if args.config:
        from lidar_mapping.config import load_config
        cfg = load_config(args.config)
    else:
        cfg = kit_config()
    if args.fullscreen:
        cfg.ui.fullscreen = True
    TouchscreenDashboard(config=cfg, simulate=args.simulate).run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
