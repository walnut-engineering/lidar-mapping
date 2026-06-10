"""Headless tests for :class:`MappingSession` (no Tk).

These tests exercise the session lifecycle, sensor wiring, and mapper
integration using the simulation backends.  The Tk dashboard itself is
not imported here so the tests are safe to run on headless CI.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

try:
    import open3d as o3d  # noqa: F401
    _O3D_AVAILABLE = True
except ImportError:
    _O3D_AVAILABLE = False

from lidar_mapping.config import KitConfig, LidarConfig, IMUConfig, MapperConfig
from lidar_mapping.ui.dashboard import MappingSession


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sim_config(with_imu=True):
    cfg = KitConfig()
    cfg.lidar = LidarConfig(enabled=True)
    cfg.imu = IMUConfig(enabled=with_imu, rate_hz=50)
    cfg.mapper = MapperConfig(
        voxel_size=0.3,        # coarse so registration is fast
        min_range=0.5,
        max_range=80.0,
        z_min=-3.0,
        z_max=20.0,
        remove_ground=False,
    )
    cfg.camera.enabled = False
    return cfg


def _wait_for(predicate, timeout=5.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_defaults(self):
        s = MappingSession()
        assert s.frames_processed == 0
        assert s.imu_samples == 0
        assert not s.lidar_running
        assert not s.imu_running
        assert s.last_error is None
        assert s.started_at is None
        assert s.map_size == 0
        assert not s.is_running()

    def test_accepts_explicit_config(self):
        cfg = _sim_config()
        s = MappingSession(config=cfg, simulate=True)
        assert s.config is cfg
        assert s.simulate is True


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _O3D_AVAILABLE, reason="open3d not installed")
class TestLifecycle:
    def test_start_then_stop_clean(self):
        s = MappingSession(config=_sim_config(with_imu=False), simulate=True)
        s.start()
        try:
            assert s.is_running()
            assert s.started_at is not None
            # Wait until the session reports lidar running
            assert _wait_for(lambda: s.lidar_running, timeout=2.0)
        finally:
            s.stop()
        assert not s.is_running()
        assert not s.lidar_running

    def test_start_twice_is_idempotent(self):
        s = MappingSession(config=_sim_config(with_imu=False), simulate=True)
        s.start()
        first_thread = s._thread
        s.start()  # should not spawn a second thread
        try:
            assert s._thread is first_thread
        finally:
            s.stop()

    def test_stop_without_start_is_noop(self):
        s = MappingSession(config=_sim_config(), simulate=True)
        s.stop()  # should not raise
        assert not s.is_running()

    def test_frames_processed_grows(self):
        s = MappingSession(config=_sim_config(with_imu=False), simulate=True)
        s.start()
        try:
            assert _wait_for(lambda: s.frames_processed >= 1, timeout=5.0)
        finally:
            s.stop()

    def test_imu_samples_collected(self):
        s = MappingSession(config=_sim_config(with_imu=True), simulate=True)
        s.start()
        try:
            assert _wait_for(lambda: s.imu_samples >= 5, timeout=5.0)
            assert s.imu_running
        finally:
            s.stop()

    def test_map_grows(self):
        s = MappingSession(config=_sim_config(with_imu=False), simulate=True)
        s.start()
        try:
            assert _wait_for(lambda: s.map_size > 0, timeout=5.0)
        finally:
            s.stop()


# ---------------------------------------------------------------------------
# save_map
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _O3D_AVAILABLE, reason="open3d not installed")
class TestSaveMap:
    def test_save_before_start_raises(self, tmp_path):
        s = MappingSession(config=_sim_config(), simulate=True)
        with pytest.raises(RuntimeError):
            s.save_map(tmp_path / "x.ply")

    def test_save_writes_file(self, tmp_path):
        s = MappingSession(config=_sim_config(with_imu=False), simulate=True)
        s.start()
        try:
            assert _wait_for(lambda: s.map_size > 0, timeout=5.0)
            out = tmp_path / "map.ply"
            s.save_map(out)
            assert out.exists()
            assert out.stat().st_size > 0
        finally:
            s.stop()


# ---------------------------------------------------------------------------
# Sensor builders
# ---------------------------------------------------------------------------

class TestSensorBuilders:
    def test_simulate_lidar(self):
        s = MappingSession(config=_sim_config(), simulate=True)
        from lidar_mapping.simulation import VLP16Simulator
        sensor = s._build_lidar()
        assert isinstance(sensor, VLP16Simulator)

    def test_simulate_imu(self):
        s = MappingSession(config=_sim_config(), simulate=True)
        from lidar_mapping.simulation import IMUSimulator
        sensor = s._build_imu()
        assert isinstance(sensor, IMUSimulator)

    def test_imu_driver_selection_witmotion(self, monkeypatch):
        cfg = _sim_config()
        cfg.imu.driver = "witmotion"
        s = MappingSession(config=cfg, simulate=False)

        captured = {}

        class FakeWit:
            def __init__(self, port, baud):
                captured["port"] = port
                captured["baud"] = baud

        monkeypatch.setattr(
            "lidar_mapping.sensors.imu.WitMotionDriver", FakeWit
        )
        sensor = s._build_imu()
        assert isinstance(sensor, FakeWit)
        assert captured["port"] == cfg.imu.port
        assert captured["baud"] == cfg.imu.baud

    def test_imu_driver_falls_back_to_serial_ahrs(self, monkeypatch):
        cfg = _sim_config()
        cfg.imu.driver = "unknown_driver"
        s = MappingSession(config=cfg, simulate=False)

        captured = {}

        class FakeSerial:
            def __init__(self, port, baud):
                captured["port"] = port

        monkeypatch.setattr(
            "lidar_mapping.sensors.imu.SerialAHRSDriver", FakeSerial
        )
        sensor = s._build_imu()
        assert isinstance(sensor, FakeSerial)


# ---------------------------------------------------------------------------
# Disabled sensors
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _O3D_AVAILABLE, reason="open3d not installed")
class TestDisabledSensors:
    def test_lidar_only(self):
        cfg = _sim_config(with_imu=False)
        s = MappingSession(config=cfg, simulate=True)
        s.start()
        try:
            assert _wait_for(lambda: s.lidar_running, timeout=2.0)
            assert not s.imu_running
            assert s.imu_samples == 0
        finally:
            s.stop()

    def test_no_sensors_enabled_does_not_crash(self):
        cfg = _sim_config()
        cfg.lidar.enabled = False
        cfg.imu.enabled = False
        s = MappingSession(config=cfg, simulate=True)
        s.start()
        try:
            # Without sensors the loop sleeps; verify nothing crashed
            time.sleep(0.3)
            assert s.is_running()
            assert s.last_error is None
        finally:
            s.stop()
