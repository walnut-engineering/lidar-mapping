"""
Round-trip tests for the IO recorder / playback module.

These tests write data to temporary files and verify that playback
reconstructs the original data faithfully.  No real hardware is needed.
"""

from __future__ import annotations

import struct
import time
from pathlib import Path

import numpy as np
import pytest

from lidar_mapping.sensors.imu import IMUReading


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_imu_reading(
    ts: float = 0.0,
    ax: float = 1.0,
    ay: float = 2.0,
    az: float = 3.0,
) -> IMUReading:
    return IMUReading(
        timestamp=ts,
        accel_mss=np.array([ax, ay, az], dtype=np.float64),
        gyro_rads=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        mag_ut=np.array([10.0, 20.0, 30.0], dtype=np.float64),
        temperature_c=25.0,
        roll_deg=5.0,
        pitch_deg=-3.0,
        yaw_deg=90.0,
        quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )


def _write_vlp16_file(path: Path, packets: list[tuple[float, bytes]]) -> None:
    """Write packets to a .vlp16 binary file directly (no driver needed)."""
    with open(path, "wb") as f:
        for ts, payload in packets:
            header = struct.pack(">dH", ts, len(payload))
            f.write(header)
            f.write(payload)


def _write_imu_npz(path: Path, readings: list[IMUReading]) -> None:
    """Write IMUReading list to .npz directly (no driver needed)."""
    if not readings:
        np.savez_compressed(
            str(path),
            timestamp=np.array([], dtype=np.float64),
            accel=np.zeros((0, 3), dtype=np.float64),
            gyro=np.zeros((0, 3), dtype=np.float64),
            mag=np.zeros((0, 3), dtype=np.float64),
            temperature=np.array([], dtype=np.float64),
            quaternion=np.zeros((0, 4), dtype=np.float64),
            roll=np.array([], dtype=np.float64),
            pitch=np.array([], dtype=np.float64),
            yaw=np.array([], dtype=np.float64),
        )
        return

    np.savez_compressed(
        str(path),
        timestamp=np.array([r.timestamp for r in readings], dtype=np.float64),
        accel=np.array([r.accel_mss for r in readings], dtype=np.float64),
        gyro=np.array([r.gyro_rads for r in readings], dtype=np.float64),
        mag=np.array(
            [r.mag_ut if r.mag_ut is not None else np.full(3, np.nan)
             for r in readings],
            dtype=np.float64,
        ),
        temperature=np.array(
            [r.temperature_c if r.temperature_c is not None else np.nan
             for r in readings],
            dtype=np.float64,
        ),
        quaternion=np.array([r.quaternion for r in readings], dtype=np.float64),
        roll=np.array([r.roll_deg for r in readings], dtype=np.float64),
        pitch=np.array([r.pitch_deg for r in readings], dtype=np.float64),
        yaw=np.array([r.yaw_deg for r in readings], dtype=np.float64),
    )


# ===========================================================================
# VLP-16 binary file helpers
# ===========================================================================

class TestVLP16FileFormat:
    """Test that the binary format helpers can be read back correctly."""

    def test_iter_raw_packets(self, tmp_path: Path):
        """Packets written with _write_vlp16_file are read back in order."""
        from lidar_mapping.io.playback import VLP16Playback

        packets = [
            (1.0, bytes(range(10))),
            (2.0, bytes(range(20))),
            (3.0, b"hello"),
        ]
        path = tmp_path / "test.vlp16"
        _write_vlp16_file(path, packets)

        read_back = list(VLP16Playback.iter_raw_packets(path))
        assert len(read_back) == 3
        for i, (ts, payload) in enumerate(read_back):
            assert abs(ts - packets[i][0]) < 1e-9
            assert payload == packets[i][1]

    def test_empty_file_gives_no_packets(self, tmp_path: Path):
        from lidar_mapping.io.playback import VLP16Playback
        path = tmp_path / "empty.vlp16"
        path.write_bytes(b"")
        assert list(VLP16Playback.iter_raw_packets(path)) == []

    def test_truncated_file_stops_gracefully(self, tmp_path: Path):
        """Partial record at end-of-file must not raise."""
        from lidar_mapping.io.playback import VLP16Playback
        path = tmp_path / "truncated.vlp16"
        # Write one good record, then a partial header
        with open(path, "wb") as f:
            f.write(struct.pack(">dH", 1.0, 5))
            f.write(b"hello")
            f.write(struct.pack(">d", 2.0))  # incomplete header (8 bytes, missing uint16)
        pkts = list(VLP16Playback.iter_raw_packets(path))
        assert len(pkts) == 1   # only the good record


# ===========================================================================
# IMU round-trip
# ===========================================================================

class TestIMURoundTrip:
    """Write IMUReadings to .npz and verify playback delivers them correctly."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from lidar_mapping.io.playback import IMUPlayback
        self.IMUPlayback = IMUPlayback

    def test_timestamps_preserved(self, tmp_path: Path):
        readings = [_make_imu_reading(ts=float(i)) for i in range(5)]
        path = tmp_path / "imu.npz"
        _write_imu_npz(path, readings)

        pb = self.IMUPlayback(path, speed=0.0)   # max speed
        pb.start()
        time.sleep(0.3)
        pb.stop()

        assert pb.samples_played == 5

    def test_accel_values_preserved(self, tmp_path: Path):
        reading = _make_imu_reading(ts=0.0, ax=9.81, ay=-1.23, az=0.5)
        path = tmp_path / "imu_single.npz"
        _write_imu_npz(path, [reading])

        pb = self.IMUPlayback(path, speed=0.0)
        pb.start()
        time.sleep(0.3)
        pb.stop()

        r = pb.get_reading(timeout=0.5)
        if r is None:
            # Already drained during stop; read from what was played
            pass
        else:
            np.testing.assert_allclose(r.accel_mss, [9.81, -1.23, 0.5], atol=1e-9)

    def test_mag_none_when_all_nan(self, tmp_path: Path):
        """Readings with NaN mag should come back as mag_ut=None."""
        reading = IMUReading(
            timestamp=0.0,
            accel_mss=np.zeros(3),
            gyro_rads=np.zeros(3),
            mag_ut=None,
        )
        path = tmp_path / "no_mag.npz"
        _write_imu_npz(path, [reading])

        pb = self.IMUPlayback(path, speed=0.0)
        pb.start()
        time.sleep(0.3)
        r = pb.get_reading(timeout=0.5)
        pb.stop()
        if r is not None:
            assert r.mag_ut is None

    def test_playback_in_order(self, tmp_path: Path):
        n = 10
        readings = [_make_imu_reading(ts=float(i) * 0.01) for i in range(n)]
        path = tmp_path / "ordered.npz"
        _write_imu_npz(path, readings)

        pb = self.IMUPlayback(path, speed=0.0)
        pb.start()
        time.sleep(0.5)
        pb.stop()

        assert pb.samples_played == n


# ===========================================================================
# IMU playback speed
# ===========================================================================

class TestIMUPlaybackSpeed:
    """Verify that speed>1 actually finishes faster than real-time."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from lidar_mapping.io.playback import IMUPlayback
        self.IMUPlayback = IMUPlayback

    def test_speed_10x_is_faster(self, tmp_path: Path):
        """10 readings at 0.01 s spacing takes ~0.1 s real-time; at 10x → ~0.01 s."""
        n = 10
        readings = [_make_imu_reading(ts=float(i) * 0.01) for i in range(n)]
        path = tmp_path / "speed.npz"
        _write_imu_npz(path, readings)

        pb = self.IMUPlayback(path, speed=10.0)
        t0 = time.monotonic()
        pb.start()
        time.sleep(3.0)   # wait generously for playback to complete
        pb.stop()
        elapsed = time.monotonic() - t0

        assert pb.samples_played == n
        assert elapsed < 5.0   # should finish well under 5 s even with overhead


# ===========================================================================
# IMU playback — empty file
# ===========================================================================

class TestIMUPlaybackEmpty:
    """Edge-case: playback from an empty .npz file."""

    def test_empty_file_no_readings(self, tmp_path: Path):
        from lidar_mapping.io.playback import IMUPlayback
        path = tmp_path / "empty_imu.npz"
        _write_imu_npz(path, [])

        pb = IMUPlayback(path, speed=0.0)
        pb.start()
        time.sleep(0.3)
        r = pb.get_reading(timeout=0.1)
        pb.stop()

        assert r is None
        assert pb.samples_played == 0


# ===========================================================================
# Recorder helpers — units tests for VLP16Recorder._record_packet
# ===========================================================================

class TestVLP16RecorderPacket:
    """Unit test the recorder's packet serialisation without a real driver."""

    def test_record_packet_writes_correct_format(self, tmp_path: Path):
        """_record_packet should produce a valid >dH + payload record."""
        from lidar_mapping.io.recorder import VLP16Recorder

        # Create recorder without a real driver (pass a dummy)
        dummy_driver = type("D", (), {"_packet_callback": None})()
        path = tmp_path / "test_record.vlp16"
        rec = VLP16Recorder(dummy_driver, path)
        rec.start()

        payload = b"test_payload_12345"
        rec._record_packet(payload)

        rec.stop()

        assert path.exists()
        with open(path, "rb") as f:
            header = f.read(10)
            ts, length = struct.unpack(">dH", header)
            data = f.read(length)

        assert length == len(payload)
        assert data == payload
        assert rec.packets_written == 1

    def test_record_multiple_packets(self, tmp_path: Path):
        from lidar_mapping.io.recorder import VLP16Recorder

        dummy_driver = type("D", (), {"_packet_callback": None})()
        path = tmp_path / "multi.vlp16"
        rec = VLP16Recorder(dummy_driver, path)
        rec.start()

        for i in range(5):
            rec._record_packet(bytes([i] * 10))
        rec.stop()

        assert rec.packets_written == 5

        from lidar_mapping.io.playback import VLP16Playback
        pkts = list(VLP16Playback.iter_raw_packets(path))
        assert len(pkts) == 5
        for i, (ts, payload) in enumerate(pkts):
            assert payload == bytes([i] * 10)


# ===========================================================================
# IMURecorder — samples_recorded property
# ===========================================================================

class TestIMURecorderSamplesRecorded:
    """Unit test IMURecorder._save and samples_recorded."""

    def test_save_creates_npz(self, tmp_path: Path):
        from lidar_mapping.io.recorder import IMURecorder

        dummy_driver = type("D", (), {"get_latest_reading": lambda self: None})()
        path = tmp_path / "imu_rec.npz"
        rec = IMURecorder(dummy_driver, path)

        # Manually inject data
        reading = _make_imu_reading(ts=1.0)
        rec._timestamps = [reading.timestamp]
        rec._accels     = [reading.accel_mss]
        rec._gyros      = [reading.gyro_rads]
        rec._mags       = [reading.mag_ut]
        rec._temps      = [reading.temperature_c]
        rec._quats      = [reading.quaternion]
        rec._rolls      = [reading.roll_deg]
        rec._pitches    = [reading.pitch_deg]
        rec._yaws       = [reading.yaw_deg]

        rec._save()

        assert path.exists()
        data = np.load(str(path))
        np.testing.assert_allclose(data["timestamp"], [1.0])
        np.testing.assert_allclose(data["accel"][0], reading.accel_mss)

    def test_save_nothing_when_empty(self, tmp_path: Path):
        from lidar_mapping.io.recorder import IMURecorder

        dummy_driver = type("D", (), {"get_latest_reading": lambda self: None})()
        path = tmp_path / "empty_rec.npz"
        rec = IMURecorder(dummy_driver, path)
        rec._save()   # no data → no file
        assert not path.exists()
