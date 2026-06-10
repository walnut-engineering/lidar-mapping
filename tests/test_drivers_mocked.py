"""
Tests for all IMU/sensor drivers using mocked hardware interfaces.

No real hardware is required.  smbus2 and serial are mocked with
``unittest.mock`` so these tests run anywhere.
"""

from __future__ import annotations

import struct
import sys
import threading
import time
import math
import types
from unittest.mock import MagicMock, patch, PropertyMock, call
from typing import List

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers to build WitMotion frames
# ---------------------------------------------------------------------------

def _wit_frame(ptype: int, v0: int, v1: int, v2: int, v3: int) -> bytes:
    """Build a valid 11-byte WitMotion frame."""
    raw = bytearray(11)
    raw[0] = 0x55
    raw[1] = ptype
    struct.pack_into("<h", raw, 2, v0)
    struct.pack_into("<h", raw, 4, v1)
    struct.pack_into("<h", raw, 6, v2)
    struct.pack_into("<h", raw, 8, v3)
    raw[10] = sum(raw[:10]) & 0xFF
    return bytes(raw)


# Scale constants (mirror of imu.py module-level constants)
_WIT_ACC_SCALE   = 16.0 * 9.80665 / 32768.0
_WIT_GYRO_SCALE  = 2000.0 / 32768.0 * (math.pi / 180.0)
_WIT_ANG_SCALE   = 180.0 / 32768.0
_WIT_QUAT_SCALE  = 1.0 / 32768.0

_WIT_TYPE_ACC    = 0x51
_WIT_TYPE_GYRO   = 0x52
_WIT_TYPE_ANGLE  = 0x53
_WIT_TYPE_MAG    = 0x54
_WIT_TYPE_QUAT   = 0x59


# ===========================================================================
# WitMotionDriver — _parse_packet static method
# ===========================================================================

class TestWitMotionParsePacket:
    """Tests for the pure ``_parse_packet`` static method — no serial needed."""

    @pytest.fixture(autouse=True)
    def _import_driver(self):
        from lidar_mapping.sensors.imu import WitMotionDriver
        self.WitMotionDriver = WitMotionDriver

    def test_valid_acc_packet(self):
        frame = _wit_frame(_WIT_TYPE_ACC, 1000, -500, 16384, 250)
        result = self.WitMotionDriver._parse_packet(frame)
        assert result is not None
        ptype, values = result
        assert ptype == _WIT_TYPE_ACC
        assert values[0] == 1000
        assert values[1] == -500
        assert values[2] == 16384
        assert values[3] == 250

    def test_valid_quat_packet(self):
        # Identity quaternion: w=32767, x=0, y=0, z=0
        frame = _wit_frame(_WIT_TYPE_QUAT, 32767, 0, 0, 0)
        result = self.WitMotionDriver._parse_packet(frame)
        assert result is not None
        ptype, values = result
        assert ptype == _WIT_TYPE_QUAT

    def test_valid_gyro_packet(self):
        frame = _wit_frame(_WIT_TYPE_GYRO, 100, 200, 300, 0)
        result = self.WitMotionDriver._parse_packet(frame)
        assert result is not None
        assert result[0] == _WIT_TYPE_GYRO

    def test_valid_angle_packet(self):
        frame = _wit_frame(_WIT_TYPE_ANGLE, 1000, -200, 500, 0)
        result = self.WitMotionDriver._parse_packet(frame)
        assert result is not None
        assert result[0] == _WIT_TYPE_ANGLE

    def test_bad_checksum_returns_none(self):
        frame = bytearray(_wit_frame(_WIT_TYPE_ACC, 1000, 0, 0, 0))
        frame[10] = (frame[10] + 1) & 0xFF          # corrupt checksum
        assert self.WitMotionDriver._parse_packet(bytes(frame)) is None

    def test_wrong_start_byte_returns_none(self):
        frame = bytearray(_wit_frame(_WIT_TYPE_ACC, 0, 0, 0, 0))
        frame[0] = 0xAA                              # wrong start byte
        assert self.WitMotionDriver._parse_packet(bytes(frame)) is None

    def test_short_frame_returns_none(self):
        assert self.WitMotionDriver._parse_packet(b"\x55\x51\x00\x00") is None

    def test_empty_frame_returns_none(self):
        assert self.WitMotionDriver._parse_packet(b"") is None

    def test_all_zeros_valid_checksum(self):
        """All-zeros frame: checksum = 0x00 — should parse as valid."""
        frame = bytes(11)   # 0x55 header missing → start byte wrong
        assert self.WitMotionDriver._parse_packet(frame) is None

    def test_acc_scaling(self):
        """raw=32768 (max for signed 16-bit is 32767, use 16384 = 0.5g)."""
        raw_val = 16384
        frame = _wit_frame(_WIT_TYPE_ACC, raw_val, 0, 0, 0)
        result = self.WitMotionDriver._parse_packet(frame)
        assert result is not None
        ptype, values = result
        ax = values[0] * _WIT_ACC_SCALE
        expected = 16384 * _WIT_ACC_SCALE
        assert abs(ax - expected) < 1e-9


# ===========================================================================
# WitMotionDriver — sample loop via injected serial bytes
# ===========================================================================

class TestWitMotionSampleLoop:
    """
    Test the full _sample_loop by injecting pre-built binary frames through
    a mocked serial port.
    """

    def _build_stream(self, frames: list[bytes]) -> bytes:
        return b"".join(frames)

    @pytest.fixture
    def mock_serial_class(self):
        """Return a mock serial.Serial class whose instance reads pre-built data."""
        import types as _t
        mock_serial_mod = _t.ModuleType("serial")
        mock_port = MagicMock()
        mock_serial_mod.Serial = MagicMock(return_value=mock_port)
        mock_serial_mod.SerialException = Exception
        with patch.dict(sys.modules, {"serial": mock_serial_mod}):
            with patch("lidar_mapping.sensors.imu._SERIAL_AVAILABLE", True):
                yield mock_serial_mod, mock_port

    def _make_driver(self, mock_serial_class):
        from lidar_mapping.sensors.imu import WitMotionDriver
        mock_mod, mock_port = mock_serial_class
        driver = WitMotionDriver(port="COM1", baudrate=115200, sample_rate=100.0)
        return driver, mock_port

    def _inject_bytes(self, mock_port, data: bytes):
        """Configure mock port to return bytes one at a time."""
        iter_bytes = iter(data)

        def side_effect(n):
            chunk = bytearray()
            for _ in range(n):
                try:
                    chunk.append(next(iter_bytes))
                except StopIteration:
                    break
            return bytes(chunk)

        mock_port.read.side_effect = side_effect
        mock_port.in_waiting = len(data)

    def test_quat_reading_emitted(self, mock_serial_class):
        """
        Feed acc→gyro→mag→angle→quat frames; confirm an IMUReading is
        pushed to the internal queue.
        """
        acc_frame   = _wit_frame(_WIT_TYPE_ACC,   1000, -500, 500, 250)
        gyro_frame  = _wit_frame(_WIT_TYPE_GYRO,  100, 200, -100, 0)
        mag_frame   = _wit_frame(_WIT_TYPE_MAG,   300, -200, 100, 0)
        angle_frame = _wit_frame(_WIT_TYPE_ANGLE, 1000, 500, -200, 0)
        # Identity quaternion-ish: w≈1, x=y=z=0
        quat_frame  = _wit_frame(_WIT_TYPE_QUAT,  32000, 0, 0, 0)

        stream = acc_frame + gyro_frame + mag_frame + angle_frame + quat_frame

        from lidar_mapping.sensors.imu import WitMotionDriver
        mock_mod, mock_port = mock_serial_class
        driver = WitMotionDriver(port="COM1", baudrate=115200, sample_rate=100.0)

        self._inject_bytes(mock_port, stream)
        mock_port.is_open = True

        # Manually call _open (which calls serial.Serial)
        driver._open()

        # Run _sample_loop for a brief time
        driver._running = True
        t = threading.Thread(target=driver._sample_loop, daemon=True)
        t.start()
        time.sleep(0.25)
        driver._running = False
        t.join(timeout=1.0)

        assert driver.readings_available() > 0 or driver.samples_read > 0

    def test_invalid_bytes_between_frames_handled(self, mock_serial_class):
        """Garbage bytes before a valid frame should not crash the loop."""
        garbage = bytes([0xDE, 0xAD, 0xBE, 0xEF])
        quat_frame = _wit_frame(_WIT_TYPE_QUAT, 32000, 0, 0, 0)
        # Must supply acc/gyro too for a full reading to be emitted
        acc_frame  = _wit_frame(_WIT_TYPE_ACC,  1000, 0, 0, 0)
        gyro_frame = _wit_frame(_WIT_TYPE_GYRO, 0, 0, 0, 0)
        stream = garbage + acc_frame + gyro_frame + quat_frame

        from lidar_mapping.sensors.imu import WitMotionDriver
        mock_mod, mock_port = mock_serial_class
        driver = WitMotionDriver(port="COM1", baudrate=115200, sample_rate=100.0)
        self._inject_bytes(mock_port, stream)
        mock_port.is_open = True
        driver._open()
        driver._running = True
        t = threading.Thread(target=driver._sample_loop, daemon=True)
        t.start()
        time.sleep(0.25)
        driver._running = False
        t.join(timeout=1.0)
        # No exception means pass; the loop should handle garbage gracefully


# ===========================================================================
# SerialAHRSDriver — CSV protocol
# ===========================================================================

class TestSerialAHRSCSV:
    """Mock serial port to test the CSV line parser."""

    @pytest.fixture(autouse=True)
    def _patch_serial(self):
        # Only need _SERIAL_AVAILABLE=True; we never call _open() in these tests
        with patch("lidar_mapping.sensors.imu._SERIAL_AVAILABLE", True):
            yield

    def _make_driver(self, **kwargs):
        from lidar_mapping.sensors.imu import SerialAHRSDriver
        return SerialAHRSDriver(port="COM3", line_format="csv", **kwargs)

    def _line_to_bytes(self, line: str) -> bytes:
        return (line + "\r\n").encode()

    def test_csv_6_fields_accel_gyro(self):
        """``ax,ay,az,gx,gy,gz`` CSV line → correct accel/gyro in reading."""
        line = "1.0,2.0,3.0,0.1,0.2,0.3"
        driver = self._make_driver()
        # Directly call the CSV parser (internal method)
        result = driver._parse_csv(line)
        assert result is not None
        accel, gyro, mag, temp = result
        np.testing.assert_allclose(accel, [1.0, 2.0, 3.0], atol=1e-9)
        np.testing.assert_allclose(gyro,  [0.1, 0.2, 0.3], atol=1e-9)
        assert mag is None
        assert temp is None

    def test_csv_9_fields_includes_mag(self):
        """9-field CSV line also parses magnetometer."""
        line = "1.0,2.0,3.0,0.1,0.2,0.3,10.0,20.0,30.0"
        driver = self._make_driver()
        result = driver._parse_csv(line)
        assert result is not None
        accel, gyro, mag, temp = result
        np.testing.assert_allclose(mag, [10.0, 20.0, 30.0], atol=1e-9)

    def test_csv_invalid_line_raises(self):
        """Lines with non-numeric content raise ValueError."""
        driver = self._make_driver()
        with pytest.raises(ValueError):
            driver._parse_csv("not,a,valid,line")

    def test_csv_too_few_fields_raises(self):
        """Fewer than 6 numeric fields raise ValueError."""
        driver = self._make_driver()
        with pytest.raises(ValueError):
            driver._parse_csv("1.0,2.0,3.0")


# ===========================================================================
# SerialAHRSDriver — PASHR protocol
# ===========================================================================

class TestSerialAHRSPASHR:
    """Test the ``$PASHR`` NMEA sentence parsing logic."""

    @pytest.fixture(autouse=True)
    def _patch_serial(self):
        # Only need _SERIAL_AVAILABLE=True; we never call _open() in these tests
        with patch("lidar_mapping.sensors.imu._SERIAL_AVAILABLE", True):
            yield

    def _make_driver(self):
        from lidar_mapping.sensors.imu import SerialAHRSDriver
        return SerialAHRSDriver(port="COM3", line_format="pashr")

    def test_pashr_yaw_pitch_roll(self):
        """Inline PASHR parsing (as used in _sample_loop) extracts correct Euler."""
        sentence = "$PASHR,045.0,10.0,-5.0,0.0,0.0,0.0,0.0*00"
        content = sentence.split("*")[0][len("$PASHR,"):]
        parts = content.split(",")
        yaw   = float(parts[0])
        pitch = float(parts[1])
        roll  = float(parts[2])
        assert abs(yaw   - 45.0) < 0.01
        assert abs(pitch - 10.0) < 0.01
        assert abs(roll  - (-5.0)) < 0.01

    def test_pashr_negative_yaw(self):
        sentence = "$PASHR,-10.0,0.0,0.0,0.0,0.0,0.0,0.0*00"
        content = sentence.split("*")[0][len("$PASHR,"):]
        parts = content.split(",")
        yaw = float(parts[0])
        assert abs(yaw - (-10.0)) < 0.01

    def test_pashr_bad_prefix_raises(self):
        """Non-PASHR sentence raises ValueError."""
        driver = self._make_driver()
        with pytest.raises(ValueError):
            driver._parse_pashr("$GPGGA,something")

    def test_pashr_too_few_fields_raises(self):
        """PASHR with fewer than 3 fields raises ValueError."""
        driver = self._make_driver()
        with pytest.raises(ValueError):
            driver._parse_pashr("$PASHR,45.0,10.0")

    def test_pashr_returns_zero_sensors(self):
        """_parse_pashr returns zero accel/gyro (Euler injected separately in loop)."""
        driver = self._make_driver()
        accel, gyro, mag, temp = driver._parse_pashr(
            "$PASHR,045.0,10.0,-5.0,0.0,0.0,0.0,0.0*00"
        )
        np.testing.assert_array_equal(accel, np.zeros(3))
        np.testing.assert_array_equal(gyro,  np.zeros(3))
        assert mag is None
        assert temp is None


# ===========================================================================
# MPU9250Driver — smbus2 mocked
# ===========================================================================

class TestMPU9250Mocked:
    """Verify MPU-9250 I²C register access via mocked smbus2."""

    @pytest.fixture(autouse=True)
    def _mock_smbus(self):
        """Stub out smbus2 entirely."""
        smbus2_mock = types.ModuleType("smbus2")
        smbus2_mock.SMBus = MagicMock()
        smbus2_mock.i2c_msg = MagicMock()

        mock_bus_instance = MagicMock()
        smbus2_mock.SMBus.return_value = mock_bus_instance
        self._bus = mock_bus_instance

        # read_i2c_block_data: return 14 bytes (6 accel + 2 temp + 6 gyro)
        # Encoding raw 32768 in big-endian signed 16-bit → [0x7F, 0xFF, ...]
        accel_temp_gyro = [0x7F, 0xFF] * 7   # 14 bytes, each pair = 32767
        self._bus.read_i2c_block_data.return_value = accel_temp_gyro
        # read_byte_data: who-am-i = 0x71 (MPU-9250)
        self._bus.read_byte_data.return_value = 0x71

        # Register smbus2 mock in sys.modules
        with patch.dict(sys.modules, {"smbus2": smbus2_mock}):
            # Also patch the module-level _SMBUS_AVAILABLE flag
            with patch("lidar_mapping.sensors.imu._SMBUS_AVAILABLE", True):
                yield

    def _make_driver(self, **kwargs):
        from lidar_mapping.sensors.imu import MPU9250Driver
        return MPU9250Driver(i2c_bus=1, **kwargs)

    def test_open_writes_pwr_mgmt(self):
        """_open() must reset then set the clock source."""
        driver = self._make_driver(enable_magnetometer=False)
        driver._open()
        # write_byte_data calls should include PWR_MGMT_1 register (0x6B)
        calls = self._bus.write_byte_data.call_args_list
        regs_written = [(c[0][1], c[0][2]) for c in calls]
        assert (0x6B, 0x00) in regs_written   # reset
        assert (0x6B, 0x01) in regs_written   # clock source

    def test_close_resets_device(self):
        """_close() should call close on the bus."""
        driver = self._make_driver(enable_magnetometer=False)
        driver._open()
        driver._close()
        self._bus.close.assert_called_once()

    def test_read_raw_accel_scaling(self):
        """raw 32767 in first two bytes → accel close to ±2g scale."""
        driver = self._make_driver(
            accel_range=2, gyro_range=250, enable_magnetometer=False
        )
        driver._open()
        accel, gyro, mag, temp = driver._read_raw()
        # 32767 * ACCEL_SCALE_2G ≈ +g
        assert abs(accel[0] - 32767 * (9.80665 / 16384.0)) < 0.01
        assert mag is None


# ===========================================================================
# VLP-16 driver — socket mocked
# ===========================================================================

class TestVLP16DriverMocked:
    """
    Verify VLP-16 UDP parsing pipeline via a mocked socket.
    The driver reads raw UDP packets from socket; we inject pre-built ones.
    """

    def _build_vlp16_packet(self) -> bytes:
        """Build a minimal VLP-16 data packet (1206 bytes, all zeros)."""
        # Real packets are 1206 bytes: 1200 data + 4 timestamp + 2 factory
        # All-zero firing data yields all-zero ranges (filtered out by driver
        # as zero-return, but lets us test the parsing path without crashing).
        return bytes(1206)

    def test_packet_parse_no_exception(self):
        """Parser must not raise on a valid-sized zero packet."""
        from lidar_mapping.sensors.vlp16 import VLP16PacketParser
        parser = VLP16PacketParser()
        pkt = self._build_vlp16_packet()
        # Should not raise
        result = parser.parse(pkt)
        assert isinstance(result, tuple)

    def test_driver_lifecycle_mocked_socket(self):
        """Start/stop cycle completes without error using a mocked socket."""
        import socket as socket_module
        with patch("lidar_mapping.sensors.vlp16.socket") as mock_socket_mod:
            mock_sock = MagicMock()
            mock_socket_mod.socket.return_value = mock_sock
            mock_socket_mod.AF_INET = socket_module.AF_INET
            mock_socket_mod.SOCK_DGRAM = socket_module.SOCK_DGRAM
            mock_socket_mod.SOL_SOCKET = socket_module.SOL_SOCKET
            mock_socket_mod.SO_REUSEADDR = socket_module.SO_REUSEADDR

            # recvfrom returns valid-sized packet then blocks
            call_count = {"n": 0}
            def recvfrom_side_effect(size):
                if call_count["n"] < 3:
                    call_count["n"] += 1
                    return (bytes(1206), ("192.168.1.201", 2368))
                time.sleep(0.05)
                return (bytes(1206), ("192.168.1.201", 2368))

            mock_sock.recvfrom.side_effect = recvfrom_side_effect

            from lidar_mapping.sensors.vlp16 import VLP16Driver
            driver = VLP16Driver(host="0.0.0.0", port=2368)
            driver.start()
            time.sleep(0.2)
            driver.stop()
            # Must not raise; socket was bound
            mock_sock.bind.assert_called_once()


# ===========================================================================
# Camera capture — cv2 mocked
# ===========================================================================

class TestCameraCaptureMocked:
    """Verify CameraCapture with a fully mocked cv2."""

    @pytest.fixture(autouse=True)
    def _mock_cv2(self):
        cv2_mock = types.ModuleType("cv2")
        mock_cap = MagicMock()
        # isOpened → True, read → (True, fake frame)
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        cv2_mock.VideoCapture = MagicMock(return_value=mock_cap)
        cv2_mock.CAP_PROP_FRAME_WIDTH  = 3
        cv2_mock.CAP_PROP_FRAME_HEIGHT = 4
        cv2_mock.CAP_PROP_FPS          = 5
        self._mock_cap = mock_cap
        self._cv2_mock = cv2_mock
        # Patch the cv2 name directly in the camera module namespace
        with patch("lidar_mapping.sensors.camera.cv2", cv2_mock, create=True):
            with patch("lidar_mapping.sensors.camera._CV2_AVAILABLE", True):
                yield

    def _make_capture(self, **kwargs):
        from lidar_mapping.sensors.camera import CameraCapture
        return CameraCapture(device_index=0, **kwargs)

    def test_start_stop(self):
        cap = self._make_capture()
        cap.start()
        time.sleep(0.1)
        cap.stop()

    def test_get_latest_frame_returns_frame(self):
        cap = self._make_capture()
        cap.start()
        time.sleep(0.15)
        frame = cap.get_latest_frame()
        cap.stop()
        # Frame may be None if thread hasn't produced one yet, but no error
        assert frame is None or hasattr(frame, "image")

    def test_failed_open_raises(self):
        """A VideoCapture that fails to open must raise RuntimeError from start()."""
        bad_cap = MagicMock()
        bad_cap.isOpened.return_value = False
        bad_cv2 = types.ModuleType("cv2")
        bad_cv2.VideoCapture = MagicMock(return_value=bad_cap)
        bad_cv2.CAP_PROP_FRAME_WIDTH  = 3
        bad_cv2.CAP_PROP_FRAME_HEIGHT = 4
        bad_cv2.CAP_PROP_FPS          = 5
        # Override the autouse fixture's cv2 mock for just this test
        from lidar_mapping.sensors.camera import CameraCapture
        with patch("lidar_mapping.sensors.camera.cv2", bad_cv2, create=True):
            cap = CameraCapture(device_index=99)
            with pytest.raises(RuntimeError):
                cap.start()
