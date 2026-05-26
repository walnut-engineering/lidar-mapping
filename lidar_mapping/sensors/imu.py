"""
IMU sensor driver.

Provides a unified, threaded interface for reading **accelerometer**,
**gyroscope**, and **magnetometer** data from common IMU hardware.

Supported backends
------------------
I²C (Raspberry Pi / Linux)
    - **MPU-9250** / MPU-6500 — 6-axis IMU (gyro + accel) with AK8963 mag
    - **ICM-42688-P** — 6-axis high-performance IMU (gyro + accel)
    - **LSM9DS1** — 9-axis IMU (accel + gyro + mag in one package)
    - **BNO055** — 9-axis AHRS with on-chip sensor fusion

Serial / USB AHRS (Windows and Linux)
    - Any device that outputs ``$PASHR`` NMEA sentences or a simple
      ``CSV: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z [, mag_x, mag_y, mag_z]``
      at configurable baud rate.

All drivers run a background thread and expose a common
:meth:`~BaseIMUDriver.get_reading` interface that returns an
:class:`IMUReading` dataclass.  Onboard AHRS fusion (Madgwick or Mahony)
is applied automatically to raw-data devices.

Usage (I²C, RPi)::

    from lidar_mapping.sensors.imu import MPU9250Driver

    imu = MPU9250Driver(i2c_bus=1, address=0x68, sample_rate=100)
    imu.start()
    try:
        reading = imu.get_reading(timeout=1.0)
        print(f"Roll={reading.roll_deg:.1f}°  "
              f"Pitch={reading.pitch_deg:.1f}°  "
              f"Yaw={reading.yaw_deg:.1f}°")
    finally:
        imu.stop()

Usage (USB serial AHRS)::

    from lidar_mapping.sensors.imu import SerialAHRSDriver

    imu = SerialAHRSDriver(port="COM3", baudrate=115200)
    imu.start()
    reading = imu.get_reading(timeout=1.0)
"""

from __future__ import annotations

import abc
import logging
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from lidar_mapping.sensors.ahrs import MadgwickAHRS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    import smbus2  # type: ignore

    _SMBUS_AVAILABLE = True
except ImportError:
    _SMBUS_AVAILABLE = False

try:
    import serial  # pyserial

    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False


# ---------------------------------------------------------------------------
# IMU reading dataclass
# ---------------------------------------------------------------------------

@dataclass
class IMUReading:
    """
    A single IMU sample with fused orientation.

    Attributes
    ----------
    timestamp:
        Monotonic host timestamp in seconds (``time.monotonic()``).
    accel_mss:
        Raw accelerometer output in m/s² — ``[ax, ay, az]``.
    gyro_rads:
        Raw gyroscope output in rad/s — ``[gx, gy, gz]``.
    mag_ut:
        Magnetometer output in µT — ``[mx, my, mz]``, or ``None`` if the
        device has no magnetometer.
    temperature_c:
        Die temperature in °C, or ``None`` if not available.
    roll_deg:
        Fused roll angle (rotation about X) in degrees.
    pitch_deg:
        Fused pitch angle (rotation about Y) in degrees.
    yaw_deg:
        Fused yaw angle (rotation about Z / compass heading) in degrees.
        Without a magnetometer this will drift over time.
    quaternion:
        Fused orientation as a ``(w, x, y, z)`` unit quaternion.
    """

    timestamp: float
    accel_mss: np.ndarray
    gyro_rads: np.ndarray
    mag_ut: Optional[np.ndarray] = None
    temperature_c: Optional[float] = None
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    quaternion: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0])
    )


# ---------------------------------------------------------------------------
# Abstract base driver
# ---------------------------------------------------------------------------

class BaseIMUDriver(abc.ABC):
    """
    Abstract base class for all IMU drivers.

    Subclasses implement :meth:`_read_raw` to return raw sensor measurements.
    The base class wraps this in a background thread, runs an AHRS filter,
    and buffers :class:`IMUReading` objects for consumer threads.

    Parameters
    ----------
    sample_rate:
        Target sample rate in Hz.
    filter_type:
        ``"madgwick"`` (default) or ``"mahony"`` for the AHRS fusion filter.
    filter_beta:
        Madgwick beta gain (ignored for Mahony).
    max_queue:
        Maximum number of readings to buffer before dropping oldest.
    """

    def __init__(
        self,
        sample_rate: float = 100.0,
        filter_type: str = "madgwick",
        filter_beta: float = 0.033,
        max_queue: int = 200,
    ) -> None:
        self._sample_rate = sample_rate
        self._max_queue = max_queue
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._readings: list[IMUReading] = []

        # AHRS filter
        self._ahrs = MadgwickAHRS(
            sample_rate=sample_rate, beta=filter_beta
        )
        if filter_type not in ("madgwick", "mahony"):
            raise ValueError(
                f"filter_type must be 'madgwick' or 'mahony', got {filter_type!r}"
            )
        if filter_type == "mahony":
            from lidar_mapping.sensors.ahrs import MahonyAHRS
            self._ahrs = MahonyAHRS(sample_rate=sample_rate)  # type: ignore[assignment]

        self.samples_read: int = 0
        self.read_errors: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the hardware interface and start the sampling thread."""
        if self._running:
            return
        self._open()
        self._running = True
        self._thread = threading.Thread(
            target=self._sample_loop, daemon=True, name="imu-sample"
        )
        self._thread.start()
        logger.info("%s started at %.0f Hz.", type(self).__name__, self._sample_rate)

    def stop(self) -> None:
        """Stop the sampling thread and close the hardware interface."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._close()
        logger.info("%s stopped.", type(self).__name__)

    # ------------------------------------------------------------------
    # Reading access
    # ------------------------------------------------------------------

    def get_reading(self, timeout: float = 1.0) -> Optional[IMUReading]:
        """
        Block until a reading is available, then return it.

        Parameters
        ----------
        timeout:
            Maximum seconds to wait.

        Returns
        -------
        :class:`IMUReading` or ``None`` if the timeout elapsed.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._readings:
                    return self._readings.pop(0)
            time.sleep(1.0 / self._sample_rate / 2.0)
        return None

    def get_latest_reading(self) -> Optional[IMUReading]:
        """Return the most recent reading without blocking, or ``None``."""
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
    # Abstract interface for subclasses
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _open(self) -> None:
        """Open the hardware connection.  Called once before the loop."""

    @abc.abstractmethod
    def _close(self) -> None:
        """Close the hardware connection."""

    @abc.abstractmethod
    def _read_raw(self) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[float]]:
        """
        Read one sample from the hardware.

        Returns
        -------
        accel_mss:
            (3,) float64 in m/s².
        gyro_rads:
            (3,) float64 in rad/s.
        mag_ut:
            (3,) float64 in µT, or ``None``.
        temperature_c:
            Float in °C, or ``None``.
        """

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sample_loop(self) -> None:
        period = 1.0 / self._sample_rate
        while self._running:
            t_start = time.monotonic()
            try:
                accel, gyro, mag, temp = self._read_raw()
            except Exception as exc:  # noqa: BLE001
                self.read_errors += 1
                logger.debug("IMU read error: %s", exc)
                time.sleep(period)
                continue

            # Run AHRS filter
            if mag is not None:
                self._ahrs.update(gyro, accel, mag)
            else:
                self._ahrs.update_imu(gyro, accel)

            roll, pitch, yaw = self._ahrs.euler_degrees
            reading = IMUReading(
                timestamp=t_start,
                accel_mss=accel.copy(),
                gyro_rads=gyro.copy(),
                mag_ut=mag.copy() if mag is not None else None,
                temperature_c=temp,
                roll_deg=roll,
                pitch_deg=pitch,
                yaw_deg=yaw,
                quaternion=self._ahrs.quaternion,
            )

            with self._lock:
                self._readings.append(reading)
                if len(self._readings) > self._max_queue:
                    self._readings.pop(0)

            self.samples_read += 1

            # Sleep for the remainder of the period
            elapsed = time.monotonic() - t_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


# ---------------------------------------------------------------------------
# MPU-9250 / MPU-6500 I²C driver
# ---------------------------------------------------------------------------

# Register addresses (MPU-9250)
_MPU_ADDR_DEFAULT = 0x68
_MPU_REG_PWR_MGMT_1 = 0x6B
_MPU_REG_SMPLRT_DIV = 0x19
_MPU_REG_CONFIG = 0x1A
_MPU_REG_GYRO_CONFIG = 0x1B
_MPU_REG_ACCEL_CONFIG = 0x1C
_MPU_REG_ACCEL_CONFIG2 = 0x1D
_MPU_REG_INT_PIN_CFG = 0x37
_MPU_REG_INT_ENABLE = 0x38
_MPU_REG_ACCEL_XOUT_H = 0x3B
_MPU_REG_TEMP_OUT_H = 0x41
_MPU_REG_GYRO_XOUT_H = 0x43
_MPU_REG_USER_CTRL = 0x6A
_MPU_REG_WHO_AM_I = 0x75

# AK8963 magnetometer (connected as MPU aux I²C slave)
_AK8963_ADDR = 0x0C
_AK8963_REG_WIA = 0x00    # Who Am I
_AK8963_REG_ST1 = 0x02    # Status 1 (data ready)
_AK8963_REG_XOUT_L = 0x03
_AK8963_REG_ST2 = 0x09    # Status 2 (overflow flag)
_AK8963_REG_CNTL1 = 0x0A  # Control 1 (mode, resolution)
_AK8963_REG_ASAX = 0x10   # Sensitivity adjustment X

# Scale factors
_ACCEL_SCALE_2G = 9.80665 / 16384.0    # LSB → m/s² at ±2 g
_GYRO_SCALE_250 = (250.0 / 32768.0) * (3.14159265 / 180.0)  # LSB → rad/s at ±250°/s
_MAG_SCALE_16BIT = 4912.0 / 32760.0    # LSB → µT at 16-bit mode


class MPU9250Driver(BaseIMUDriver):
    """
    I²C driver for the InvenSense MPU-9250 (or MPU-6500) IMU.

    The MPU-9250 contains a 3-axis gyroscope, 3-axis accelerometer and a
    3-axis AK8963 magnetometer on the auxiliary I²C bus.

    Requires the ``smbus2`` package::

        pip install smbus2

    Parameters
    ----------
    i2c_bus:
        I²C bus number (typically ``1`` on Raspberry Pi).
    address:
        MPU-9250 I²C address (``0x68`` when AD0=low, ``0x69`` when AD0=high).
    sample_rate:
        Target sample rate in Hz (1–1000).  The actual rate depends on the
        MPU-9250's sample-rate divider register.
    accel_range:
        Accelerometer full-scale range: 2, 4, 8, or 16 (g).
    gyro_range:
        Gyroscope full-scale range: 250, 500, 1000, or 2000 (°/s).
    enable_magnetometer:
        If ``True`` (default), initialise the AK8963 magnetometer.
    """

    _ACCEL_SCALES = {
        2: (0x00, 9.80665 / 16384.0),
        4: (0x08, 9.80665 / 8192.0),
        8: (0x10, 9.80665 / 4096.0),
        16: (0x18, 9.80665 / 2048.0),
    }
    _GYRO_SCALES = {
        250:  (0x00, (250.0 / 32768.0) * (3.14159265 / 180.0)),
        500:  (0x08, (500.0 / 32768.0) * (3.14159265 / 180.0)),
        1000: (0x10, (1000.0 / 32768.0) * (3.14159265 / 180.0)),
        2000: (0x18, (2000.0 / 32768.0) * (3.14159265 / 180.0)),
    }

    def __init__(
        self,
        i2c_bus: int = 1,
        address: int = _MPU_ADDR_DEFAULT,
        sample_rate: float = 100.0,
        accel_range: int = 2,
        gyro_range: int = 250,
        enable_magnetometer: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)
        if not _SMBUS_AVAILABLE:
            raise ImportError(
                "smbus2 is required for I²C IMU drivers. "
                "Install it with: pip install smbus2"
            )
        if accel_range not in self._ACCEL_SCALES:
            raise ValueError(f"accel_range must be one of {list(self._ACCEL_SCALES)}")
        if gyro_range not in self._GYRO_SCALES:
            raise ValueError(f"gyro_range must be one of {list(self._GYRO_SCALES)}")

        self._bus_num = i2c_bus
        self._address = address
        self._enable_mag = enable_magnetometer
        self._accel_reg, self._accel_scale = self._ACCEL_SCALES[accel_range]
        self._gyro_reg, self._gyro_scale = self._GYRO_SCALES[gyro_range]
        self._mag_asa = np.ones(3, dtype=np.float64)  # sensitivity adjustment
        self._bus: Optional[smbus2.SMBus] = None  # type: ignore[name-defined]

    # ------------------------------------------------------------------
    # BaseIMUDriver interface
    # ------------------------------------------------------------------

    def _open(self) -> None:
        import smbus2  # re-import to satisfy type checker in test environments

        self._bus = smbus2.SMBus(self._bus_num)
        time.sleep(0.05)

        # Wake the MPU-9250
        self._bus.write_byte_data(self._address, _MPU_REG_PWR_MGMT_1, 0x00)
        time.sleep(0.1)
        # Use PLL clock
        self._bus.write_byte_data(self._address, _MPU_REG_PWR_MGMT_1, 0x01)
        time.sleep(0.05)

        # Sample rate divider: target rate = 1000 / (1 + divider)
        divider = max(0, int(1000.0 / self._sample_rate) - 1)
        self._bus.write_byte_data(self._address, _MPU_REG_SMPLRT_DIV, divider)

        # DLPF (Digital Low Pass Filter) — 44 Hz accel / 42 Hz gyro
        self._bus.write_byte_data(self._address, _MPU_REG_CONFIG, 0x03)

        # Gyro full scale
        self._bus.write_byte_data(self._address, _MPU_REG_GYRO_CONFIG, self._gyro_reg)

        # Accel full scale
        self._bus.write_byte_data(self._address, _MPU_REG_ACCEL_CONFIG, self._accel_reg)
        self._bus.write_byte_data(self._address, _MPU_REG_ACCEL_CONFIG2, 0x03)

        if self._enable_mag:
            self._init_ak8963()

        logger.debug("MPU-9250 at 0x%02X initialised.", self._address)

    def _close(self) -> None:
        if self._bus is not None:
            self._bus.close()
            self._bus = None

    def _read_raw(
        self,
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[float]]:
        assert self._bus is not None

        # Read 14 bytes: accel(6) + temp(2) + gyro(6)
        data = self._bus.read_i2c_block_data(
            self._address, _MPU_REG_ACCEL_XOUT_H, 14
        )
        ax_raw = struct.unpack(">h", bytes(data[0:2]))[0]
        ay_raw = struct.unpack(">h", bytes(data[2:4]))[0]
        az_raw = struct.unpack(">h", bytes(data[4:6]))[0]
        t_raw = struct.unpack(">h", bytes(data[6:8]))[0]
        gx_raw = struct.unpack(">h", bytes(data[8:10]))[0]
        gy_raw = struct.unpack(">h", bytes(data[10:12]))[0]
        gz_raw = struct.unpack(">h", bytes(data[12:14]))[0]

        accel = np.array(
            [ax_raw, ay_raw, az_raw], dtype=np.float64
        ) * self._accel_scale
        gyro = np.array(
            [gx_raw, gy_raw, gz_raw], dtype=np.float64
        ) * self._gyro_scale
        temp_c = t_raw / 333.87 + 21.0

        mag: Optional[np.ndarray] = None
        if self._enable_mag:
            mag = self._read_ak8963()

        return accel, gyro, mag, temp_c

    # ------------------------------------------------------------------
    # AK8963 (magnetometer) helpers
    # ------------------------------------------------------------------

    def _init_ak8963(self) -> None:
        assert self._bus is not None

        # Enable I²C bypass so we can talk to AK8963 directly
        self._bus.write_byte_data(self._address, _MPU_REG_INT_PIN_CFG, 0x02)
        time.sleep(0.01)

        # Power down, then enter Fuse ROM access mode to read sensitivity
        self._bus.write_byte_data(_AK8963_ADDR, _AK8963_REG_CNTL1, 0x00)
        time.sleep(0.01)
        self._bus.write_byte_data(_AK8963_ADDR, _AK8963_REG_CNTL1, 0x0F)
        time.sleep(0.01)

        asa = self._bus.read_i2c_block_data(_AK8963_ADDR, _AK8963_REG_ASAX, 3)
        self._mag_asa = np.array(
            [(a - 128.0) / 256.0 + 1.0 for a in asa], dtype=np.float64
        )

        # Power down again, then switch to continuous measurement mode 2 (100 Hz, 16-bit)
        self._bus.write_byte_data(_AK8963_ADDR, _AK8963_REG_CNTL1, 0x00)
        time.sleep(0.01)
        self._bus.write_byte_data(_AK8963_ADDR, _AK8963_REG_CNTL1, 0x16)
        time.sleep(0.01)
        logger.debug("AK8963 magnetometer initialised; ASA=%s", self._mag_asa)

    def _read_ak8963(self) -> Optional[np.ndarray]:
        assert self._bus is not None
        try:
            st1 = self._bus.read_byte_data(_AK8963_ADDR, _AK8963_REG_ST1)
            if not (st1 & 0x01):
                return None  # data not ready

            data = self._bus.read_i2c_block_data(_AK8963_ADDR, _AK8963_REG_XOUT_L, 7)
            if data[6] & 0x08:
                return None  # overflow

            mx_raw = struct.unpack("<h", bytes(data[0:2]))[0]
            my_raw = struct.unpack("<h", bytes(data[2:4]))[0]
            mz_raw = struct.unpack("<h", bytes(data[4:6]))[0]

            mag = np.array(
                [mx_raw, my_raw, mz_raw], dtype=np.float64
            ) * _MAG_SCALE_16BIT * self._mag_asa
            return mag
        except OSError:
            return None


# ---------------------------------------------------------------------------
# BNO055 I²C driver  (onboard Bosch sensor fusion)
# ---------------------------------------------------------------------------

_BNO055_ADDR_DEFAULT = 0x28
_BNO055_REG_OPR_MODE = 0x3D
_BNO055_REG_PWR_MODE = 0x3E
_BNO055_REG_SYS_TRIGGER = 0x3F
_BNO055_REG_UNIT_SEL = 0x3B
_BNO055_REG_ACC_DATA_X_LSB = 0x08
_BNO055_REG_GYR_DATA_X_LSB = 0x14
_BNO055_REG_MAG_DATA_X_LSB = 0x0E
_BNO055_REG_EUL_HEADING_LSB = 0x1A
_BNO055_REG_QUA_DATA_W_LSB = 0x20
_BNO055_REG_TEMP = 0x34
_BNO055_OPR_MODE_NDOF = 0x0C  # Nine Degrees of Freedom (full fusion)
_BNO055_OPR_MODE_AMG = 0x07   # Accel + Mag + Gyro, no fusion


class BNO055Driver(BaseIMUDriver):
    """
    I²C driver for the Bosch BNO055 9-axis AHRS.

    The BNO055 performs its own onboard sensor fusion and outputs fused Euler
    angles / quaternions directly, so the software AHRS filter is bypassed.
    This is the simplest and most accurate option for beginners.

    Requires the ``smbus2`` package::

        pip install smbus2

    Parameters
    ----------
    i2c_bus:
        I²C bus number (``1`` on Raspberry Pi).
    address:
        BNO055 I²C address (``0x28`` when ADR=low, ``0x29`` when ADR=high).
    sample_rate:
        Polling rate in Hz.  The BNO055 outputs fused data at up to 100 Hz.
    use_onboard_fusion:
        If ``True`` (default), use the BNO055's on-chip NDOF fusion mode
        and read quaternion output directly.  If ``False``, run in raw
        AMG mode and apply the software Madgwick filter.
    """

    def __init__(
        self,
        i2c_bus: int = 1,
        address: int = _BNO055_ADDR_DEFAULT,
        sample_rate: float = 100.0,
        use_onboard_fusion: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)
        if not _SMBUS_AVAILABLE:
            raise ImportError(
                "smbus2 is required for I²C IMU drivers. "
                "Install it with: pip install smbus2"
            )
        self._bus_num = i2c_bus
        self._address = address
        self._use_onboard_fusion = use_onboard_fusion
        self._bus = None

    def _open(self) -> None:
        import smbus2

        self._bus = smbus2.SMBus(self._bus_num)
        time.sleep(0.05)

        # Switch to CONFIG mode
        self._bus.write_byte_data(self._address, _BNO055_REG_OPR_MODE, 0x00)
        time.sleep(0.02)

        # Set units: m/s², rad/s, degrees, Celsius
        self._bus.write_byte_data(self._address, _BNO055_REG_UNIT_SEL, 0x00)
        time.sleep(0.01)

        # Normal power mode
        self._bus.write_byte_data(self._address, _BNO055_REG_PWR_MODE, 0x00)
        time.sleep(0.01)

        # Use external oscillator for better accuracy
        self._bus.write_byte_data(self._address, _BNO055_REG_SYS_TRIGGER, 0x00)
        time.sleep(0.01)

        mode = _BNO055_OPR_MODE_NDOF if self._use_onboard_fusion else _BNO055_OPR_MODE_AMG
        self._bus.write_byte_data(self._address, _BNO055_REG_OPR_MODE, mode)
        time.sleep(0.02)

        logger.debug(
            "BNO055 at 0x%02X initialised (fusion=%s).",
            self._address, self._use_onboard_fusion,
        )

    def _close(self) -> None:
        if self._bus is not None:
            self._bus.close()
            self._bus = None

    def _read_raw(
        self,
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[float]]:
        assert self._bus is not None

        # Accelerometer (6 bytes, LSB/MSB pairs, 1 m/s² = 100 LSB)
        acc_data = self._bus.read_i2c_block_data(
            self._address, _BNO055_REG_ACC_DATA_X_LSB, 6
        )
        accel = np.array(
            [
                struct.unpack("<h", bytes(acc_data[0:2]))[0] / 100.0,
                struct.unpack("<h", bytes(acc_data[2:4]))[0] / 100.0,
                struct.unpack("<h", bytes(acc_data[4:6]))[0] / 100.0,
            ],
            dtype=np.float64,
        )

        # Gyroscope (6 bytes, 1 rad/s = 900 LSB)
        gyr_data = self._bus.read_i2c_block_data(
            self._address, _BNO055_REG_GYR_DATA_X_LSB, 6
        )
        gyro = np.array(
            [
                struct.unpack("<h", bytes(gyr_data[0:2]))[0] / 900.0,
                struct.unpack("<h", bytes(gyr_data[2:4]))[0] / 900.0,
                struct.unpack("<h", bytes(gyr_data[4:6]))[0] / 900.0,
            ],
            dtype=np.float64,
        )

        # Magnetometer (6 bytes, 1 µT = 16 LSB)
        mag_data = self._bus.read_i2c_block_data(
            self._address, _BNO055_REG_MAG_DATA_X_LSB, 6
        )
        mag = np.array(
            [
                struct.unpack("<h", bytes(mag_data[0:2]))[0] / 16.0,
                struct.unpack("<h", bytes(mag_data[2:4]))[0] / 16.0,
                struct.unpack("<h", bytes(mag_data[4:6]))[0] / 16.0,
            ],
            dtype=np.float64,
        )

        # Temperature
        temp_c = float(
            struct.unpack("b", bytes([self._bus.read_byte_data(
                self._address, _BNO055_REG_TEMP
            )]))[0]
        )

        return accel, gyro, mag, temp_c

    # ------------------------------------------------------------------
    # Override the sample loop to read fused output directly if available
    # ------------------------------------------------------------------

    def _sample_loop(self) -> None:
        if self._use_onboard_fusion:
            self._sample_loop_fused()
        else:
            super()._sample_loop()

    def _sample_loop_fused(self) -> None:
        """Read fused quaternion + raw data directly from the BNO055."""
        assert self._bus is not None
        period = 1.0 / self._sample_rate

        while self._running:
            t_start = time.monotonic()
            try:
                accel, gyro, mag, temp_c = self._read_raw()

                # Fused quaternion (4 × int16, 1 unit = 16384 LSB)
                qdata = self._bus.read_i2c_block_data(
                    self._address, _BNO055_REG_QUA_DATA_W_LSB, 8
                )
                qw = struct.unpack("<h", bytes(qdata[0:2]))[0] / 16384.0
                qx = struct.unpack("<h", bytes(qdata[2:4]))[0] / 16384.0
                qy = struct.unpack("<h", bytes(qdata[4:6]))[0] / 16384.0
                qz = struct.unpack("<h", bytes(qdata[6:8]))[0] / 16384.0
                quaternion = np.array([qw, qx, qy, qz], dtype=np.float64)

                from lidar_mapping.sensors.ahrs import _quaternion_to_euler_degrees
                roll, pitch, yaw = _quaternion_to_euler_degrees(quaternion)

            except Exception as exc:  # noqa: BLE001
                self.read_errors += 1
                logger.debug("BNO055 read error: %s", exc)
                time.sleep(period)
                continue

            reading = IMUReading(
                timestamp=t_start,
                accel_mss=accel,
                gyro_rads=gyro,
                mag_ut=mag,
                temperature_c=temp_c,
                roll_deg=roll,
                pitch_deg=pitch,
                yaw_deg=yaw,
                quaternion=quaternion,
            )

            with self._lock:
                self._readings.append(reading)
                if len(self._readings) > self._max_queue:
                    self._readings.pop(0)

            self.samples_read += 1
            elapsed = time.monotonic() - t_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


# ---------------------------------------------------------------------------
# LSM9DS1 I²C driver
# ---------------------------------------------------------------------------

_LSM_AG_ADDR_DEFAULT = 0x6B   # Accel/Gyro (SA0=high)
_LSM_MAG_ADDR_DEFAULT = 0x1E  # Magnetometer (SDO_M=low)
_LSM_REG_CTRL_REG1_G = 0x10
_LSM_REG_CTRL_REG6_XL = 0x20
_LSM_REG_CTRL_REG1_M = 0x20
_LSM_REG_CTRL_REG2_M = 0x21
_LSM_REG_CTRL_REG3_M = 0x22
_LSM_REG_CTRL_REG4_M = 0x23
_LSM_REG_OUT_X_L_G = 0x18
_LSM_REG_OUT_X_L_XL = 0x28
_LSM_REG_OUT_X_L_M = 0x28
_LSM_REG_OUT_TEMP_L = 0x15


class LSM9DS1Driver(BaseIMUDriver):
    """
    I²C driver for the STMicroelectronics LSM9DS1 9-axis IMU.

    The LSM9DS1 provides a 3-axis gyroscope, 3-axis accelerometer, and
    3-axis magnetometer, each with its own I²C address.

    Requires the ``smbus2`` package::

        pip install smbus2

    Parameters
    ----------
    i2c_bus:
        I²C bus number.
    ag_address:
        Accelerometer/Gyro I²C address.
    mag_address:
        Magnetometer I²C address.
    sample_rate:
        Target sample rate in Hz.
    """

    def __init__(
        self,
        i2c_bus: int = 1,
        ag_address: int = _LSM_AG_ADDR_DEFAULT,
        mag_address: int = _LSM_MAG_ADDR_DEFAULT,
        sample_rate: float = 100.0,
        **kwargs,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)
        if not _SMBUS_AVAILABLE:
            raise ImportError(
                "smbus2 is required for I²C IMU drivers. "
                "Install it with: pip install smbus2"
            )
        self._bus_num = i2c_bus
        self._ag_addr = ag_address
        self._mag_addr = mag_address
        self._bus = None

    def _open(self) -> None:
        import smbus2

        self._bus = smbus2.SMBus(self._bus_num)
        time.sleep(0.05)

        # Gyro: 119 Hz ODR, 500 dps, default DLPF
        self._bus.write_byte_data(self._ag_addr, _LSM_REG_CTRL_REG1_G, 0x68)
        # Accel: 119 Hz ODR, ±2g, default DLPF
        self._bus.write_byte_data(self._ag_addr, _LSM_REG_CTRL_REG6_XL, 0x60)

        # Mag: Temp compensated, low power XY, 10 Hz ODR
        self._bus.write_byte_data(self._mag_addr, _LSM_REG_CTRL_REG1_M, 0x10)
        # ±4 gauss
        self._bus.write_byte_data(self._mag_addr, _LSM_REG_CTRL_REG2_M, 0x00)
        # Continuous conversion
        self._bus.write_byte_data(self._mag_addr, _LSM_REG_CTRL_REG3_M, 0x00)
        # Z-axis high performance
        self._bus.write_byte_data(self._mag_addr, _LSM_REG_CTRL_REG4_M, 0x0C)

        time.sleep(0.1)
        logger.debug("LSM9DS1 initialised (AG=0x%02X, MAG=0x%02X).",
                     self._ag_addr, self._mag_addr)

    def _close(self) -> None:
        if self._bus is not None:
            self._bus.close()
            self._bus = None

    def _read_raw(
        self,
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[float]]:
        assert self._bus is not None

        # Read gyro (6 bytes, ±500 dps → 17.5 mdps/LSB)
        gdata = self._bus.read_i2c_block_data(
            self._ag_addr, _LSM_REG_OUT_X_L_G | 0x80, 6
        )
        gyro_scale = (500.0 / 32768.0) * (3.14159265 / 180.0)
        gyro = np.array(
            [
                struct.unpack("<h", bytes(gdata[0:2]))[0],
                struct.unpack("<h", bytes(gdata[2:4]))[0],
                struct.unpack("<h", bytes(gdata[4:6]))[0],
            ],
            dtype=np.float64,
        ) * gyro_scale

        # Read accel (6 bytes, ±2g → 0.061 mg/LSB)
        adata = self._bus.read_i2c_block_data(
            self._ag_addr, _LSM_REG_OUT_X_L_XL | 0x80, 6
        )
        accel_scale = (2.0 / 32768.0) * 9.80665
        accel = np.array(
            [
                struct.unpack("<h", bytes(adata[0:2]))[0],
                struct.unpack("<h", bytes(adata[2:4]))[0],
                struct.unpack("<h", bytes(adata[4:6]))[0],
            ],
            dtype=np.float64,
        ) * accel_scale

        # Read magnetometer (6 bytes, ±4 gauss → 0.14 mgauss/LSB → µT)
        mdata = self._bus.read_i2c_block_data(
            self._mag_addr, _LSM_REG_OUT_X_L_M | 0x80, 6
        )
        mag_scale = (4.0 * 100.0) / 32768.0  # gauss → µT: 1 gauss = 100 µT
        mag = np.array(
            [
                struct.unpack("<h", bytes(mdata[0:2]))[0],
                struct.unpack("<h", bytes(mdata[2:4]))[0],
                struct.unpack("<h", bytes(mdata[4:6]))[0],
            ],
            dtype=np.float64,
        ) * mag_scale

        # Temperature (2 bytes, 16 LSB/°C, offset 25°C)
        tdata = self._bus.read_i2c_block_data(
            self._ag_addr, _LSM_REG_OUT_TEMP_L | 0x80, 2
        )
        temp_c = struct.unpack("<h", bytes(tdata))[0] / 16.0 + 25.0

        return accel, gyro, mag, temp_c


# ---------------------------------------------------------------------------
# Serial / USB AHRS driver  (cross-platform: Windows + Linux)
# ---------------------------------------------------------------------------

class SerialAHRSDriver(BaseIMUDriver):
    """
    Serial / USB driver for AHRS modules and custom sensor boards.

    Supports two line formats:

    ``PASHR``
        Standard NMEA-like sentence emitted by many commercial AHRS units::

            $PASHR,HHH.HH,PP.PP,TT.TT,RRR.RR,XX.XX,YY.YY,ZZ.ZZ,Q*CS

        Where fields after ``PASHR,`` are heading, pitch, roll, heave,
        roll-rate, pitch-rate, heading-rate, quality.  Only heading/pitch/roll
        are used here.

    ``CSV``
        Comma-separated raw sensor values, one line per sample::

            ax,ay,az,gx,gy,gz[,mx,my,mz]

        Units: m/s² for accel, rad/s for gyro, µT for mag.

    Parameters
    ----------
    port:
        Serial port name (e.g. ``"COM3"`` on Windows, ``"/dev/ttyUSB0"`` on Linux).
    baudrate:
        Serial baud rate (default 115200).
    line_format:
        ``"pashr"`` or ``"csv"`` (default).
    sample_rate:
        Nominal sample rate — used only for the AHRS filter ``dt``.
    timeout:
        Serial read timeout in seconds.
    """

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 115200,
        line_format: str = "csv",
        sample_rate: float = 100.0,
        timeout: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)
        if not _SERIAL_AVAILABLE:
            raise ImportError(
                "pyserial is required for SerialAHRSDriver. "
                "Install it with: pip install pyserial"
            )
        if line_format not in ("pashr", "csv"):
            raise ValueError(
                f"line_format must be 'pashr' or 'csv', got {line_format!r}"
            )
        self._port = port
        self._baudrate = baudrate
        self._format = line_format
        self._timeout = timeout
        self._serial = None

    def _open(self) -> None:
        import serial as pyserial

        self._serial = pyserial.Serial(
            self._port,
            baudrate=self._baudrate,
            timeout=self._timeout,
        )
        logger.debug("Serial AHRS opened on %s @ %d baud.", self._port, self._baudrate)

    def _close(self) -> None:
        if self._serial is not None and self._serial.is_open:
            self._serial.close()
        self._serial = None

    def _read_raw(
        self,
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[float]]:
        assert self._serial is not None

        line = self._serial.readline().decode("ascii", errors="ignore").strip()

        if self._format == "pashr":
            return self._parse_pashr(line)
        else:
            return self._parse_csv(line)

    # ------------------------------------------------------------------
    # Parser helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_csv(
        line: str,
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[float]]:
        """
        Parse a CSV line: ``ax,ay,az,gx,gy,gz[,mx,my,mz]``

        Expected units: m/s², rad/s, µT.
        """
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            raise ValueError(f"CSV line too short: {line!r}")

        accel = np.array([float(parts[0]), float(parts[1]), float(parts[2])],
                         dtype=np.float64)
        gyro = np.array([float(parts[3]), float(parts[4]), float(parts[5])],
                        dtype=np.float64)
        mag: Optional[np.ndarray] = None
        if len(parts) >= 9:
            mag = np.array([float(parts[6]), float(parts[7]), float(parts[8])],
                           dtype=np.float64)
        return accel, gyro, mag, None

    @staticmethod
    def _parse_pashr(
        line: str,
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[float]]:
        """
        Parse a ``$PASHR`` NMEA sentence.

        Returns synthetic accel/gyro values of zero — orientation is read
        from the heading/pitch/roll fields and injected directly into the
        AHRS filter state after a partial parse.

        The sentence format is::

            $PASHR,HHH.HH,PP.PP,TRR.RR,XX.XX,YY.YY,ZZ.ZZ,Q*CS

        where the first three float fields are heading (yaw), pitch, and
        tilt (roll).  We return zero raw sensor values and rely on the
        sub-class to inject the Euler angles directly.
        """
        if not line.startswith("$PASHR,"):
            raise ValueError(f"Not a PASHR sentence: {line!r}")
        # Strip checksum
        content = line.split("*")[0][len("$PASHR,"):]
        parts = content.split(",")
        if len(parts) < 3:
            raise ValueError(f"Too few fields in PASHR: {line!r}")

        # Fields: heading, pitch, roll (degrees)
        # Return zeros for raw sensors; the PASHR sub-class overrides the
        # AHRS update to inject Euler angles directly.
        return (
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            None,
            None,
        )

    # ------------------------------------------------------------------
    # Override sample loop for PASHR — inject Euler directly
    # ------------------------------------------------------------------

    def _sample_loop(self) -> None:
        if self._format != "pashr":
            super()._sample_loop()
            return

        period = 1.0 / self._sample_rate
        while self._running:
            t_start = time.monotonic()
            try:
                assert self._serial is not None
                line = self._serial.readline().decode("ascii", errors="ignore").strip()
                if not line.startswith("$PASHR,"):
                    continue

                content = line.split("*")[0][len("$PASHR,"):]
                parts = content.split(",")
                if len(parts) < 3:
                    continue

                yaw_deg = float(parts[0])
                pitch_deg = float(parts[1])
                roll_deg = float(parts[2])

            except Exception as exc:  # noqa: BLE001
                self.read_errors += 1
                logger.debug("SerialAHRS read error: %s", exc)
                time.sleep(period)
                continue

            import math
            from lidar_mapping.utils.transforms import rotation_from_euler
            from lidar_mapping.utils.transforms import _rotation_to_quaternion as r2q

            R = rotation_from_euler(roll_deg, pitch_deg, yaw_deg, degrees=True)

            # Inject the Euler-derived quaternion directly
            q = r2q(R)
            self._ahrs._q = q  # type: ignore[attr-defined]

            reading = IMUReading(
                timestamp=t_start,
                accel_mss=np.zeros(3, dtype=np.float64),
                gyro_rads=np.zeros(3, dtype=np.float64),
                mag_ut=None,
                temperature_c=None,
                roll_deg=roll_deg,
                pitch_deg=pitch_deg,
                yaw_deg=yaw_deg,
                quaternion=q,
            )

            with self._lock:
                self._readings.append(reading)
                if len(self._readings) > self._max_queue:
                    self._readings.pop(0)

            self.samples_read += 1
            elapsed = time.monotonic() - t_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

# ---------------------------------------------------------------------------
# WitMotion UART Driver
# ---------------------------------------------------------------------------

class WitMotionDriver:
    """
    Parser for WTGAHRS2 and similar WitMotion UART IMUs using the 0x55 protocol.
    Outputs native fused angles (Roll, Pitch, Yaw) directly from the device DSP.
    """
    import math

    def __init__(self, port="/dev/ttyS1", baudrate=230400, max_queue=200):
        self.port = port
        self.baudrate = baudrate
        self.max_queue = max_queue
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._readings = []
        self.samples_read = 0
        
        # Intermediate state
        self._last_accel = np.zeros(3)
        self._last_gyro = np.zeros(3)
        self._last_angle = np.zeros(3)
        
    def start(self):
        if self._running: return
        try:
            import serial
            self._ser = serial.Serial(self.port, self.baudrate, timeout=1.0)
        except ImportError:
            raise RuntimeError("pyserial is required")
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        logger.info(f"WitMotion IMU started on {self.port} at {self.baudrate}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        if hasattr(self, '_ser') and self._ser.is_open:
            self._ser.close()

    def get_latest_reading(self) -> Optional[IMUReading]:
        with self._lock:
            if self._readings:
                r = self._readings[-1]
                self._readings.clear()
                return r
        return None

    def _euler_to_quat(self, r, p, y):
        # r, p, y in degrees
        import math
        r_rad = math.radians(r)
        p_rad = math.radians(p)
        y_rad = math.radians(y)
        
        cr = math.cos(r_rad * 0.5)
        sr = math.sin(r_rad * 0.5)
        cp = math.cos(p_rad * 0.5)
        sp = math.sin(p_rad * 0.5)
        cy = math.cos(y_rad * 0.5)
        sy = math.sin(y_rad * 0.5)
        
        q_w = cr * cp * cy + sr * sp * sy
        q_x = sr * cp * cy - cr * sp * sy
        q_y = cr * sp * cy + sr * cp * sy
        q_z = cr * cp * sy - sr * sp * cy
        return np.array([q_w, q_x, q_y, q_z])

    def _read_loop(self):
        buffer = bytearray()
        while self._running:
            try:
                if self._ser.in_waiting:
                    buffer.extend(self._ser.read(self._ser.in_waiting))
                else:
                    d = self._ser.read(11)
                    if d: buffer.extend(d)
                
                while len(buffer) >= 11:
                    if buffer[0] != 0x55:
                        buffer.pop(0)
                        continue
                    
                    packet = buffer[:11]
                    checksum = sum(packet[:10]) & 0xFF
                    if checksum != packet[10]:
                        buffer.pop(0)
                        continue
                        
                    ptype = packet[1]
                    updated = False
                    
                    if ptype == 0x51: # Accel (g -> m/s2)
                        ax = int.from_bytes(packet[2:4], 'little', signed=True) / 32768.0 * 16 * 9.81
                        ay = int.from_bytes(packet[4:6], 'little', signed=True) / 32768.0 * 16 * 9.81
                        az = int.from_bytes(packet[6:8], 'little', signed=True) / 32768.0 * 16 * 9.81
                        self._last_accel = np.array([ax, ay, az])
                        
                    elif ptype == 0x52: # Gyro (deg/s -> rad/s)
                        import math
                        gx = int.from_bytes(packet[2:4], 'little', signed=True) / 32768.0 * 2000 * (math.pi/180.0)
                        gy = int.from_bytes(packet[4:6], 'little', signed=True) / 32768.0 * 2000 * (math.pi/180.0)
                        gz = int.from_bytes(packet[6:8], 'little', signed=True) / 32768.0 * 2000 * (math.pi/180.0)
                        self._last_gyro = np.array([gx, gy, gz])
                        
                    elif ptype == 0x53: # Angle (deg)
                        r = int.from_bytes(packet[2:4], 'little', signed=True) / 32768.0 * 180
                        p = int.from_bytes(packet[4:6], 'little', signed=True) / 32768.0 * 180
                        y = int.from_bytes(packet[6:8], 'little', signed=True) / 32768.0 * 180
                        self._last_angle = np.array([r, p, y])
                        updated = True  # Emit reading on Angle packet, which usually comes last
                    
                    buffer = buffer[11:]
                    
                    if updated:
                        reading = IMUReading(
                            timestamp=time.monotonic(),
                            accel_mss=self._last_accel.copy(),
                            gyro_rads=self._last_gyro.copy(),
                            roll_deg=self._last_angle[0],
                            pitch_deg=self._last_angle[1],
                            yaw_deg=self._last_angle[2],
                            quaternion=self._euler_to_quat(*self._last_angle)
                        )
                        with self._lock:
                            self._readings.append(reading)
                            if len(self._readings) > self.max_queue:
                                self._readings.pop(0)
                        self.samples_read += 1
                        
            except Exception as e:
                import time
                time.sleep(0.1)
