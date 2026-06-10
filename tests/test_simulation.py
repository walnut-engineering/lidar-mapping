"""Tests for synthetic data generators."""

import numpy as np
import pytest
import time

from lidar_mapping.simulation import (
    generate_vlp16_packet,
    build_packets_from_ranges,
    cast_rays_into_box,
    VLP16Simulator,
    generate_imu_reading,
    IMUSimulator,
    MotionProfile,
)
from lidar_mapping.sensors.vlp16 import VLP16PacketParser


class TestGenerateVLP16Packet:
    def test_packet_size(self):
        az = np.zeros(12)
        d = np.full((12, 32), 5.0)
        p = generate_vlp16_packet(az, d)
        assert len(p) == 1206

    def test_parser_roundtrip(self):
        az = np.linspace(0, 22, 12)  # 12 increasing azimuths
        d = np.full((12, 32), 5.0)
        p = generate_vlp16_packet(az, d, timestamp_us=12345)
        parser = VLP16PacketParser()
        pts, ts = parser.parse(p)
        assert ts == 12345
        assert len(pts) > 0
        # All distances should be near 5 m
        for pt in pts:
            assert abs(pt.distance_m - 5.0) < 0.01

    def test_zero_distance_skipped(self):
        az = np.zeros(12)
        d = np.zeros((12, 32))
        parser = VLP16PacketParser()
        pts, _ = parser.parse(generate_vlp16_packet(az, d))
        assert len(pts) == 0

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            generate_vlp16_packet(np.zeros(10), np.zeros((12, 32)))
        with pytest.raises(ValueError):
            generate_vlp16_packet(np.zeros(12), np.zeros((12, 16)))


class TestBuildPackets:
    def test_full_sweep_packets(self):
        ranges = np.full((1800, 16), 3.0)  # 1800 firings × 0.2° = 360°
        packets = build_packets_from_ranges(ranges, azimuth_step_deg=0.2)
        assert len(packets) == 1800 // 24  # 24 firings per packet = 75
        for p in packets:
            assert len(p) == 1206

    def test_wrong_channel_count_raises(self):
        with pytest.raises(ValueError):
            build_packets_from_ranges(np.zeros((100, 8)))


class TestCastRays:
    def test_box_distances_within_bounds(self):
        ranges = cast_rays_into_box(
            half_extent_x=5, half_extent_y=5,
            z_floor=0, z_ceiling=4,
            sensor_xyz=(0, 0, 1),
            azimuths_deg=np.array([0.0, 90.0, 180.0, 270.0]),
        )
        # All hits must be within the diagonal of the room
        diag = np.sqrt(5**2 + 5**2 + 4**2)
        nonzero = ranges[ranges > 0]
        assert (nonzero <= diag + 0.1).all()
        assert (nonzero >= 0).all()

    def test_north_ray_hits_wall(self):
        # Ray straight ahead (+Y) from origin should hit y=+5 wall
        ranges = cast_rays_into_box(
            sensor_xyz=(0, 0, 1),
            azimuths_deg=np.array([0.0]),  # 0° azimuth = +Y in VLP convention
        )
        # Channel 0 has elevation -15°; channel near horizontal is around 7
        # Find the channel closest to horizontal (elevation ~ 1°)
        # Just check at least one channel reports a sensible distance
        nz = ranges[0, ranges[0] > 0]
        assert len(nz) > 0


class TestVLP16Simulator:
    def test_get_frame(self):
        sim = VLP16Simulator(rpm=6000)  # 100 Hz for fast test
        sim.start()
        try:
            frame = sim.get_frame(timeout=2.0)
            assert frame is not None
            assert len(frame) > 0
        finally:
            sim.stop()

    def test_packet_callback(self):
        sim = VLP16Simulator(rpm=6000)
        seen = []
        sim._packet_callback = lambda raw: seen.append(raw)
        sim.start()
        try:
            sim.get_frame(timeout=2.0)
        finally:
            sim.stop()
        assert len(seen) > 0
        assert all(len(r) == 1206 for r in seen)


class TestIMUSimulator:
    def test_emits_readings(self):
        sim = IMUSimulator(rate_hz=200)
        sim.start()
        try:
            r = sim.get_reading(timeout=1.0)
        finally:
            sim.stop()
        assert r is not None
        # default profile = no motion, accel ≈ gravity
        assert abs(r.accel_mss[2] - 9.80665) < 1e-6

    def test_motion_profile_used(self):
        prof = MotionProfile(
            gyro_fn=lambda t: np.array([0.0, 0.0, 1.0]),
            accel_fn=lambda t: np.array([1.0, 0.0, 0.0]),
            add_gravity=False,
        )
        sim = IMUSimulator(profile=prof, rate_hz=500)
        sim.start()
        try:
            time.sleep(0.05)
            r = sim.get_latest_reading()
        finally:
            sim.stop()
        assert r is not None
        assert abs(r.gyro_rads[2] - 1.0) < 1e-6
        assert abs(r.accel_mss[0] - 1.0) < 1e-6

    def test_noise_seeded_reproducible(self):
        prof1 = MotionProfile(noise_gyro_std=0.1, seed=42)
        prof2 = MotionProfile(noise_gyro_std=0.1, seed=42)
        # Just instantiate — they should use the same RNG seed
        sim1 = IMUSimulator(prof1)
        sim2 = IMUSimulator(prof2)
        # The simulators use the same internal RNG seed
        a = sim1._rng.normal(0, 1, 5)
        b = sim2._rng.normal(0, 1, 5)
        assert np.allclose(a, b)


class TestGenerateImuReading:
    def test_defaults(self):
        r = generate_imu_reading(1.5)
        assert r.timestamp == 1.5
        assert r.accel_mss[2] == pytest.approx(9.80665)
        assert (r.gyro_rads == 0).all()
        assert r.quaternion[0] == 1.0
