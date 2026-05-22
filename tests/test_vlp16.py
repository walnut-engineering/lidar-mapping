"""
Unit tests for lidar_mapping.sensors.vlp16

Tests the VLP16PacketParser and VLP16Frame without any hardware or network
connectivity.  All packets are crafted synthetically.
"""

from __future__ import annotations

import math
import struct

import numpy as np
import pytest

from lidar_mapping.sensors.vlp16 import (
    VLP16Frame,
    VLP16PacketParser,
    _BLOCK_FLAG,
    _BLOCKS_PER_PACKET,
    _DISTANCE_RESOLUTION,
    _LASERS,
    _PACKET_SIZE,
    _VERTICAL_ANGLES_DEG,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_packet(
    blocks: list[dict] | None = None,
    timestamp_us: int = 0,
    return_mode: int = 0x37,
    product_id: int = 0x22,
) -> bytes:
    """
    Build a syntactically valid VLP-16 1206-byte UDP payload.

    Each block dict may have:
      - ``azimuth``: raw azimuth (0–35999, hundredths of a degree)
      - ``channels``: list of (distance_raw, intensity) for 32 channels
    """
    payload = bytearray(_PACKET_SIZE)

    if blocks is None:
        blocks = []

    for i in range(_BLOCKS_PER_PACKET):
        offset = i * 100  # 100 bytes per block
        block = blocks[i] if i < len(blocks) else {}
        azimuth = block.get("azimuth", i * 300)
        channels = block.get("channels", [(100, 10)] * (_LASERS * 2))

        struct.pack_into("<HH", payload, offset, _BLOCK_FLAG, azimuth)
        for j, (dist, inten) in enumerate(channels[: _LASERS * 2]):
            struct.pack_into("<HB", payload, offset + 4 + j * 3, dist, inten)

    # GPS timestamp at bytes 1200–1203
    struct.pack_into("<I", payload, 1200, timestamp_us)
    payload[1204] = return_mode
    payload[1205] = product_id

    return bytes(payload)


# ---------------------------------------------------------------------------
# VLP16PacketParser
# ---------------------------------------------------------------------------

class TestVLP16PacketParser:
    def test_wrong_length_raises(self):
        parser = VLP16PacketParser()
        with pytest.raises(ValueError, match="1206 bytes"):
            parser.parse(b"\x00" * 1000)

    def test_empty_distances_produce_no_points(self):
        """All zero distances → all returns skipped (below 0.1 m min range)."""
        blocks = [{"azimuth": i * 3000, "channels": [(0, 0)] * 32}
                  for i in range(_BLOCKS_PER_PACKET)]
        raw = _build_packet(blocks)
        parser = VLP16PacketParser()
        points, ts = parser.parse(raw)
        assert points == []
        assert ts == 0

    def test_timestamp_parsed_correctly(self):
        raw = _build_packet(timestamp_us=123456789)
        parser = VLP16PacketParser()
        _, ts = parser.parse(raw)
        assert ts == 123456789

    def test_single_return_xyz(self):
        """
        One block with a known distance and azimuth — verify XYZ geometry.
        Azimuth 0° → point lies on the +Y axis.
        """
        dist_raw = 1000          # 1000 × 0.002 m = 2.0 m
        intensity = 50
        channels = [(dist_raw, intensity)] * 32

        blocks = [{"azimuth": 0, "channels": channels}] + [
            {"azimuth": j * 100, "channels": [(0, 0)] * 32}
            for j in range(1, _BLOCKS_PER_PACKET)
        ]
        raw = _build_packet(blocks)
        parser = VLP16PacketParser()
        points, _ = parser.parse(raw)

        # Should have at least one valid point (first block, channel 0)
        assert len(points) > 0

        # Channel 0 has vertical angle −15°
        ch0_points = [p for p in points if p.channel == 0 and p.azimuth_deg == 0.0]
        assert len(ch0_points) >= 1
        p = ch0_points[0]

        dist_m = dist_raw * _DISTANCE_RESOLUTION  # 2.0 m
        az_rad = math.radians(0.0)
        el_rad = math.radians(_VERTICAL_ANGLES_DEG[0])  # -15°

        expected_x = dist_m * math.cos(el_rad) * math.sin(az_rad)
        expected_y = dist_m * math.cos(el_rad) * math.cos(az_rad)
        expected_z = dist_m * math.sin(el_rad)

        assert abs(p.x - expected_x) < 1e-4
        assert abs(p.y - expected_y) < 1e-4
        assert abs(p.z - expected_z) < 1e-4
        assert p.distance_m == pytest.approx(dist_m, abs=1e-6)
        assert p.intensity == float(intensity)

    def test_malformed_block_flag_skipped(self):
        """Blocks with a bad flag byte should be silently skipped."""
        raw = bytearray(_build_packet())
        # Corrupt the first block's flag
        struct.pack_into("<H", raw, 0, 0x1234)
        parser = VLP16PacketParser()
        # Should not raise
        points, _ = parser.parse(bytes(raw))
        # Cannot assert a specific count — just that it doesn't crash
        assert isinstance(points, list)

    def test_point_count_plausible(self):
        """A fully populated packet should produce O(1000) points."""
        dist_raw = 500
        channels = [(dist_raw, 100)] * 32
        blocks = [{"azimuth": i * 300, "channels": channels}
                  for i in range(_BLOCKS_PER_PACKET)]
        raw = _build_packet(blocks)
        parser = VLP16PacketParser()
        points, _ = parser.parse(raw)
        # 12 blocks × 2 sequences × 16 lasers = 384 possible returns
        assert 100 <= len(points) <= 384

    def test_azimuth_interpolation_second_sequence(self):
        """
        The second firing sequence in a block must use the interpolated azimuth
        (halfway between the current and next block's azimuth).
        """
        dist_raw = 1000
        channels = [(dist_raw, 10)] * 32
        blocks = [{"azimuth": i * 1000, "channels": channels}
                  for i in range(_BLOCKS_PER_PACKET)]
        raw = _build_packet(blocks)
        parser = VLP16PacketParser()
        points, _ = parser.parse(raw)

        # Collect unique azimuths from block 1 (index 1), first two firing seqs
        azimuths = sorted({round(p.azimuth_deg, 2) for p in points
                           if 9.0 < p.azimuth_deg < 16.0})
        # Expect both the raw block azimuth (10.0°) and the interpolated one (15.0°)
        assert 10.0 in azimuths
        assert 15.0 in azimuths


# ---------------------------------------------------------------------------
# VLP16Frame
# ---------------------------------------------------------------------------

class TestVLP16Frame:
    def test_empty_frame_to_numpy(self):
        frame = VLP16Frame()
        arr = frame.to_numpy()
        assert arr.shape == (0, 4)
        assert arr.dtype == np.float32

    def test_frame_len(self):
        frame = VLP16Frame()
        assert len(frame) == 0

    def test_to_numpy_shape_and_dtype(self):
        """to_numpy() should return (N, 4) float32 with x,y,z,intensity."""
        dist_raw = 2000
        channels = [(dist_raw, 20)] * 32
        blocks = [{"azimuth": i * 300, "channels": channels}
                  for i in range(_BLOCKS_PER_PACKET)]
        raw = _build_packet(blocks)
        parser = VLP16PacketParser()
        points, _ = parser.parse(raw)

        frame = VLP16Frame(points=points)
        arr = frame.to_numpy()

        assert arr.ndim == 2
        assert arr.shape[1] == 4
        assert arr.dtype == np.float32
        assert len(arr) == len(points)

    def test_intensity_column_in_range(self):
        """Intensity values should be in 0–255."""
        dist_raw = 500
        channels = [(dist_raw, 200)] * 32
        blocks = [{"azimuth": i * 300, "channels": channels}
                  for i in range(_BLOCKS_PER_PACKET)]
        raw = _build_packet(blocks)
        parser = VLP16PacketParser()
        points, _ = parser.parse(raw)
        frame = VLP16Frame(points=points)
        arr = frame.to_numpy()
        assert np.all(arr[:, 3] >= 0)
        assert np.all(arr[:, 3] <= 255)
