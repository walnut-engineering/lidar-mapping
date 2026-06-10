"""Capture and inspect HBX serial traffic for protocol inference."""

from __future__ import annotations

import argparse
import collections
import string
import time
from dataclasses import dataclass
from typing import Iterable

try:
    import serial  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pyserial is required. Install with: pip install pyserial") from exc


DEFAULT_BAUDS: tuple[int, ...] = (9600, 19200, 38400, 57600, 115200, 230400)


@dataclass(frozen=True)
class CaptureStats:
    baud: int
    total_bytes: int
    duration_s: float
    printable_ratio: float
    has_lx200_markers: bool
    ascii_preview: str
    hex_preview: str
    top_bytes: list[tuple[int, int]]
    sync_fdfe_count: int
    sync_ee_count: int
    fdfe_gap_hist: list[tuple[int, int]]
    ee_gap_hist: list[tuple[int, int]]
    frame_count: int
    frame_len_hist: list[tuple[int, int]]
    top_frames: list[tuple[bytes, int]]
    crc8_valid: int
    crc8_invalid: int
    inferred_axis_summary: dict[str, list[tuple[int, int]]]
    inferred_remote_summary: dict[str, list[tuple[int, int]]]

    @property
    def bytes_per_second(self) -> float:
        if self.duration_s <= 0:
            return 0.0
        return self.total_bytes / self.duration_s


def _make_ascii_preview(payload: bytes, limit: int = 160) -> str:
    charset = set(string.printable.encode("ascii"))
    out: list[str] = []
    for b in payload[:limit]:
        if b in charset and b not in (0x0b, 0x0c):
            out.append(chr(b))
        else:
            out.append(".")
    return "".join(out)


def _make_hex_preview(payload: bytes, limit: int = 64) -> str:
    return " ".join(f"{b:02X}" for b in payload[:limit])


def _sync_positions(payload: bytes, marker: bytes) -> list[int]:
    positions: list[int] = []
    start = 0
    while True:
        idx = payload.find(marker, start)
        if idx < 0:
            break
        positions.append(idx)
        start = idx + 1
    return positions


def _gap_histogram(positions: list[int]) -> list[tuple[int, int]]:
    if len(positions) < 2:
        return []
    gaps = [b - a for a, b in zip(positions, positions[1:])]
    return collections.Counter(gaps).most_common(6)


def _crc8_poly07(data: bytes, init: int = 0x00, xorout: int = 0x00) -> int:
    crc = init
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc ^ xorout


def _extract_fdfe_frames(payload: bytes) -> list[bytes]:
    frames: list[bytes] = []
    i = 0
    n = len(payload)
    while i < n - 2:
        if payload[i] != 0xFD or payload[i + 1] != 0xFE:
            i += 1
            continue

        if i + 3 >= n:
            break

        frame_id = payload[i + 2]
        remote_len_by_id = {
            0x11: 5,
            0x13: 7,
            0x15: 9,
            0x21: 5,
            0x23: 7,
            0x25: 9,
            0x41: 5,
            0x65: 9,
        }
        frame_len: int | None = None
        if frame_id in remote_len_by_id:
            frame_len = remote_len_by_id[frame_id]
        elif frame_id in (0x11, 0x21):
            frame_len = 5
        elif frame_id in (0x16, 0x26):
            frame_len = 10

        if frame_len is not None and i + frame_len <= n:
            frame = payload[i : i + frame_len]
            frames.append(frame)
            i += frame_len
            continue

        # Unknown ID: fallback to next sync boundary.
        j = payload.find(b"\xFD\xFE", i + 2)
        if j == -1:
            break
        frame = payload[i:j]
        if len(frame) >= 4:
            frames.append(frame)
        i = j

    return frames


def _decode_axis_fields(frame: bytes) -> tuple[str, int] | None:
    if len(frame) != 10:
        return None
    frame_id = frame[2]
    if frame_id not in (0x16, 0x26):
        return None

    # Empirically, bytes 5..6 track axis movement strongly, while 7..8 carry
    # additional state / coarse extension bits that occasionally toggle.
    fine = frame[5] | (frame[6] << 8)
    ext = frame[7] | (frame[8] << 8)
    composite = fine | (ext << 16)
    axis_name = "azimuth_raw" if frame_id == 0x16 else "tilt_raw"
    return axis_name, composite


def _decode_remote_fields(frame: bytes) -> tuple[str, int] | None:
    if len(frame) < 5:
        return None

    frame_id = frame[2]

    # 0x13 / 0x23 look like axis command channels with 2-byte values.
    if frame_id in (0x13, 0x23) and len(frame) == 7:
        val16 = frame[4] | (frame[5] << 8)
        name = "remote_axis_a" if frame_id == 0x13 else "remote_axis_b"
        return name, val16

    # 0x15 / 0x25 / 0x65 carry 4-byte state-like values.
    if frame_id in (0x15, 0x25, 0x65) and len(frame) == 9:
        val32 = frame[4] | (frame[5] << 8) | (frame[6] << 16) | (frame[7] << 24)
        name_map = {
            0x15: "remote_state_a",
            0x25: "remote_state_b",
            0x65: "remote_state_c",
        }
        return name_map[frame_id], val32

    return None


def _capture(port: str, baud: int, duration: float, timeout: float) -> bytes:
    start = time.monotonic()
    data = bytearray()

    with serial.Serial(port=port, baudrate=baud, timeout=0) as ser:
        while time.monotonic() - start < duration:
            chunk = ser.read_all()
            if chunk:
                data.extend(chunk)
            else:
                # Small backoff prevents a busy loop while keeping reads non-blocking.
                time.sleep(max(timeout, 0.001))

    return bytes(data)


def analyze_capture(baud: int, payload: bytes, duration: float) -> CaptureStats:
    printable = sum(1 for b in payload if 32 <= b <= 126 or b in (9, 10, 13))
    ratio = (printable / len(payload)) if payload else 0.0

    has_colon = b":" in payload
    has_hash = b"#" in payload
    has_lx200 = has_colon and has_hash

    counts = collections.Counter(payload)
    top_bytes = counts.most_common(8)
    fdfe_positions = _sync_positions(payload, b"\xFD\xFE")
    ee_positions = _sync_positions(payload, b"\xEE")
    frames = _extract_fdfe_frames(payload)

    frame_counts = collections.Counter(frames)
    top_frames = frame_counts.most_common(8)
    frame_len_hist = collections.Counter(len(frame) for frame in frames).most_common()

    crc8_valid = 0
    crc8_invalid = 0
    axis_counts: dict[str, collections.Counter[int]] = {
        "azimuth_raw": collections.Counter(),
        "tilt_raw": collections.Counter(),
    }
    remote_counts: dict[str, collections.Counter[int]] = {
        "remote_axis_a": collections.Counter(),
        "remote_axis_b": collections.Counter(),
        "remote_state_a": collections.Counter(),
        "remote_state_b": collections.Counter(),
        "remote_state_c": collections.Counter(),
    }
    for frame in frames:
        if len(frame) >= 4:
            expected = _crc8_poly07(frame[:-1])
            if expected == frame[-1]:
                crc8_valid += 1
            else:
                crc8_invalid += 1

        axis_field = _decode_axis_fields(frame)
        if axis_field is not None:
            axis_name, axis_val = axis_field
            axis_counts[axis_name][axis_val] += 1

        remote_field = _decode_remote_fields(frame)
        if remote_field is not None:
            remote_name, remote_val = remote_field
            remote_counts[remote_name][remote_val] += 1

    inferred_axis_summary = {
        name: counter.most_common(6) for name, counter in axis_counts.items() if counter
    }
    inferred_remote_summary = {
        name: counter.most_common(6) for name, counter in remote_counts.items() if counter
    }

    return CaptureStats(
        baud=baud,
        total_bytes=len(payload),
        duration_s=duration,
        printable_ratio=ratio,
        has_lx200_markers=has_lx200,
        ascii_preview=_make_ascii_preview(payload),
        hex_preview=_make_hex_preview(payload),
        top_bytes=top_bytes,
        sync_fdfe_count=len(fdfe_positions),
        sync_ee_count=len(ee_positions),
        fdfe_gap_hist=_gap_histogram(fdfe_positions),
        ee_gap_hist=_gap_histogram(ee_positions),
        frame_count=len(frames),
        frame_len_hist=frame_len_hist,
        top_frames=top_frames,
        crc8_valid=crc8_valid,
        crc8_invalid=crc8_invalid,
        inferred_axis_summary=inferred_axis_summary,
        inferred_remote_summary=inferred_remote_summary,
    )


def _format_top_bytes(top_bytes: Iterable[tuple[int, int]]) -> str:
    return ", ".join(f"0x{val:02X}={count}" for val, count in top_bytes)


def _run_one(port: str, baud: int, duration: float, timeout: float) -> CaptureStats:
    payload = _capture(port=port, baud=baud, duration=duration, timeout=timeout)
    return analyze_capture(baud=baud, payload=payload, duration=duration)


def _print_stats(stats: CaptureStats) -> None:
    print(f"baud={stats.baud}")
    print(
        "summary: "
        f"bytes={stats.total_bytes}, rate={stats.bytes_per_second:.1f} B/s, "
        f"printable={stats.printable_ratio:.1%}, lx200_markers={stats.has_lx200_markers}"
    )
    print(f"top-bytes: {_format_top_bytes(stats.top_bytes)}")
    print(
        "sync-markers: "
        f"FD FE={stats.sync_fdfe_count}, EE={stats.sync_ee_count}, "
        f"FD FE gaps={stats.fdfe_gap_hist or 'n/a'}, "
        f"EE gaps={stats.ee_gap_hist or 'n/a'}"
    )
    if stats.frame_count:
        print(f"fdfe-frames: count={stats.frame_count}, len-hist={stats.frame_len_hist}")
        print(f"crc8: valid={stats.crc8_valid}, invalid={stats.crc8_invalid}")
        if stats.inferred_axis_summary:
            print(f"axis-summary: {stats.inferred_axis_summary}")
        if stats.inferred_remote_summary:
            print(f"remote-summary: {stats.inferred_remote_summary}")
        for frame, count in stats.top_frames:
            frame_hex = " ".join(f"{b:02X}" for b in frame)
            print(f"frame[{count}]: {frame_hex}")
    print(f"hex-preview: {stats.hex_preview}")
    print(f"ascii-preview: {stats.ascii_preview}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lidar-hbx-sniff")
    p.add_argument("--port", default="COM36", help="Serial port")
    p.add_argument("--baud", type=int, action="append", dest="bauds", help="Baud to test")
    p.add_argument("--duration", type=float, default=3.0, help="Capture duration per baud (s)")
    p.add_argument("--timeout", type=float, default=0.05, help="Serial read timeout (s)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bauds = args.bauds if args.bauds else list(DEFAULT_BAUDS)

    for i, baud in enumerate(bauds):
        if i:
            print("-" * 72)
        try:
            stats = _run_one(args.port, baud, args.duration, args.timeout)
        except Exception as exc:  # noqa: BLE001
            print(f"baud={baud}")
            print(f"error: {exc}")
            continue
        _print_stats(stats)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
