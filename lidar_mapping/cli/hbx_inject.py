"""Transmit inferred HBX control packets to a mount over serial."""

from __future__ import annotations

import argparse
import time

try:
    import serial  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pyserial is required. Install with: pip install pyserial") from exc


SYNC = bytes((0xFD, 0xFE))
AXIS_FRAME_ID = {"azimuth": 0x13, "tilt": 0x23}

# Inferred command words from captured handset traffic.
DIRECTION_VALUES = {
    "positive": 0xBC14,
    "negative": 0xE010,
}


def crc8_poly07(data: bytes, init: int = 0x00, xorout: int = 0x00) -> int:
    crc = init
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc ^ xorout


def make_packet(frame_id: int, mode: int, value_lo: int, value_hi: int) -> bytes:
    payload = bytes((frame_id, mode, value_lo, value_hi))
    raw = SYNC + payload
    return raw + bytes((crc8_poly07(raw),))


def heartbeat_packets() -> tuple[bytes, bytes]:
    p11 = make_packet(0x11, 0x01, 0x00, 0x00)
    p21 = make_packet(0x21, 0x01, 0x00, 0x00)
    return p11, p21


def axis_packets(
    axis: str,
    *,
    active_mode: int,
    active_value: int,
    release_mode: int,
    release_value: int,
) -> tuple[bytes, bytes]:
    if axis not in AXIS_FRAME_ID:  # pragma: no cover
        raise ValueError(f"Unknown axis: {axis}")
    frame_id = AXIS_FRAME_ID[axis]

    active = make_packet(
        frame_id,
        active_mode & 0xFF,
        active_value & 0xFF,
        (active_value >> 8) & 0xFF,
    )
    release = make_packet(
        frame_id,
        release_mode & 0xFF,
        release_value & 0xFF,
        (release_value >> 8) & 0xFF,
    )
    return active, release


def to_hex(packet: bytes) -> str:
    return " ".join(f"{b:02X}" for b in packet)


def run_sequence(
    *,
    port: str,
    baud: int,
    axis: str,
    hold_seconds: float,
    pulse_count: int,
    gap_seconds: float,
    active_mode: int,
    active_value: int,
    release_mode: int,
    release_value: int,
    move_seconds: float,
    tx_hz: float,
    dry_run: bool,
) -> None:
    hb11, hb21 = heartbeat_packets()
    active, release = axis_packets(
        axis,
        active_mode=active_mode,
        active_value=active_value,
        release_mode=release_mode,
        release_value=release_value,
    )

    print(f"heartbeat 11: {to_hex(hb11)}")
    print(f"heartbeat 21: {to_hex(hb21)}")
    print(f"axis active: {to_hex(active)}")
    print(f"axis release: {to_hex(release)}")

    if dry_run:
        print("dry-run enabled, nothing transmitted")
        return

    with serial.Serial(port=port, baudrate=baud, timeout=0) as ser:
        if move_seconds > 0:
            tx_count = _stream_motion(
                ser=ser,
                hb11=hb11,
                hb21=hb21,
                active=active,
                release=release,
                move_seconds=move_seconds,
                tx_hz=tx_hz,
            )
            print(f"stream sent for {move_seconds:.2f}s ({tx_count} active frames)")
            return

        for i in range(pulse_count):
            ser.write(hb11)
            ser.write(hb21)
            ser.write(active)
            time.sleep(max(hold_seconds, 0.0))
            ser.write(release)
            ser.write(hb11)
            ser.write(hb21)
            if i < pulse_count - 1:
                time.sleep(max(gap_seconds, 0.0))
            print(f"pulse {i + 1}/{pulse_count} sent")


def _stream_motion(
    *,
    ser,
    hb11: bytes,
    hb21: bytes,
    active: bytes,
    release: bytes,
    move_seconds: float,
    tx_hz: float,
) -> int:
    period = 1.0 / max(tx_hz, 1.0)
    deadline = time.monotonic() + max(move_seconds, 0.0)
    tx_count = 0
    while time.monotonic() < deadline:
        ser.write(hb11)
        ser.write(hb21)
        ser.write(active)
        tx_count += 1
        time.sleep(period)

    ser.write(release)
    ser.write(hb11)
    ser.write(hb21)
    return tx_count


def run_scan(
    *,
    port: str,
    baud: int,
    az_seconds: float,
    tilt_seconds: float,
    settle_seconds: float,
    tx_hz: float,
    dry_run: bool,
) -> None:
    hb11, hb21 = heartbeat_packets()

    segments = (
        ("azimuth", "positive", az_seconds),
        ("tilt", "positive", tilt_seconds),
        ("azimuth", "negative", az_seconds),
        ("tilt", "negative", tilt_seconds),
    )

    sequence = []
    for axis, direction, duration in segments:
        active, release = axis_packets(
            axis,
            active_mode=0x04,
            active_value=DIRECTION_VALUES[direction],
            release_mode=0x02,
            release_value=0x0000,
        )
        sequence.append((axis, direction, duration, active, release))

    for axis, direction, duration, active, release in sequence:
        print(
            f"segment {axis}/{direction}: duration={duration:.2f}s "
            f"active={to_hex(active)} release={to_hex(release)}"
        )

    if dry_run:
        print("dry-run enabled, nothing transmitted")
        return

    with serial.Serial(port=port, baudrate=baud, timeout=0) as ser:
        for idx, (axis, direction, duration, active, release) in enumerate(sequence, start=1):
            tx_count = _stream_motion(
                ser=ser,
                hb11=hb11,
                hb21=hb21,
                active=active,
                release=release,
                move_seconds=duration,
                tx_hz=tx_hz,
            )
            print(f"segment {idx}/4 {axis}/{direction} sent ({tx_count} active frames)")
            if idx < len(sequence):
                time.sleep(max(settle_seconds, 0.0))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lidar-hbx-inject")
    p.add_argument("--port", default="COM36", help="Serial port")
    p.add_argument("--baud", type=int, default=19200, help="Serial baud")
    p.add_argument("--axis", choices=("azimuth", "tilt"), required=True, help="Axis to pulse")
    p.add_argument(
        "--direction",
        choices=("positive", "negative"),
        default="negative",
        help="Direction preset used when --active-value is not provided",
    )
    p.add_argument("--hold", type=float, default=0.10, help="Seconds to hold active command")
    p.add_argument("--pulses", type=int, default=1, help="Number of pulses to send")
    p.add_argument("--gap", type=float, default=0.30, help="Gap between pulses in seconds")
    p.add_argument(
        "--move-seconds",
        type=float,
        default=0.0,
        help="If > 0, stream active commands for this duration instead of pulse mode",
    )
    p.add_argument(
        "--tx-hz",
        type=float,
        default=25.0,
        help="Transmit rate in sustained mode (default 25 Hz)",
    )
    p.add_argument(
        "--max-move-seconds",
        type=float,
        default=30.0,
        help="Safety cap for any single sustained segment (default 30s)",
    )
    p.add_argument(
        "--scan-room",
        action="store_true",
        help="Run a 4-segment az/tilt scan sequence for lidar acquisition",
    )
    p.add_argument(
        "--az-seconds",
        type=float,
        default=4.0,
        help="Segment duration for azimuth moves in scan mode",
    )
    p.add_argument(
        "--tilt-seconds",
        type=float,
        default=2.0,
        help="Segment duration for tilt moves in scan mode",
    )
    p.add_argument(
        "--settle-seconds",
        type=float,
        default=0.5,
        help="Pause between scan segments",
    )
    p.add_argument(
        "--active-mode",
        type=lambda s: int(s, 0),
        default=0x04,
        help="Active frame mode byte (default 0x04)",
    )
    p.add_argument(
        "--active-value",
        type=lambda s: int(s, 0),
        default=None,
        help="Active 16-bit little-endian value (default from --direction)",
    )
    p.add_argument(
        "--release-mode",
        type=lambda s: int(s, 0),
        default=0x02,
        help="Release frame mode byte (default 0x02)",
    )
    p.add_argument(
        "--release-value",
        type=lambda s: int(s, 0),
        default=0x0000,
        help="Release 16-bit little-endian value (default 0x0000)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print packets without transmitting")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.move_seconds > args.max_move_seconds:
        raise SystemExit(
            f"move-seconds {args.move_seconds} exceeds max-move-seconds {args.max_move_seconds}"
        )
    if args.az_seconds > args.max_move_seconds:
        raise SystemExit(
            f"az-seconds {args.az_seconds} exceeds max-move-seconds {args.max_move_seconds}"
        )
    if args.tilt_seconds > args.max_move_seconds:
        raise SystemExit(
            f"tilt-seconds {args.tilt_seconds} exceeds max-move-seconds {args.max_move_seconds}"
        )

    active_value = (
        args.active_value
        if args.active_value is not None
        else DIRECTION_VALUES[args.direction]
    )

    if args.scan_room:
        run_scan(
            port=args.port,
            baud=args.baud,
            az_seconds=args.az_seconds,
            tilt_seconds=args.tilt_seconds,
            settle_seconds=args.settle_seconds,
            tx_hz=args.tx_hz,
            dry_run=args.dry_run,
        )
        return 0

    run_sequence(
        port=args.port,
        baud=args.baud,
        axis=args.axis,
        hold_seconds=args.hold,
        pulse_count=args.pulses,
        gap_seconds=args.gap,
        active_mode=args.active_mode,
        active_value=active_value,
        release_mode=args.release_mode,
        release_value=args.release_value,
        move_seconds=args.move_seconds,
        tx_hz=args.tx_hz,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
