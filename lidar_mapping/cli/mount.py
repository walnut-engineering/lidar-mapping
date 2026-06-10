"""Probe and control an iOptron/LX200-compatible mount over serial.

This tool is aimed at bring-up and validation for UART-connected mounts.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

try:
    import serial  # type: ignore

    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False


log = logging.getLogger("lidar-mount")

COMMON_BAUD_RATES: tuple[int, ...] = (
    9600,
    115200,
    115000,
    230400,
    19200,
    38400,
    57600,
    4800,
    2400,
)
IDENT_COMMANDS: tuple[str, ...] = (":V#", ":GVP#", ":GR#", ":GD#", ":GA#", ":GZ#")
IOPTRON_COMMANDS: tuple[str, ...] = (
    ":MountInfo#",
    ":GVP#",
    ":FW1#",
    ":GEP#",
    ":GLS#",
    ":GUT#",
    ":GU#",
    ":GX90#",
    ":V#",
    ":GR#",
    ":GD#",
    ":GA#",
    ":GZ#",
)

LX200_JOG_COMMANDS: dict[str, tuple[str, str]] = {
    "north": (":Mn#", ":Qn#"),
    "south": (":Ms#", ":Qs#"),
    "east": (":Me#", ":Qe#"),
    "west": (":Mw#", ":Qw#"),
}

IOPTRON_V3_JOG_COMMANDS: dict[str, tuple[str, str]] = {
    "north": (":mn#", ":qD#"),
    "south": (":ms#", ":qD#"),
    "east": (":me#", ":qR#"),
    "west": (":mw#", ":qR#"),
}


def _expects_terminator(command: str) -> bool:
    command = command.strip()
    if command.startswith(":G") or command.startswith(":V") or command.startswith(":F"):
        return True
    if command in (":MountInfo#",):
        return True
    return False


def _validate_response(command: str, response: str) -> None:
    if not response:
        raise RuntimeError(f"Mount did not respond to command {command}")

    if _expects_terminator(command):
        if not response.endswith("#"):
            raise RuntimeError(f"Incomplete response for {command}: {response!r}")
    else:
        cleaned = response.strip()
        if len(cleaned) == 1 and cleaned not in ("0", "1"):
            raise RuntimeError(f"Unexpected single-byte response for {command}: {response!r}")


@dataclass(frozen=True)
class ProbeResult:
    baudrate: int
    responses: dict[str, str]
    opened: bool

    @property
    def score(self) -> int:
        return len(self.responses)


@dataclass(frozen=True)
class MountStatus:
    longitude_deg: float
    latitude_deg: float
    gps_state: str
    motion_state: str
    tracking_rate: str
    key_speed: str
    time_source: str
    hemisphere: str


class LX200SerialSession:
    def __init__(self, port: str, baudrate: int, timeout: float = 0.5) -> None:
        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout
        self._serial = None

    def open(self) -> None:
        if not _SERIAL_AVAILABLE:
            raise ImportError(
                "pyserial is required for lidar-mount. Install it with: pip install pyserial"
            )

        import serial as pyserial  # type: ignore

        self._serial = pyserial.Serial(
            self._port,
            baudrate=self._baudrate,
            timeout=self._timeout,
        )
        self._flush_input()

    def close(self) -> None:
        if self._serial is not None:
            try:
                if hasattr(self._serial, "close"):
                    self._serial.close()
            finally:
                self._serial = None

    def __enter__(self) -> "LX200SerialSession":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _flush_input(self) -> None:
        if self._serial is None:
            return
        for name in ("reset_input_buffer", "flushInput"):
            if hasattr(self._serial, name):
                getattr(self._serial, name)()
                break
        for name in ("reset_output_buffer", "flushOutput"):
            if hasattr(self._serial, name):
                getattr(self._serial, name)()
                break

    def send(self, command: str) -> None:
        if self._serial is None:
            raise RuntimeError("Serial session is not open")
        self._serial.write(command.encode("ascii"))

    def read_reply(self, timeout: Optional[float] = None) -> str:
        if self._serial is None:
            raise RuntimeError("Serial session is not open")

        deadline = time.monotonic() + (self._timeout if timeout is None else timeout)
        payload = bytearray()
        while time.monotonic() < deadline:
            chunk = self._serial.read(1)
            if not chunk:
                continue
            payload.extend(chunk)
            if payload.endswith(b"#") or payload.endswith(b"\n") or payload.endswith(b"\r"):
                break

        return payload.decode("ascii", errors="ignore").strip()

    def send_and_receive(self, command: str, timeout: Optional[float] = None) -> str:
        self.send(command)
        response = self.read_reply(timeout=timeout)
        _validate_response(command, response)
        return response.rstrip("#")

    def jog(
        self,
        direction: str,
        duration: float,
        commands: Optional[dict[str, tuple[str, str]]] = None,
    ) -> None:
        selected_commands = commands or LX200_JOG_COMMANDS

        if direction not in selected_commands:
            raise ValueError(f"Unknown direction: {direction!r}")

        start_command, stop_command = selected_commands[direction]
        self.send(start_command)
        try:
            time.sleep(max(duration, 0.0))
        finally:
            self.send(stop_command)


def probe_port(
    port: str,
    baudrate: int,
    commands: Sequence[str] = IDENT_COMMANDS,
    timeout: float = 0.5,
) -> ProbeResult:
    responses: dict[str, str] = {}
    try:
        with LX200SerialSession(port, baudrate, timeout=timeout) as session:
            for command in commands:
                try:
                    response = session.send_and_receive(command, timeout=timeout)
                except Exception as exc:  # noqa: BLE001
                    log.debug("Command %s failed @ %d: %s", command, baudrate, exc)
                    continue
                if response:
                    responses[command] = response
            return ProbeResult(baudrate=baudrate, responses=responses, opened=True)
    except Exception as exc:  # noqa: BLE001
        log.debug("Probe failed at %d baud: %s", baudrate, exc)
        return ProbeResult(baudrate=baudrate, responses={}, opened=False)


def scan_baud_rates(
    port: str,
    baudrates: Iterable[int],
    commands: Sequence[str] = IDENT_COMMANDS,
    timeout: float = 0.5,
) -> list[ProbeResult]:
    return [probe_port(port, baudrate, commands=commands, timeout=timeout) for baudrate in baudrates]


def choose_best_result(results: Sequence[ProbeResult]) -> Optional[ProbeResult]:
    successful = [result for result in results if result.opened and result.score > 0]
    if not successful:
        return None
    return max(successful, key=lambda result: result.score)


def read_ioptron_status(port: str, baudrate: int, timeout: float = 0.5) -> MountStatus:
    gps_state = ("No GPS", "No data", "Valid")
    motion_state = (
        "Stopped",
        "Tracking no PEC",
        "Slewing",
        "Auto-guiding",
        "Meridian flipping",
        "Tracking with PEC",
        "Parked",
        "At home",
    )
    tracking_rate = ("Sidereal", "Lunar", "Solar", "King", "Custom")
    key_speed = ("1x", "2x", "4x", "8x", "16x", "32x", "64x", "128x", "256x", "512x", "Max")
    time_source = ("Communicated", "Hand controller", "GPS")
    hemisphere = ("South", "North")

    with LX200SerialSession(port, baudrate, timeout=timeout) as session:
        resp = session.send_and_receive(":GLS#", timeout=timeout)

    if len(resp) < 23:
        raise RuntimeError(f"Unexpected :GLS response length: {len(resp)} ({resp!r})")

    return MountStatus(
        longitude_deg=float(int(resp[0:9])) / 360000.0,
        latitude_deg=float(int(resp[9:17])) / 360000.0 - 90.0,
        gps_state=gps_state[int(resp[17])],
        motion_state=motion_state[int(resp[18])],
        tracking_rate=tracking_rate[int(resp[19])],
        key_speed=key_speed[int(resp[20])],
        time_source=time_source[int(resp[21])],
        hemisphere=hemisphere[int(resp[22])],
    )


def build_arg_parser() -> argparse.ArgumentParser:
    default_port = "COM36" if os.name == "nt" else "/dev/ttyUSB0"
    parser = argparse.ArgumentParser(prog="lidar-mount")
    parser.add_argument("--port", default=default_port, help="Serial port to test")
    parser.add_argument("--log-level", default="INFO")

    subparsers = parser.add_subparsers(dest="command", required=True)

    probe = subparsers.add_parser("probe", help="Scan baud rates and query the mount")
    probe.add_argument("--baud", type=int, action="append", dest="bauds", help="Baud rate to test")
    probe.add_argument(
        "--profile",
        choices=("lx200", "ioptron-v3"),
        default="ioptron-v3",
        help="Command set to try while probing",
    )
    probe.add_argument("--timeout", type=float, default=0.5, help="Serial read timeout in seconds")

    jog = subparsers.add_parser("jog", help="Move the mount in one direction for a short time")
    jog.add_argument("direction", choices=sorted(LX200_JOG_COMMANDS), help="Direction to jog")
    jog.add_argument("--baud", type=int, default=9600, help="Serial baud rate")
    jog.add_argument(
        "--profile",
        choices=("lx200", "ioptron-v3"),
        default="ioptron-v3",
        help="Command set to use for jogging",
    )
    jog.add_argument("--duration", type=float, default=0.5, help="Move duration in seconds")
    jog.add_argument("--timeout", type=float, default=0.5, help="Serial read timeout in seconds")

    status = subparsers.add_parser("status", help="Read iOptron :GLS status")
    status.add_argument("--baud", type=int, default=9600, help="Serial baud rate")
    status.add_argument("--timeout", type=float, default=0.5, help="Serial read timeout in seconds")

    demo = subparsers.add_parser("demo", help="Run a short motion pattern")
    demo.add_argument("--baud", type=int, default=9600, help="Serial baud rate")
    demo.add_argument(
        "--profile",
        choices=("lx200", "ioptron-v3"),
        default="ioptron-v3",
        help="Command set to use for demo movement",
    )
    demo.add_argument("--duration", type=float, default=0.4, help="Seconds per move")
    demo.add_argument("--timeout", type=float, default=0.5, help="Serial read timeout in seconds")

    return parser


def _format_probe_result(result: ProbeResult) -> str:
    if not result.opened:
        return f"{result.baudrate}: no response"
    if not result.responses:
        return f"{result.baudrate}: opened, no replies"
    pieces = [f"{command} -> {reply}" for command, reply in result.responses.items()]
    return f"{result.baudrate}: " + "; ".join(pieces)


def _run_probe(args: argparse.Namespace) -> int:
    baudrates = args.bauds if args.bauds else COMMON_BAUD_RATES
    commands = IOPTRON_COMMANDS if args.profile == "ioptron-v3" else IDENT_COMMANDS
    results = scan_baud_rates(args.port, baudrates, commands=commands, timeout=args.timeout)
    for result in results:
        print(_format_probe_result(result))

    best = choose_best_result(results)
    if best is None:
        return 1

    print(f"best baud: {best.baudrate}")
    return 0


def _run_jog(args: argparse.Namespace) -> int:
    with LX200SerialSession(args.port, args.baud, timeout=args.timeout) as session:
        commands = IOPTRON_V3_JOG_COMMANDS if args.profile == "ioptron-v3" else LX200_JOG_COMMANDS
        session.jog(args.direction, args.duration, commands=commands)
    print(f"jogged {args.direction} for {args.duration:.2f}s")
    return 0


def _run_status(args: argparse.Namespace) -> int:
    status = read_ioptron_status(args.port, args.baud, timeout=args.timeout)
    print(
        "status: "
        f"lon={status.longitude_deg:.6f}, lat={status.latitude_deg:.6f}, "
        f"gps={status.gps_state}, motion={status.motion_state}, track={status.tracking_rate}, "
        f"key={status.key_speed}, time={status.time_source}, hemi={status.hemisphere}"
    )
    return 0


def _run_demo(args: argparse.Namespace) -> int:
    pattern = ("north", "south", "east", "west")
    with LX200SerialSession(args.port, args.baud, timeout=args.timeout) as session:
        commands = IOPTRON_V3_JOG_COMMANDS if args.profile == "ioptron-v3" else LX200_JOG_COMMANDS
        for direction in pattern:
            session.jog(direction, args.duration, commands=commands)
    print("demo completed")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.command == "probe":
        return _run_probe(args)
    if args.command == "jog":
        return _run_jog(args)
    if args.command == "status":
        return _run_status(args)
    if args.command == "demo":
        return _run_demo(args)
    raise ValueError(f"Unknown command: {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
