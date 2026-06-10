"""Tests for the serial mount probe/jog CLI."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mount_module():
    mock_serial_mod = types.ModuleType("serial")
    mock_port = MagicMock()
    mock_serial_mod.Serial = MagicMock(return_value=mock_port)

    with patch.dict(sys.modules, {"serial": mock_serial_mod}):
        import lidar_mapping.cli.mount as mount

        with patch.object(mount, "_SERIAL_AVAILABLE", True):
            yield mount, mock_serial_mod, mock_port


def test_send_and_receive_reads_reply(mount_module):
    mount, _, mock_port = mount_module
    mock_port.read.side_effect = [b"V", b"1", b".", b"0", b"0", b"#"]

    session = mount.LX200SerialSession("COM36", 9600, timeout=0.1)
    session.open()

    reply = session.send_and_receive(":V#", timeout=0.1)

    assert reply == "V1.00"
    mock_port.write.assert_called_with(b":V#")


def test_jog_sends_start_and_stop(mount_module):
    mount, _, mock_port = mount_module
    session = mount.LX200SerialSession("COM36", 9600, timeout=0.1)
    session._serial = mock_port

    with patch.object(mount.time, "sleep", return_value=None):
        session.jog("north", 0.25)

    assert mock_port.write.call_args_list[0].args[0] == b":Mn#"
    assert mock_port.write.call_args_list[1].args[0] == b":Qn#"


def test_choose_best_result_prefers_most_replies(mount_module):
    mount, _, _ = mount_module
    results = [
        mount.ProbeResult(baudrate=9600, responses={}, opened=True),
        mount.ProbeResult(baudrate=115200, responses={":V#": "V1.00", ":GR#": "01:02:03"}, opened=True),
        mount.ProbeResult(baudrate=19200, responses={":V#": "V1.00"}, opened=True),
    ]

    best = mount.choose_best_result(results)

    assert best is not None
    assert best.baudrate == 115200


def test_probe_port_collects_responses(mount_module):
    mount, _, mock_port = mount_module
    mock_port.read.side_effect = [b"V", b"1", b".", b"0", b"0", b"#"]

    result = mount.probe_port("COM36", 9600, commands=(":V#",), timeout=0.05)

    assert result.opened is True
    assert result.responses == {":V#": "V1.00"}


def test_scan_baud_rates_returns_all_results(mount_module):
    mount, _, _ = mount_module

    def fake_probe_port(port, baudrate, commands=(), timeout=0.5):
        return mount.ProbeResult(baudrate=baudrate, responses={":V#": "V1.00"} if baudrate == 115200 else {}, opened=True)

    with patch.object(mount, "probe_port", side_effect=fake_probe_port):
        results = mount.scan_baud_rates("COM36", [9600, 115200], timeout=0.1)

    assert [result.baudrate for result in results] == [9600, 115200]
    assert results[1].responses == {":V#": "V1.00"}


def test_incomplete_get_response_raises(mount_module):
    mount, _, mock_port = mount_module

    data = iter([b"V", b"1", b".", b"0", b"0"])

    def read_side_effect(_n):
        try:
            return next(data)
        except StopIteration:
            return b""

    mock_port.read.side_effect = read_side_effect

    session = mount.LX200SerialSession("COM36", 9600, timeout=0.05)
    session.open()

    with pytest.raises(RuntimeError, match="Incomplete response"):
        session.send_and_receive(":V#", timeout=0.05)


def test_read_ioptron_status_parses_fields(mount_module):
    mount, _, mock_port = mount_module
    # 23-char payload + '#':
    # lon 001234567, lat 03240000, gps=2 motion=7 track=0 key=4 time=2 hemisphere=1
    payload = b"00123456703240000270421#"
    mock_port.read.side_effect = [bytes([b]) for b in payload]

    status = mount.read_ioptron_status("COM36", 9600, timeout=0.1)

    assert status.gps_state == "Valid"
    assert status.motion_state == "At home"
    assert status.tracking_rate == "Sidereal"
    assert status.key_speed == "16x"
    assert status.time_source == "GPS"
    assert status.hemisphere == "North"