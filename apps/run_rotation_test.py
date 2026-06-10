"""Timed rotational rate capture.

Polls the running pipeline's /imu endpoint over a fixed duration and reports
measured yaw rate from two independent sources:

  1. Unwrapped absolute yaw (from AHRS quaternion), linear-fit slope
  2. Mean gyro_z (body-frame angular velocity around the up axis)

Use this to validate the IMU output against a known rotational input
(e.g. spinning the rig on a turntable, or a controlled hand-spin at a fixed
rate). The script also snapshots /snapshot/top_down before and after so the
agent can visually compare the accumulated map.

Usage::

    # Start the pipeline in another terminal:
    #   python3 -m apps.run_stationary --no-viewer --calibration-yaml calibration.yaml
    # Then in this terminal, with the rig spinning at ~30 deg/s:
    python3 -m apps.run_rotation_test --duration 10 --expected-rate 30
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import List, Tuple

import urllib.request
import json


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=2.0) as r:
        return json.loads(r.read().decode("utf-8"))


def _save(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url, timeout=5.0) as r:
        dest.write_bytes(r.read())


def _unwrap_yaw(samples: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not samples:
        return samples
    out = [samples[0]]
    offset = 0.0
    prev = samples[0][1]
    for t, y in samples[1:]:
        dy = y - prev
        if dy > 180.0:
            offset -= 360.0
        elif dy < -180.0:
            offset += 360.0
        out.append((t, y + offset))
        prev = y
    return out


def _linear_fit_slope(samples: List[Tuple[float, float]]) -> float:
    if len(samples) < 2:
        return float("nan")
    n = len(samples)
    t0 = samples[0][0]
    xs = [t - t0 for t, _ in samples]
    ys = [y for _, y in samples]
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0.0:
        return float("nan")
    return (n * sxy - sx * sy) / denom


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--duration", type=float, required=True,
                   help="Capture window in seconds")
    p.add_argument("--expected-rate", type=float, default=None,
                   help="Expected yaw rate in deg/s (sign matters); enables pass/fail")
    p.add_argument("--tolerance", type=float, default=5.0,
                   help="Pass band: |measured - expected| / |expected| (percent)")
    p.add_argument("--rate-hz", type=float, default=100.0,
                   help="Polling rate for /imu")
    p.add_argument("--csv", type=str, default=None,
                   help="Optional CSV path for raw samples")
    p.add_argument("--snapshots-dir", type=str, default=None,
                   help="Where before/after top-down PNGs are written")
    p.add_argument("--snapshot-range", type=float, default=10.0,
                   help="Top-down view range (m) passed to /snapshot/top_down")
    p.add_argument("--countdown", type=int, default=3,
                   help="Seconds to wait before starting capture")
    args = p.parse_args()

    base = f"http://{args.host}:{args.port}"
    if args.snapshots_dir is None:
        snap_dir = Path(__file__).resolve().parents[1] / "rotation_test_captures"
    else:
        snap_dir = Path(args.snapshots_dir)
    snap_dir.mkdir(parents=True, exist_ok=True)

    # Sanity check
    try:
        with urllib.request.urlopen(f"{base}/healthz", timeout=2.0) as r:
            print(f"connected: /healthz -> {r.read().decode().strip()}")
    except Exception as e:
        print(f"ERROR: cannot reach pipeline at {base}: {e}", file=sys.stderr)
        return 2

    for k in range(args.countdown, 0, -1):
        print(f"  starting in {k}...", flush=True)
        time.sleep(1.0)

    print("[snapshot] before")
    try:
        _save(f"{base}/snapshot/top_down?range={args.snapshot_range}",
              snap_dir / "before_top_down.png")
        _save(f"{base}/snapshot/camera", snap_dir / "before_camera.png")
    except Exception as e:
        print(f"  snapshot warning: {e}")

    print(f"[capture] {args.duration:.2f} s @ ~{args.rate_hz:.0f} Hz")
    samples: List[Tuple[float, float]] = []        # (t, yaw_deg raw)
    gyro_z_samples: List[Tuple[float, float]] = []  # (t, gyro_z rad/s)
    period = 1.0 / args.rate_hz
    t_end = time.monotonic() + args.duration
    next_t = time.monotonic()
    while time.monotonic() < t_end:
        try:
            d = _get_json(f"{base}/imu")
            t = float(d.get("timestamp") or time.time())
            y = float(d.get("yaw_deg") or 0.0)
            samples.append((t, y))
            gyro = d.get("gyro_rads")
            if gyro is not None and len(gyro) >= 3:
                gyro_z_samples.append((t, float(gyro[2])))
        except Exception as e:
            print(f"  poll error: {e}")
        next_t += period
        sleep_for = next_t - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)

    print(f"[capture] collected {len(samples)} yaw samples, {len(gyro_z_samples)} gyro samples")

    print("[snapshot] after")
    try:
        _save(f"{base}/snapshot/top_down?range={args.snapshot_range}",
              snap_dir / "after_top_down.png")
        _save(f"{base}/snapshot/camera", snap_dir / "after_camera.png")
    except Exception as e:
        print(f"  snapshot warning: {e}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "yaw_deg_raw", "gyro_z_rads"])
            gz_by_t = dict(gyro_z_samples)
            for t, y in samples:
                w.writerow([f"{t:.6f}", f"{y:.4f}", f"{gz_by_t.get(t, ''): }"])
        print(f"[csv] wrote {args.csv}")

    # ---- analysis ----
    unwrapped = _unwrap_yaw(samples)
    yaw_rate_fit = _linear_fit_slope(unwrapped)  # deg/s
    if gyro_z_samples:
        mean_gz_rads = sum(g for _, g in gyro_z_samples) / len(gyro_z_samples)
        # Gyro Z is body-frame; positive Z is up for the WitMotion in NED-ish.
        # Convert to deg/s. Sign convention may differ from yaw fit; report both.
        mean_gz_degs = math.degrees(mean_gz_rads)
    else:
        mean_gz_degs = float("nan")

    if unwrapped:
        total_yaw_change = unwrapped[-1][1] - unwrapped[0][1]
        elapsed = unwrapped[-1][0] - unwrapped[0][0]
    else:
        total_yaw_change = float("nan"); elapsed = float("nan")

    print()
    print("=" * 60)
    print(f"  duration             : {elapsed:.3f} s")
    print(f"  total yaw delta      : {total_yaw_change:+.3f} deg")
    print(f"  yaw rate (linear fit): {yaw_rate_fit:+.3f} deg/s")
    print(f"  yaw rate (mean gyro_z): {mean_gz_degs:+.3f} deg/s")
    print("=" * 60)

    rc = 0
    if args.expected_rate is not None:
        exp = args.expected_rate
        err_fit_pct = abs(yaw_rate_fit - exp) / max(abs(exp), 1e-6) * 100.0
        err_gyro_pct = abs(mean_gz_degs - exp) / max(abs(exp), 1e-6) * 100.0
        print(f"  expected             : {exp:+.3f} deg/s   (tolerance ±{args.tolerance:.1f}%)")
        print(f"  error (fit)          : {err_fit_pct:5.2f}%   -> {'PASS' if err_fit_pct <= args.tolerance else 'FAIL'}")
        print(f"  error (gyro)         : {err_gyro_pct:5.2f}%   -> {'PASS' if err_gyro_pct <= args.tolerance else 'FAIL'}")
        if err_fit_pct > args.tolerance and err_gyro_pct > args.tolerance:
            rc = 1
    print(f"  snapshots            : {snap_dir.resolve()}/")
    return rc


if __name__ == "__main__":
    sys.exit(main())
