"""
Snapshot rendering helpers — produce PNG bytes from live fusion state.

These functions are called by the observability HTTP server when an agent
or human requests a view. They never touch hardware; they read from
``FusionState`` and render with numpy + PIL (no OpenGL context required).

Three views are provided:
  * ``camera_png``       — latest camera frame (with optional overlay).
  * ``top_down_png``     — XY projection of latest LiDAR frame + trajectory.
  * ``composite_png``    — side-by-side camera + top-down for at-a-glance.
"""

from __future__ import annotations

import io
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lidar_mapping.observability.state import FusionState


def _np_to_png_bytes(arr_rgb: np.ndarray) -> bytes:
    img = Image.fromarray(np.ascontiguousarray(arr_rgb))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _placeholder(width: int, height: int, message: str) -> bytes:
    arr = np.full((height, width, 3), 32, dtype=np.uint8)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((10, height // 2 - 10), message, fill=(220, 220, 220), font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def camera_png(state: FusionState, prefer_overlay: bool = True) -> bytes:
    with state.lock:
        bgr = state.latest_camera_overlay_bgr if prefer_overlay else None
        if bgr is None:
            bgr = state.latest_camera_bgr
        if bgr is None:
            return _placeholder(640, 480, "no camera frame yet")
        rgb = bgr[..., ::-1].copy()  # BGR -> RGB
    return _np_to_png_bytes(rgb)


def top_down_png(
    state: FusionState,
    width: int = 720,
    height: int = 720,
    range_m: float = 30.0,
) -> bytes:
    """Render an XY (top-down) projection of the latest LiDAR cloud +
    trajectory. The map is centered on the current pose's XY."""
    with state.lock:
        cloud = state.latest_cloud
        cloud_copy = cloud.copy() if cloud is not None else None
        pose = state.current_pose.copy()
    traj = state.trajectory_snapshot(max_points=2000)

    arr = np.full((height, width, 3), 18, dtype=np.uint8)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    cx, cy = width // 2, height // 2
    scale = (min(width, height) / 2.0) / range_m  # pixels per meter

    # Grid (every 5 m)
    for r_m in range(5, int(range_m) + 1, 5):
        r_px = int(r_m * scale)
        draw.ellipse([cx - r_px, cy - r_px, cx + r_px, cy + r_px],
                     outline=(50, 50, 50))

    # Cardinal axes
    draw.line([cx, 0, cx, height], fill=(60, 60, 60))
    draw.line([0, cy, width, cy], fill=(60, 60, 60))

    origin_xy = pose[:2, 3]

    # Point cloud (transformed into world via current pose if cloud is body-frame)
    if cloud_copy is not None and len(cloud_copy) > 0:
        xyz = cloud_copy[:, :3]
        intensity = cloud_copy[:, 3] if cloud_copy.shape[1] >= 4 else None
        # apply current pose so plotted points are world-frame
        world = xyz @ pose[:3, :3].T + pose[:3, 3]
        # filter to view radius
        rel = world[:, :2] - origin_xy
        mask = np.abs(rel[:, 0]) < range_m
        mask &= np.abs(rel[:, 1]) < range_m
        rel = rel[mask]
        if intensity is not None:
            intensity = intensity[mask]
        px = (cx + rel[:, 0] * scale).astype(np.int32)
        py = (cy - rel[:, 1] * scale).astype(np.int32)
        # vectorized scatter via numpy indexing on the array view
        canvas = np.array(img)
        in_bounds = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        px = px[in_bounds]; py = py[in_bounds]
        if intensity is not None:
            inten = np.clip(intensity[in_bounds] / 255.0, 0, 1)
            colors = np.stack([
                (inten * 255).astype(np.uint8),
                ((1 - inten) * 200 + 55).astype(np.uint8),
                np.full_like(px, 180, dtype=np.uint8),
            ], axis=1)
        else:
            colors = np.full((len(px), 3), 200, dtype=np.uint8)
        canvas[py, px] = colors
        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)

    # Trajectory (world-frame)
    if traj:
        pts = []
        for _, T in traj:
            rel = T[:2, 3] - origin_xy
            pts.append((cx + rel[0] * scale, cy - rel[1] * scale))
        if len(pts) >= 2:
            draw.line(pts, fill=(80, 220, 80), width=2)

    # Current pose marker
    draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(0, 255, 0))
    # Heading line from yaw
    yaw_rad = np.arctan2(pose[1, 0], pose[0, 0])
    hx = cx + int(np.cos(yaw_rad) * 20)
    hy = cy - int(np.sin(yaw_rad) * 20)
    draw.line([cx, cy, hx, hy], fill=(0, 255, 0), width=2)

    # HUD text
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    hud = (
        f"pos=({origin_xy[0]:+.2f},{origin_xy[1]:+.2f}) "
        f"yaw={np.degrees(yaw_rad):+.1f}deg  "
        f"range={range_m:.0f}m"
    )
    draw.text((6, 4), hud, fill=(220, 220, 220), font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def composite_png(state: FusionState) -> bytes:
    """Side-by-side composite: camera (left) + top-down (right)."""
    cam_bytes = camera_png(state)
    top_bytes = top_down_png(state, width=540, height=540)
    cam = Image.open(io.BytesIO(cam_bytes)).convert("RGB")
    top = Image.open(io.BytesIO(top_bytes)).convert("RGB")
    H = max(cam.height, top.height)
    cam_w = int(cam.width * (H / cam.height))
    top_w = int(top.width * (H / top.height))
    cam = cam.resize((cam_w, H))
    top = top.resize((top_w, H))
    out = Image.new("RGB", (cam_w + top_w, H), (10, 10, 10))
    out.paste(cam, (0, 0))
    out.paste(top, (cam_w, 0))
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()
