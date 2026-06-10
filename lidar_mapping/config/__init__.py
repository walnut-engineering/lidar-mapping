"""
Central configuration system for the lidar-mapping kit.

Supports loading from JSON, TOML, or YAML (PyYAML optional) and exposes
strongly-typed dataclasses for each subsystem.  A default `kit_config()`
returns sane production values for the Pi 5 / VLP-16 / WTGAHRS2 / Pi-Camera /
7" touchscreen build.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import tomllib  # py3.11+
    _TOML_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TOML_AVAILABLE = False

try:
    import yaml  # type: ignore
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Subsystem configs
# ---------------------------------------------------------------------------

@dataclass
class LidarConfig:
    """VLP-16 driver settings."""
    host: str = "0.0.0.0"
    data_port: int = 2368
    position_port: int = 8308
    frame_buffer: int = 5
    enabled: bool = True


@dataclass
class IMUConfig:
    """IMU driver settings (default = WitMotion WTGAHRS2)."""
    driver: str = "witmotion"   # witmotion | mpu9250 | bno055 | lsm9ds1 | serial_ahrs
    port: str = "/dev/ttyUSB0"  # serial port on Pi
    baud: int = 115200
    rate_hz: int = 100
    i2c_bus: int = 1            # used by I²C drivers
    enabled: bool = True


@dataclass
class CameraConfig:
    """Pi Camera capture settings."""
    device: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    enabled: bool = True


@dataclass
class MapperConfig:
    """Mapper / ICP / preprocessing settings."""
    voxel_size: float = 0.1
    min_range: float = 0.5
    max_range: float = 80.0
    z_min: float = -3.0
    z_max: float = 20.0
    remove_ground: bool = False
    icp_max_correspondence_distance: Optional[float] = None
    max_map_points: int = 2_000_000


@dataclass
class UIConfig:
    """7" touchscreen UI."""
    width: int = 800
    height: int = 480
    fullscreen: bool = False
    fps: int = 15


@dataclass
class StorageConfig:
    """Where to write recordings, maps, screenshots."""
    base_dir: str = "./recordings"
    map_dir: str = "./maps"
    screenshot_dir: str = "./screenshots"


@dataclass
class KitConfig:
    """Top-level kit configuration."""
    lidar: LidarConfig = field(default_factory=LidarConfig)
    imu: IMUConfig = field(default_factory=IMUConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    mapper: MapperConfig = field(default_factory=MapperConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Save to .json, .toml, or .yaml/.yml."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        data = self.to_dict()
        if suffix == ".json":
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        elif suffix == ".toml":
            path.write_text(_to_toml(data), encoding="utf-8")
        elif suffix in (".yaml", ".yml"):
            if not _YAML_AVAILABLE:
                raise ImportError("PyYAML is required to save YAML.")
            path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        else:
            raise ValueError(f"Unsupported config format: {suffix}")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def kit_config() -> KitConfig:
    """Return a default-populated KitConfig for the Pi5 build."""
    return KitConfig()


def load_config(path: str | Path) -> KitConfig:
    """
    Load a kit configuration from JSON, TOML, or YAML.

    Unknown keys are ignored; missing keys take their default values.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        data = json.loads(text)
    elif suffix == ".toml":
        if not _TOML_AVAILABLE:
            raise ImportError("tomllib (py3.11+) required for TOML config.")
        data = tomllib.loads(text)
    elif suffix in (".yaml", ".yml"):
        if not _YAML_AVAILABLE:
            raise ImportError("PyYAML required to load YAML config.")
        data = yaml.safe_load(text) or {}
    else:
        raise ValueError(f"Unsupported config format: {suffix}")
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")
    return _from_dict(KitConfig, data)


def _from_dict(cls, data: Dict[str, Any]):
    """Recursively build a dataclass from a dict, ignoring unknown keys."""
    if not is_dataclass(cls):
        return data
    kwargs: Dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    for k, v in data.items():
        if k not in known:
            continue
        f = known[k]
        if is_dataclass(f.type) and isinstance(v, dict):
            kwargs[k] = _from_dict(f.type, v)
        elif isinstance(v, dict) and isinstance(f.default_factory, type):
            # nested dataclass with default_factory
            kwargs[k] = _from_dict(f.default_factory, v)
        else:
            # Try nested if default_factory produces a dataclass
            try:
                default_instance = f.default_factory()  # type: ignore
                if is_dataclass(default_instance) and isinstance(v, dict):
                    kwargs[k] = _from_dict(type(default_instance), v)
                    continue
            except TypeError:
                pass
            kwargs[k] = v
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Tiny TOML writer (avoids extra dependency)
# ---------------------------------------------------------------------------

def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if v is None:
        # TOML has no null; write empty string as a stand-in
        return '""'
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"Cannot serialise {type(v).__name__} to TOML")


def _to_toml(data: Dict[str, Any]) -> str:
    """Minimal TOML writer: top-level mapping of nested dicts → [tables]."""
    out: list[str] = []
    # Scalars first (none expected, but be safe)
    for k, v in data.items():
        if not isinstance(v, dict):
            out.append(f"{k} = {_toml_value(v)}")
    if out:
        out.append("")
    for section, body in data.items():
        if not isinstance(body, dict):
            continue
        out.append(f"[{section}]")
        for k, v in body.items():
            out.append(f"{k} = {_toml_value(v)}")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Environment-aware default path lookup
# ---------------------------------------------------------------------------

def find_default_config() -> Optional[Path]:
    """
    Locate a config file in priority order:

    1. ``$LIDAR_MAPPING_CONFIG`` env var
    2. ``./lidar_mapping.toml`` / ``.yaml`` / ``.json`` in CWD
    3. ``~/.config/lidar_mapping/config.toml`` etc.
    """
    env = os.environ.get("LIDAR_MAPPING_CONFIG")
    if env and Path(env).is_file():
        return Path(env)
    for suffix in (".toml", ".yaml", ".yml", ".json"):
        candidate = Path.cwd() / f"lidar_mapping{suffix}"
        if candidate.is_file():
            return candidate
    user_dir = Path.home() / ".config" / "lidar_mapping"
    for suffix in (".toml", ".yaml", ".yml", ".json"):
        candidate = user_dir / f"config{suffix}"
        if candidate.is_file():
            return candidate
    return None
