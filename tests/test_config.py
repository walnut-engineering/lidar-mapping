"""Tests for the config subsystem."""

import json
import pytest

from lidar_mapping.config import (
    KitConfig, kit_config, load_config,
    _TOML_AVAILABLE, _YAML_AVAILABLE,
)


class TestDefaults:
    def test_defaults_populate(self):
        cfg = kit_config()
        assert cfg.lidar.data_port == 2368
        assert cfg.imu.driver == "witmotion"
        assert cfg.ui.width == 800
        assert cfg.ui.height == 480
        assert cfg.mapper.voxel_size == 0.1


class TestJsonRoundTrip:
    def test_save_and_load(self, tmp_path):
        cfg = kit_config()
        cfg.mapper.voxel_size = 0.25
        cfg.imu.driver = "mpu9250"
        p = tmp_path / "kit.json"
        cfg.save(p)
        loaded = load_config(p)
        assert loaded.mapper.voxel_size == 0.25
        assert loaded.imu.driver == "mpu9250"
        # untouched defaults preserved
        assert loaded.lidar.data_port == 2368

    def test_unknown_keys_ignored(self, tmp_path):
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps({
            "lidar": {"data_port": 9999, "bogus_field": 1},
            "unknown_section": {},
        }))
        loaded = load_config(p)
        assert loaded.lidar.data_port == 9999


@pytest.mark.skipif(not _TOML_AVAILABLE, reason="tomllib not available")
class TestTomlRoundTrip:
    def test_save_and_load(self, tmp_path):
        cfg = kit_config()
        cfg.ui.fullscreen = True
        cfg.lidar.host = "192.168.1.50"
        p = tmp_path / "kit.toml"
        cfg.save(p)
        loaded = load_config(p)
        assert loaded.ui.fullscreen is True
        assert loaded.lidar.host == "192.168.1.50"


@pytest.mark.skipif(not _YAML_AVAILABLE, reason="PyYAML not installed")
class TestYamlRoundTrip:
    def test_save_and_load(self, tmp_path):
        cfg = kit_config()
        cfg.imu.rate_hz = 200
        p = tmp_path / "kit.yaml"
        cfg.save(p)
        loaded = load_config(p)
        assert loaded.imu.rate_hz == 200


class TestErrors:
    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/no/such/file.json")

    def test_unsupported_extension(self, tmp_path):
        p = tmp_path / "x.cfg"
        p.write_text("anything")
        with pytest.raises(ValueError):
            load_config(p)

    def test_unsupported_save_format(self, tmp_path):
        with pytest.raises(ValueError):
            kit_config().save(tmp_path / "x.cfg")
