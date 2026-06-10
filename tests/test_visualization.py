"""
Tests for the visualisation module.

Interactive functions (``show_point_cloud``, ``show_map_with_trajectory``)
are skipped — they open GUI windows and block.

``save_screenshot`` is tested with Open3D's headless off-screen renderer;
this works without a display.
"""

from __future__ import annotations

import sys
import pytest
import numpy as np
from pathlib import Path

try:
    import open3d as o3d
    _O3D_AVAILABLE = True
except ImportError:
    _O3D_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _O3D_AVAILABLE, reason="open3d not installed"
)


# ===========================================================================
# create_colored_cloud
# ===========================================================================

class TestCreateColoredCloud:
    """Unit tests for ``create_colored_cloud``."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from lidar_mapping.visualization import create_colored_cloud
        self.create_colored_cloud = create_colored_cloud

    def _random_cloud(self, n: int = 500, cols: int = 3, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.uniform(-5.0, 5.0, (n, cols)).astype(np.float32)

    # --- z coloring ---

    def test_z_coloring_shape(self):
        pts = self._random_cloud(500, cols=3)
        pcd = self.create_colored_cloud(pts, color_by="z")
        assert len(pcd.points) == 500
        assert pcd.has_colors()
        colors = np.asarray(pcd.colors)
        assert colors.shape == (500, 3)

    def test_z_coloring_values_in_range(self):
        pts = self._random_cloud(500, cols=3)
        pcd = self.create_colored_cloud(pts, color_by="z")
        colors = np.asarray(pcd.colors)
        assert np.all(colors >= 0.0)
        assert np.all(colors <= 1.0)

    def test_z_flat_cloud_no_error(self):
        """All same z → no division by zero, uniform colour accepted."""
        pts = np.zeros((100, 3), dtype=np.float32)
        pcd = self.create_colored_cloud(pts, color_by="z")
        assert len(pcd.points) == 100
        assert pcd.has_colors()

    def test_z_coloring_with_intensity_column(self):
        """(N, 4) input with color_by='z' should still work (uses cols 0-2)."""
        pts = self._random_cloud(300, cols=4)
        pcd = self.create_colored_cloud(pts, color_by="z")
        assert len(pcd.points) == 300

    # --- intensity coloring ---

    def test_intensity_coloring_requires_4_columns(self):
        pts = self._random_cloud(200, cols=3)
        with pytest.raises(ValueError, match="intensity_column"):
            self.create_colored_cloud(pts, color_by="intensity")

    def test_intensity_coloring_shape(self):
        pts = self._random_cloud(300, cols=4)
        pcd = self.create_colored_cloud(pts, color_by="intensity")
        colors = np.asarray(pcd.colors)
        assert colors.shape == (300, 3)
        assert np.all(colors >= 0.0) and np.all(colors <= 1.0)

    def test_intensity_uniform_values_no_crash(self):
        """Constant intensity column (range=0) → no division by zero."""
        pts = np.ones((100, 4), dtype=np.float32)
        pcd = self.create_colored_cloud(pts, color_by="intensity")
        assert len(pcd.points) == 100

    # --- uniform coloring ---

    def test_uniform_coloring_all_same(self):
        pts = self._random_cloud(200)
        pcd = self.create_colored_cloud(pts, color_by="uniform")
        colors = np.asarray(pcd.colors)
        # All rows should be equal
        assert np.allclose(colors, colors[0])

    def test_uniform_color_value_is_grey(self):
        pts = self._random_cloud(50)
        pcd = self.create_colored_cloud(pts, color_by="uniform")
        colors = np.asarray(pcd.colors)
        # Grey: R == G == B
        np.testing.assert_allclose(colors[:, 0], colors[:, 1])
        np.testing.assert_allclose(colors[:, 1], colors[:, 2])

    # --- invalid color_by ---

    def test_invalid_color_by_raises(self):
        pts = self._random_cloud(50)
        with pytest.raises(ValueError, match="color_by"):
            self.create_colored_cloud(pts, color_by="rainbow")

    # --- empty cloud ---

    def test_empty_cloud(self):
        pts = np.zeros((0, 3), dtype=np.float32)
        pcd = self.create_colored_cloud(pts, color_by="uniform")
        assert len(pcd.points) == 0

    # --- xyz values preserved ---

    def test_xyz_preserved(self):
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        pcd = self.create_colored_cloud(pts, color_by="uniform")
        out = np.asarray(pcd.points)
        np.testing.assert_allclose(out, pts, atol=1e-9)


# ===========================================================================
# save_screenshot (headless)
# ===========================================================================

@pytest.mark.skipif(
    not _O3D_AVAILABLE, reason="open3d not installed"
)
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="OffscreenRenderer requires EGL headless; not supported on Windows",
)
class TestSaveScreenshot:
    """Tests for the headless off-screen renderer."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from lidar_mapping.visualization import save_screenshot
        self.save_screenshot = save_screenshot

    def _cloud(self, n: int = 1000) -> np.ndarray:
        rng = np.random.default_rng(7)
        return rng.uniform(-3.0, 3.0, (n, 3)).astype(np.float32)

    def test_png_created(self, tmp_path: Path):
        """save_screenshot must write a non-empty PNG."""
        out = tmp_path / "cloud.png"
        self.save_screenshot(self._cloud(), out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_custom_size(self, tmp_path: Path):
        out = tmp_path / "small.png"
        self.save_screenshot(self._cloud(), out, width=320, height=240)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_z_color_scheme(self, tmp_path: Path):
        out = tmp_path / "z_colored.png"
        self.save_screenshot(self._cloud(), out, color_by="z")
        assert out.exists()

    def test_uniform_color_scheme(self, tmp_path: Path):
        out = tmp_path / "uniform.png"
        self.save_screenshot(self._cloud(), out, color_by="uniform")
        assert out.exists()

    def test_string_path_accepted(self, tmp_path: Path):
        out = str(tmp_path / "str_path.png")
        self.save_screenshot(self._cloud(), out)
        from pathlib import Path as _P
        assert _P(out).exists()

    def test_parent_dir_created(self, tmp_path: Path):
        out = tmp_path / "subdir" / "nested" / "cloud.png"
        # save_screenshot does not create parent dirs — we do it here
        out.parent.mkdir(parents=True, exist_ok=True)
        self.save_screenshot(self._cloud(), out)
        assert out.exists()


# ===========================================================================
# require_o3d guard
# ===========================================================================

class TestRequireO3DGuard:
    """Verify that the guard raises ImportError when open3d is absent."""

    def test_guard_raises_when_o3d_missing(self, monkeypatch):
        import lidar_mapping.visualization.viewer as viewer_mod
        monkeypatch.setattr(viewer_mod, "_O3D_AVAILABLE", False)
        with pytest.raises(ImportError, match="open3d"):
            viewer_mod._require_o3d()

    def test_guard_passes_when_o3d_present(self, monkeypatch):
        import lidar_mapping.visualization.viewer as viewer_mod
        monkeypatch.setattr(viewer_mod, "_O3D_AVAILABLE", True)
        viewer_mod._require_o3d()   # must not raise
