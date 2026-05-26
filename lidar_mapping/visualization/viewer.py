import numpy as np

try:
    from vispy import scene, app
except ImportError:
    scene = None
    app = None

class LivePointCloudViewer:
    """
    Lightweight, hardware-accelerated point cloud viewer using VisPy.
    
    Safe for use on ARM SBCs (like Orange Pi 5 / RK3588) with Mali GPUs
    where heavier tools like Open3D Visualizer may crash.
    """
    
    def __init__(self, title: str = "Live Point Cloud", width: int = 1280, height: int = 720):
        if scene is None:
            raise ImportError("vispy is required for the viewer. Install it with: pip install vispy glfw")
            
        self.canvas = scene.SceneCanvas(keys='interactive', show=True, size=(width, height), title=title)
        
        # Grid layout for split screen
        grid = self.canvas.central_widget.add_grid()
        
        # 3D Point Cloud View (left side)
        self.view = grid.add_view(row=0, col=0, col_span=2)
        self.view.camera = scene.TurntableCamera(up='z', distance=20.0, azimuth=-45, elevation=30)
        self.scatter = scene.visuals.Markers()
        self.view.add(self.scatter)
        self.axis = scene.visuals.XYZAxis(parent=self.view.scene)

        # 2D Camera View (right side, optional)
        self.image_view = grid.add_view(row=0, col=2, col_span=1)
        self.image_view.camera = scene.PanZoomCamera(aspect=1)
        self.image_visual = scene.visuals.Image(parent=self.image_view.scene)
        
        self.update_callback = None
        self.timer = app.Timer(interval=0.033, connect=self._on_timer, start=False)
        
    def set_callback(self, callback_fn):
        """
        Set a function to be called periodically to fetch new data.
        The callback should return either:
        - a numpy array of shape (N, 3) or (N, 4) (points only)
        - a tuple: (points_array, image_array) where image_array is (H, W, 3) RGB
        """
        self.update_callback = callback_fn
        
    def _on_timer(self, event):
        if self.update_callback is None:
            return
            
        data = self.update_callback()
        if data is None:
            return
            
        if isinstance(data, tuple):
            points, image = data
        else:
            points = data
            image = None

        # Update Point Cloud
        if points is not None and len(points) > 0:
            xyz = points[:, 0:3]
            
            if points.shape[1] >= 4:
                # Colorize based on intensity
                intensity = points[:, 3]
                normalized = np.clip(intensity / 255.0, 0, 1)
                colors = np.zeros((len(points), 4), dtype=np.float32)
                colors[:, 0] = normalized          # R
                colors[:, 1] = 1.0 - normalized    # G
                colors[:, 2] = 0.8                 # B
                colors[:, 3] = 1.0                 # Alpha
            else:
                colors = np.array([[1.0, 1.0, 1.0, 1.0]] * len(points))
                
            self.scatter.set_data(xyz, edge_color=None, face_color=colors, size=2)

        # Update Image
        if image is not None:
            # Need to fix image orientation/colors for VisPy
            # Ensure it's rendered within the boundaries
            self.image_visual.set_data(image)
            self.image_view.camera.set_range(x=(0, image.shape[1]), y=(0, image.shape[0]))

    def start(self):
        """Start the visualization loop."""
        self.timer.start()
        app.run()
