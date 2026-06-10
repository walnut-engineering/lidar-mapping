"""
Command-line entry points for the lidar-mapping kit.

Usage examples (after ``pip install -e .``):

    lidar-record   --duration 60 --out ./recordings/run1
    lidar-playback --in  ./recordings/run1 --speed 2.0
    lidar-map      --in  ./recordings/run1 --voxel 0.1 --save map.pcd
"""
