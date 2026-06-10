"""
Data recording and playback for offline development.

Classes
-------
:class:`VLP16Recorder`
    Record raw VLP-16 UDP packets (+ timestamps) to a binary file.
:class:`IMURecorder`
    Record :class:`~lidar_mapping.sensors.imu.IMUReading` objects to a
    compressed NumPy archive.
"""

from lidar_mapping.io.recorder import IMURecorder, VLP16Recorder
from lidar_mapping.io.playback import IMUPlayback, VLP16Playback

__all__ = ["VLP16Recorder", "IMURecorder", "VLP16Playback", "IMUPlayback"]
