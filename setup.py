from setuptools import find_packages, setup

setup(
    name="lidar-mapping",
    version="0.1.0",
    description="VLP-16 + IMU + Camera fusion mapping on Orange Pi 5",
    python_requires=">=3.10",
    packages=find_packages(include=["lidar_mapping*", "apps*"]),
    install_requires=[
        "numpy>=1.23",
        "opencv-python>=4.8",
        "pyserial>=3.5",
        "scipy>=1.10",
        "vispy>=0.14",
        "glfw>=2.6",
        "Pillow>=10",
        "Flask>=3.0",
        "PyYAML>=6.0",
        "open3d>=0.18; platform_machine != 'aarch64'",
    ],
    extras_require={
        "i2c": ["smbus2>=0.4"],
        "dev": ["pytest>=7"],
    },
)
