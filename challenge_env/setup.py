from setuptools import setup

setup(
    name="challenge_env",
    version="0.1",
    author="Lennart Röstel, Felix Kroll, Johannes Pitz",
    packages=["challenge_env"],
    install_requires=[
        "requests",
        "numpy",
        "torch",
        "mujoco",
        "gymnasium[mujoco]",
        "imageio",
        "opencv-python",
        "dm_control",
        "scipy",
    ],
)
