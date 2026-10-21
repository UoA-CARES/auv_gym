"""Unified AUV adapter facade."""

from typing import Any

from auv_gym.robot_adapter import RobotAdapter
from auv_gym.vision_adapter import VisionAdapter


class AUVAdapter:
    """Expose robot control and vision through one backend object."""

    def __init__(self, robot: RobotAdapter, vision: VisionAdapter) -> None:
        self.robot = robot
        self.vision = vision

    @property
    def camera(self) -> Any:
        return self.vision.camera

    def create_marker_detector(
        self, marker_size: float, library_hd: int, error_correction: int = -1
    ) -> Any:
        return self.vision.create_marker_detector(
            marker_size, library_hd, error_correction
        )

    def close(self) -> None:
        try:
            self.robot.close()
        finally:
            self.vision.close()

    def __getattr__(self, name: str) -> Any:
        """Keep robot control methods available on the unified facade."""
        return getattr(self.robot, name)
