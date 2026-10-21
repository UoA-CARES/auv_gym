"""Robot/task bridge for CARES-compatible underwater environments."""

from auv_gym.auv_adapter import AUVAdapter
from auv_gym.robot_adapter import RobotAdapter
from auv_gym.vision_adapter import VisionAdapter

__all__ = ["AUVAdapter", "RobotAdapter", "VisionAdapter"]
