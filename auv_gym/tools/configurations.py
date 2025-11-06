from typing import Optional
from cares_reinforcement_learning.util import configurations as cares_cfg


class AUVEnvironmentConfig(cares_cfg.SubscriptableClass):
    camera_id: int
    camera_matrix: str
    camera_distortion: str
    is_inverted: Optional[bool] = False

    # actions per episode
    episode_horizon: Optional[int] = 50

    # Time steps (secs) between action updates in velocity mode for dynamic sleep
    step_time_period: Optional[float] = 0.2  # secs

    # Aruco or STAG Marker size in mm
    marker_size: Optional[int] = 30  # mm

    gripper_marker_size: Optional[int] = 30  # mm
    object_marker_size: Optional[int] = 45  # mm

    cube_ids: Optional[list] = [1,2,3,4,5,6]

    # Tolerance in position error for object being at goal
    noise_tolerance: Optional[int] = 5  # mm or degrees

    # TODO make a string enum
    goal_selection_method: Optional[int] = 0

    is_debug = False

    # For when ssh to train, display can be turned off 
    display: Optional[bool] = True