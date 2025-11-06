from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import logging
from enum import Enum


# class RewardConfig(BaseModel):?
decay_rate: Optional[float] = -0.1  # Adjust this value to control the decay rate, higher values panelise more
scaling_factor: Optional[int] = 5 # Max reward from distance

bonus_goal_range: Optional[int] = 10  # mm
bonus_reward: Optional[float] = 10  # Reward for reaching the goal

precision_tolerance: Optional[int] = 8
noise_tolerance: Optional[int] = 5  # degrees

class REWARD_CONSTANTS(Enum):
    MAX_REWARD = 10
    MIN_REWARD = -50

def _reward_function(previous_state, current_state):
    yaw_before = previous_state[-2]
    yaw_after = current_state[-2]
    target_yaw = current_state[-1]
    # print(yaw_before)
    # print(yaw_after)


    if yaw_before is None:
        logging.debug("Start Marker Pose is None")
        return 0, True

    if yaw_after is None:
        logging.debug("Final Marker Pose is None")
        return 0, True

    done = False

    yaw_before_rounded = round(yaw_before)
    yaw_after_rounded = round(yaw_after)

    goal_difference_before = rotation_min_difference(
        target_yaw, yaw_before_rounded
    )
    goal_difference_after = rotation_min_difference(
        target_yaw, yaw_after_rounded
    )
    print("Goal difference:", goal_difference_after)

    # Current yaw_before might not equal yaw_after in prev step, hence need to check before as well to see if it has reached the goal already
    if goal_difference_before <= precision_tolerance:
        logging.info("----------Reached the Goal!----------")
        logging.debug(
            "Warning: Yaw before in current step not equal to Yaw after in prev step"
        )
        reward = bonus_reward
        # done = True
        return reward, False

    delta_changes = rotation_min_difference(target_yaw, yaw_before_rounded) - rotation_min_difference(target_yaw, yaw_after_rounded)

    if -noise_tolerance <= delta_changes <= noise_tolerance:
        reward = -1
    else:
        raw_reward = delta_changes / rotation_min_difference(
            target_yaw, yaw_before_rounded
        )
        if raw_reward >= REWARD_CONSTANTS.MAX_REWARD.value:
            reward = REWARD_CONSTANTS.MAX_REWARD.value
        elif raw_reward <= REWARD_CONSTANTS.MIN_REWARD.value:
            reward = REWARD_CONSTANTS.MIN_REWARD.value
        else:
            reward = raw_reward

    distance_reward_scaler = 1
    distance_reward = distance_reward_scaler*((180-goal_difference_after)/180)

    reward += distance_reward

    if goal_difference_after <= precision_tolerance:
        logging.info("----------Reached the Goal!----------")
        reward += bonus_reward
        # done = True

    return reward, False

def rotation_min_difference(a, b):
    """
    Formula that calculates the minimum difference between two angles.

    Args:
    a: First angle.
    b: Second angle.

    Returns:
        float: The minimum angular difference.
    """
    return min(abs(a - b), (360 + min(a, b) - max(a, b)))