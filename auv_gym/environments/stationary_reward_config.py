from pydantic import BaseModel
from typing import List, Optional
import numpy as np

# class RewardConfig(BaseModel):?
decay_rate: Optional[float] = -0.1  # Adjust this value to control the decay rate, higher values panelise more
scaling_factor: Optional[int] = 5 # Max reward from distance

bonus_goal_range: Optional[int] = 10  # mm
bonus_reward: Optional[float] = 5

def _reward_function(previous_state, current_state):
    # Unpack relevant coordinates from current_state
    gripper_left_x = current_state["gripper_poses"][7]["position"][0] 
    gripper_left_y = current_state["gripper_poses"][7]["position"][1] 
    
    gripper_right_x = current_state["gripper_poses"][8]["position"][0] 
    gripper_right_y = current_state["gripper_poses"][8]["position"][1] 

    object_x = current_state["object_pose"]["position"][0]
    object_y = current_state["object_pose"]["position"][1]

    # Compute gripper center and object position
    gripper_center = np.array([(gripper_left_x + gripper_right_x) / 2,
                    (gripper_left_y + gripper_right_y) / 2])
    object_pos = np.array([object_x, object_y])

    # Calculate Euclidean distance between gripper center and object
    distance = np.linalg.norm(gripper_center - object_pos)

    # Expponential decayin reward based on distance
    reward = np.exp(decay_rate * distance) * scaling_factor

    # Bonus for being within a certain range of the gripper
    if distance < bonus_goal_range:
        reward += bonus_reward

    # Check if the object is between the gripper markers
    # if (gripper_left[0] < object_position[0] < gripper_right[0]):
    #     reward = 1.0  # Reward for object being between gripper markers

    print(reward)

    return reward, False


def _reward_function_pixel(previous_environment_info, current_environment_info):
    # object pixel distance to a needed position
    pass