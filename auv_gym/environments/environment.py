import logging
import random
from abc import ABC, abstractmethod
from functools import wraps
import numpy as np
import shutil
import os

from boxfish_lib.vision import utils
import cv2
import time
from auv_gym.tools.configurations import AUVEnvironmentConfig
import auv_gym.environments.reward_config as reward_config
from boxfish_lib.vision.camera import Camera
from scipy.spatial.transform import Rotation as R
from auv_gym.robot_adapter import BoxfishAdapter, RobotAdapter


def exception_handler(error_message):
    def decorator(function):
        @wraps(function)
        def wrapper(self, *args, **kwargs):
            try:
                return function(self, *args, **kwargs)
            except EnvironmentError as error:
                logging.error(
                    f"Environment for AUV: {error_message}"
                )
                raise EnvironmentError(
                    error.auv,
                    f"Environment for AUV: {error_message}",
                ) from error

        return wrapper

    return decorator


class EnvironmentError(IOError):
    def __init__(self, auv, message):
        self.auv = auv
        super().__init__(message)


class Environment(ABC):
    """
    Initialise the environment with the given configurations of the auv and object.

    Parameters:
    env_config: Configuration specific to the environment setup.
    auv_config: Configuration specific to the auv used.
    object_config: Configuration specific to the object in the environment.
    """

    def __init__(
        self,
        env_config: AUVEnvironmentConfig,
        auv_config=None,
        robot: RobotAdapter = None,
    ):
        self.env_config = env_config
        self.display = env_config.display

        if robot is None:
            if auv_config is None:
                raise ValueError("Provide either robot or auv_config")
            robot = BoxfishAdapter(auv_config)

        self.robot = robot
        # Keep the old attribute name for task and trainer compatibility.
        self.auv = robot
        
        self.is_inverted = env_config.is_inverted

        # camera?
        self.camera = Camera(
            env_config.camera_id, env_config.camera_matrix, env_config.camera_distortion
        )
        

        self.action_type = robot.action_type

        self.auv.home()
        self.step_counter = 0
        self.goal_reward = reward_config.bonus_reward
        print(self.goal_reward)
        self.episode_horizon = env_config.episode_horizon

        self.actions_taken = []

        self._reward_function = reward_config._reward_function

        self.noise_tolerance = env_config.noise_tolerance

        self.reward = 0

        self.max_action_value = np.array(robot.max_values)
        self.min_action_value = np.array(robot.min_values)
        self.control_actions = robot.control_actions

        
        # # Pose to normalise the other positions against - consider (0,0)
        # self.reference_marker_id = env_config.reference_marker_id
        
        # self.goal = []

        self.current_environment_info = {}
        self.previous_environment_info = {}
        self.previous_state = []

        self.chosen_marker_id = None
        self.base_log_dir = None

        self.auv.safety_check()

        self.starting = True


    # def grab_frame(self):
    #     frame = cv2.rotate(self.camera.get_frame(), cv2.ROTATE_180) if self.is_inverted else self.camera.get_frame()
    #     return frame

    def grab_rendered_frame(self):
        state = self._environment_info_to_state(self.current_environment_info)
        return self._render_environment(state, self.current_environment_info)


    @exception_handler("Environment failed to reset")
    def reset(self):
        """
        Resets the environment for a new episode.

        This method brings the AUV to its home position, resets the step counter,
        

        and resets the target servo position if necessary.

        Returns:
        list: The initial state of the environment.
        """
        if self.starting:
            self.starting = False
            if self.base_log_dir:
                self.save_extras(self.base_log_dir)

        self.step_counter = 0

        self.reward = 0
        self.success_counter = 0
        self.steps_to_success = 0

        self._reset()

        
        # logging.debug(f"New Goal Generated: {self.goal}")

        self.previous_environment_info = self.current_environment_info = (
            self._get_environment_info()
        )
        
        logging.debug(f"Env Info: {self.current_environment_info}")
        
        self.previous_state = state = self._environment_info_to_state(self.current_environment_info)
        logging.debug(f"State: {state}")
        
        return state
        

    def sample_action(self):
        action = []
        for i in range(0, len(self.auv.control_actions)):
            min_value = self.auv.min_values[i]
            max_value = self.auv.max_values[i]
            action.append(random.uniform(min_value, max_value))
        return action

    @exception_handler("Failed to step")
    def step(self, action):
        """
        Takes a step in the environment using the given action and returns the results.

        Parameters:
        action: The action to be executed.

        Returns:
        state: The new state after executing the action.
        reward: The reward obtained after the action.
        done: Whether the episode is done or not.
        truncated: Whether the step was truncated or not.
        """
        self.step_counter += 1
        # moving this over early so it doesn't overwrite marker detection issues
        truncated = self.step_counter >= self.episode_horizon

        self.auv.move([*action, 0, 0, 0])
        self.current_environment_info = self._get_environment_info()

        # if gripper marker not detected then need human to check
        if len(self.current_environment_info["gripper_poses"]) < 2:
            truncated = True
            self.auv.stop()  # Stop the AUV
            input("Gripper markers not detected, please check the camera and gripper markers. Press Enter to continue...")

        # if object not detected then finish the episode and reset
        if self.current_environment_info["object_pose"] is None:
            truncated = True
            self.auv.stop()  # Stop the AUV
            print("Object not detected") 


        state = self._environment_info_to_state(self.current_environment_info)
        image = self._render_environment(state, self.current_environment_info)

        if self.display:
            cv2.imshow("State Image", image)
            cv2.waitKey(10)

        reward, done = self._reward_function(
            self.previous_environment_info, self.current_environment_info
        )

        self.previous_environment_info = self.current_environment_info
        
        return state, reward, done, truncated, self.current_environment_info

    def denormalize(self, action_norm):

        # return action in gripper range [-min, +max] for each servo
        action_gripper = [0 for _ in range(0, len(action_norm))]
        min_value_in = -1
        max_value_in = 1
        for i in range(0, len(self.auv.control_actions)):
            servo_min_value = self.auv.min_values[i]
            servo_max_value = self.auv.max_values[i]
            action_gripper[i] = int(
                (action_norm[i] - min_value_in)
                * (servo_max_value - servo_min_value)
                / (max_value_in - min_value_in)
                + servo_min_value
            )
        return action_gripper

    def normalize(self, action_gripper):
        # return action in algorithm range [-1, +1]
        max_range_value = 1
        min_range_value = -1
        action_norm = [0 for _ in range(0, len(action_gripper))]
        for i in range(0, len(self.auv.control_actions)):
            servo_min_value = self.auv.min_values[i]
            servo_max_value = self.auv.max_values[i]
            action_norm[i] = (action_gripper[i] - servo_min_value) * (
                max_range_value - min_range_value
            ) / (servo_max_value - servo_min_value) + min_range_value
        return action_norm
    

    def quaternion_to_euler(self, quat, degrees=True):
        """
        Converts quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw).
        
        Args:
            x, y, z, w: float - Quaternion components.
            degrees: bool - If True, returns angles in degrees. Defaults to radians.
            
        Returns:
            (roll, pitch, yaw): tuple of floats
        """
        x, y, z, w = quat
        r = R.from_quat([x, y, z, w])
        roll, pitch, yaw = r.as_euler('xyz', degrees=degrees)

        return [x, y, z, roll, pitch, yaw]
    

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)

    def save_extras(self, base_log_dir: str):
        shutil.copy2(f"{os.path.dirname(os.path.abspath(__file__))}/reward_config.py", base_log_dir)
        shutil.copy2(os.path.expanduser(f"~/main/configs/auv_env_config.json"), base_log_dir)
        shutil.copy2(os.path.expanduser(f"~/main/configs/auv_config.json"), base_log_dir)


    @exception_handler("Environment failed to reboot")
    def reboot(self):
        logging.info("Rebooting not implemented yet")
        
    @abstractmethod
    def _reset(self):
        pass

    @abstractmethod
    def _get_environment_info(self):
        pass

    @abstractmethod
    def _environment_info_to_state(self, environment_info):
        pass

    @abstractmethod
    def _choose_goal(self):
        pass

    @abstractmethod
    def _render_environment(self, state, environment_info):
        pass
