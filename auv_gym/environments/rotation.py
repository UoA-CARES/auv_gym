from abc import abstractmethod

import cv2
import auv_gym.tools.utils as utils
from auv_gym.environments.environment import Environment
import logging
import numpy as np
import time
from functools import wraps
import math


from auv_gym.tools.configurations import AUVEnvironmentConfig
from auv_gym.tools.pid_controller import PIDController
from boxfish_lib.boxfish_configuration import BoxfishConfig
from boxfish_lib.vision.STagDetector import STagDetector
import random

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

class RotationTask(Environment):
    def __init__(
        self,
        env_config: AUVEnvironmentConfig,
        auv_config: BoxfishConfig,
    ):
        super().__init__(env_config, auv_config)

        self.gripper_marker_ids = auv_config.gripper_marker_ids
        self.inner_wheel_marker_ids = [1,2,3,4,5,6,7,8,9,10]
        self.outter_wheel_marker_ids = [21,22,23,24,25,26,27,28]

        self.marker_detector = STagDetector(marker_size=env_config.marker_size, library_hd=13)
        self.missed_marker_num = 0
        

    def _reset(self):
        self.auv.level()
        self.auv.reverse_hard_stop()

        marker_poses = self._get_poses()
        for i in range(5):
            if 21 not in marker_poses["outter_wheel"]:
                print("Marker 21 not detected, backing up")
                # take a step back
                self.auv.reverse_hard_stop()
            else:
                print("Object marker detected, proceed to centering")
                break

            if i == 4:
                raise EnvironmentError(self.auv, "Marker 21 not detected after multiple attempts.")
                    
        self.pid_controller()

        self.chosen_marker_id = self._choose_goal(marker_poses["inner_wheel"])


    def get_distances(self, marker_poses):
        left_marker = marker_poses["gripper"][self.gripper_marker_ids[0]]["position"]
        right_marker = marker_poses["gripper"][self.gripper_marker_ids[1]]["position"]
        object_marker = marker_poses["object"]["position"]

        distances = object_marker - left_marker

        # could use pixel positions for x/sideways, y/vertical
        # utils.position_to_pixel(position, self.camera.camera_matrix)

        return distances

    def _environment_info_to_state(self, environment_info):
        state = []

        # MMMM...
        state.append(environment_info["auv_pose"]["IMUQ"].x)
        state.append(environment_info["auv_pose"]["IMUQ"].y)
        state.append(environment_info["auv_pose"]["IMUQ"].z)
        state.append(environment_info["auv_pose"]["IMUQ"].w) #?

        # state.append(environment_info["auv_pose"]["IMUQ"].roll_rate)
        # state.append(environment_info["auv_pose"]["IMUQ"].pitch_rate)
        # state.append(environment_info["auv_pose"]["IMUQ"].yaw_rate)
        # state.append(environment_info["auv_pose"]["IMUQ"].x_accel)
        # state.append(environment_info["auv_pose"]["IMUQ"].y_accel)
        # state.append(environment_info["auv_pose"]["IMUQ"].z_accel)

        state += self._pose_to_state(environment_info["gripper_poses"][self.gripper_marker_ids[0]])
        state += self._pose_to_state(environment_info["gripper_poses"][self.gripper_marker_ids[1]])
        state += self._pose_to_state(environment_info["inner_wheel_poses"][self.chosen_marker_id])
        state.append(self._get_yaw(environment_info["inner_wheel_poses"][self.chosen_marker_id]))
        state.append(self._get_target_yaw(environment_info["outter_wheel_poses"]))

        return np.array(state)

    def _get_environment_info(self):
        """
        Retrieves the current state of the environment.

        Returns:
        A list representing the state of the environment.
        """
        environment_info = {}
        environment_info["auv_pose"] = self.auv.info()

        marker_poses = self._get_poses()
        environment_info["gripper_poses"] = marker_poses["gripper"]
        environment_info["inner_wheel_poses"] = marker_poses["inner_wheel"]
        environment_info["outter_wheel_poses"] = marker_poses["outter_wheel"]

        if self.reward >= (self.goal_reward - 1):
            if self.steps_to_success == 0:
                self.steps_to_success = self.step_counter
            self.success_counter += 1   

        environment_info["success_counter"] = self.success_counter
        environment_info["steps_to_success"] = self.steps_to_success

        return environment_info

    def _get_marker_poses(self, must_see_ids):
        self.missed_marker_num = 0
        must_see_ids = [*must_see_ids, self.chosen_marker_id] if self.chosen_marker_id is not None else must_see_ids
        flush = False
        while True:
            logging.debug(f"Attempting to Detect markers: {must_see_ids}")
            frame = cv2.rotate(self.camera.get_frame(0.6, flush), cv2.ROTATE_180) if self.is_inverted else self.camera.get_frame(0.6, flush)

            markers = self.marker_detector.get_marker_poses(
                frame,
                self.camera.camera_matrix,
                self.camera.camera_distortion,
                display=True,
            )

            # This will check that all the mandatory markers are detected correctly
            if all(ids in markers for ids in must_see_ids): #if all gripper marker are detected
                if len(markers) >= len(must_see_ids): # if any object marker is detected
                    break

            self.missed_marker_num += 1
            print(f"Missed marker detection {self.missed_marker_num} times")
            if self.missed_marker_num >= 10:
                input(f"Markers {must_see_ids} not detected, please check the camera and markers. Press Enter to continue...")
                self.auv.stop()  # Stop the AUV
                flush = True

                # to resume, you don't have to see the same inner wheel marker again, but is this going to break stuff? do i need to make sure i see the other markers?
                # must_see_ids = must_see_ids[:-1]  if len(must_see_ids)>0 else [] # remove the last element to make it easier to detect

        return markers
    

    def _get_poses(self):
        """
        Gets the current state of the environment using the markers.

        Returns:
        dict : A dictionary containing the poses of the gripper and object markers.

        gripper left: marker 18
        gripper right: marker 19
        object: 1-10, 21-28
        """
        poses = {}

        marker_poses = self._get_marker_poses(must_see_ids=self.gripper_marker_ids)
        
        poses["gripper"] = {}
        poses["inner_wheel"] = {}
        poses["outter_wheel"] = {}

        for marker_id, pose in marker_poses.items():
            if marker_id in self.gripper_marker_ids:
                poses["gripper"][marker_id] = pose
            elif marker_id <= 10:
                poses["inner_wheel"][marker_id] = pose
            elif marker_id > 20:
                poses["outter_wheel"][marker_id] = pose

        return poses


    def _get_target_yaw(self, poses):
        """
        Calculates the average yaw from a dictionary of marker poses, handling wrap-around at 360 degrees.
        """
        yaws = [self._get_yaw(pose) for pose in poses.values()]
        if not yaws:
            return 0.0
        # Use complex numbers to average angles
        avg = np.angle(np.mean(np.exp(1j * np.deg2rad(yaws))), deg=True)
        return avg % 360
        

    def _get_yaw(self, pose):
        return pose["orientation"][2]

    def _pose_to_state(self, pose):
        state = []
        position = pose["position"]
        state.append(position[0])  # X
        state.append(position[1])  # Y
        state.append(position[2])  # Z
        return state

    def sample_action(self):
        """
        Samples a random action for the AUV.
        control_action = ["roll"(float), "grabber"(-1, 0, or 1)]
        Returns:
            list: [roll_value (float), grabber_state (int)]
        """
        roll_min = self.auv.min_values[0]
        roll_max = self.auv.max_values[0]
        roll_value = random.uniform(roll_min, roll_max)
        grabber_state = random.choice([-1, 0, 1])
        return [roll_value, grabber_state]

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
        
        self.auv.move_roll_grabber(action)

        self.current_environment_info = self._get_environment_info()

        state = self._environment_info_to_state(self.current_environment_info)
        image = self._render_environment(state, self.current_environment_info)

        if self.display:
            cv2.imshow("State Image", image)
            cv2.waitKey(10)

        reward, done = self._reward_function(
            self.previous_state, state
        )

        self.previous_environment_info = self.current_environment_info
        self.previous_state = state
        
        return state, reward, done, truncated, self.current_environment_info


    def pid_controller(self, tolerance=40, hold_time=1):  # Add hold_time parameter
        """
        PID-based controller for the boxfish
        
        Args:
            tolerance: Distance tolerance - stops when all axes are within this range
            hold_time: Time to hold position within tolerance before stopping (seconds)
        """
        # PID gains - tune these values based on your system
        # Start with these values and adjust based on performance:
        # - Increase Kp for faster response but watch for oscillation
        # - Increase Ki to eliminate steady-state error
        # - Increase Kd to reduce overshoot and improve stability
        
        setpoints = {'x': 60, 'y': -230, 'z': 70}  # Target distances
        
        pid_controllers = {
            'x': PIDController(kp=0.0005, ki=0.0002, kd=0.0002, setpoint=setpoints['x']),  # Forward/backward
            'y': PIDController(kp=0.0003, ki=0.0002, kd=0.0002, setpoint=setpoints['y']), # Up/down (inverted)
            'z': PIDController(kp=0.0005, ki=0.0002, kd=0.0002, setpoint=setpoints['z'])  # Left/right (inverted)
        }
        
        # Set output limits to prevent excessive thrust
        for controller in pid_controllers.values():
            controller.set_output_limits(-0.05, 0.05)  # Adjust based on your thrust limits
            controller.set_integral_limits(-10.0, 10.0)  # Prevent integral windup
        
        # Control loop with hold time tracking
        within_tolerance_time = 0.0
        last_time = time.time()
        
        try:
            while True:
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time
                
                distances = self.pid_get_distances()  # [x, y, z]
                
                # Check if all axes are within tolerance
                errors = {
                    'x': abs(distances[0] - setpoints['x']),
                    'y': abs(distances[1] - setpoints['y']),
                    'z': abs(distances[2] - setpoints['z'])
                }
                
                max_error = max(errors.values())
                
                # Check tolerance with hold time requirement
                if max_error <= tolerance:
                    within_tolerance_time += dt
                    print(f"Within tolerance for {within_tolerance_time:.1f}s (need {hold_time}s)")
                    
                    if within_tolerance_time >= hold_time:
                        print(f"Target reached and held for {hold_time}s!")
                        print(f"Final distances: x={distances[0]:.1f}, y={distances[1]:.1f}, z={distances[2]:.1f}")
                        print(f"Final errors: x={errors['x']:.1f}, y={errors['y']:.1f}, z={errors['z']:.1f}")
                        # Stop all thrust and exit
                        self.auv.stop()
                        break
                else:
                    within_tolerance_time = 0.0  # Reset timer if we leave tolerance
                
                # Calculate PID outputs for each axis
                thrust_x = pid_controllers['x'].update(distances[0])  # Forward/backward
                thrust_y = pid_controllers['y'].update(distances[1])  # Up/down
                thrust_z = pid_controllers['z'].update(distances[2])  # Left/right
                
                # Map to thrust vector [forward, sideways, vertical] = [z, x, y]
                thrust = [-thrust_z, -thrust_x, thrust_y]
                
                # Apply thrust
                self.auv.move([*thrust, 0, 0, 0])
                
                # print(f"Distances: x={distances[0]:.1f}, y={distances[1]:.1f}, z={distances[2]:.1f}")
                # print(f"Errors: x={errors['x']:.1f}, y={errors['y']:.1f}, z={errors['z']:.1f}, max={max_error:.1f}")
                # print(f"Thrust: forward={thrust[0]:.3f}, sideways={thrust[1]:.3f}, vertical={thrust[2]:.3f}")
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.05)  # 20Hz control loop
                
        except KeyboardInterrupt:
            print("\nController stopped by user")
            # Stop all thrust
            self.auv.stop()

    def pid_get_distances(self):
        markers_poses =self._get_poses()

        left_marker= markers_poses["gripper"][self.gripper_marker_ids[0]]["position"]
        right_marker = markers_poses["gripper"][self.gripper_marker_ids[1]]["position"]
        object_marker = markers_poses["outter_wheel"][21]["position"]

        # middle_marker = (left_marker + right_marker) / 2

        # print("left_marker", left_marker)
        # print("right_marker", right_marker)
        # print("object_marker", object_marker)
        # print("middle_marker", middle_marker)

        distances = object_marker - left_marker
        print("Object distance from the left marker:", distances)
        # distances[sideway_distance,vertical_distance,forward_distance]

        return distances


    def _draw_circle(self, image, position, color):
        pixel_location = utils.position_to_pixel(
            position,
            self.camera.camera_matrix,
        )
        # Circle size now reflects the "Close enough" to goal tolerance
        cv2.circle(image, pixel_location, self.noise_tolerance, color, -1)
        return image, pixel_location

    def _render_environment(self, state, environment_state):
        """
        Draw box around detected object and gripper markers on the frame.
        reward
        object box change colour if between gripper markers?
        """

        image = cv2.rotate(self.camera.get_frame(0.6), cv2.ROTATE_180) if self.is_inverted else self.camera.get_frame(0.6)

        image = cv2.undistort(
            image, self.camera.camera_matrix, self.camera.camera_distortion
        )

        num_gripper_markers = len(self.gripper_marker_ids)

        # account for auv pose and IMU quaternion in state
        base_index = 4 # x, y, z, w

        for i in range(0, num_gripper_markers+1): # also draw the object marker
            # x = state[base_index + i * 3]
            # y = state[base_index + i * 3 + 1]
            # z = state[base_index + i * 3 + 2]

            position = self._pose_to_state(environment_state["inner_wheel_poses"][self.chosen_marker_id])


            # position = [
            #     x,
            #     y,
            #     z,
            # ]


            marker_pixel = utils.position_to_pixel(
                position,
                self.camera.camera_matrix,
            )

            cv2.circle(image, marker_pixel, 9, (0, 255, 0), -1)
            id_num = f"{self.gripper_marker_ids[i]}" if i < num_gripper_markers else str(self.chosen_marker_id)

            cv2.putText(
                image,
                id_num,
                marker_pixel,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # Draw the orientation/goal visualization in the bottom left corner

        # Define the size and position of the mini-visualization
        viz_size = 120  # Increased size of the square visualization
        margin = 24     # Slightly increased margin from the image border

        # Bottom left corner origin (top-left of the viz box)
        viz_origin_x = margin
        viz_origin_y = image.shape[0] - viz_size - margin

        # Center of the viz box
        centre = [viz_origin_x + viz_size // 2, viz_origin_y + viz_size // 2]

        # Get yaws
        target_yaw = state[-1]  # yaw of the target angle
        object_yaw = state[-2]  # yaw of the object
        lineSize = viz_size // 2 - 16  # Adjusted for larger viz

        # Arrow for current orientation
        arrow_end_axis = [
            int(centre[0] + math.sin(math.radians(object_yaw)) * lineSize),
            int(centre[1] - math.cos(math.radians(object_yaw)) * lineSize)
        ]
        # Arrow for goal orientation
        arrow_end_goal = [
            int(centre[0] + math.sin(math.radians(target_yaw)) * lineSize),
            int(centre[1] - math.cos(math.radians(target_yaw)) * lineSize)
        ]

        # Draw background rectangle for clarity (taller and moved up)
        extra_height = 40  # Increase height for extra space on top
        cv2.rectangle(
            image,
            (viz_origin_x - 7, viz_origin_y - 7 - extra_height),
            (viz_origin_x + viz_size + 7, viz_origin_y + viz_size + 7),
            (30, 30, 30),
            -1
        )

        # Draw the reward value on top of the background rectangle (move up)
        cv2.putText(
            image,
            f"R: {getattr(self, 'reward', 0):.2f}",
            (viz_origin_x + 5, viz_origin_y + 20 - extra_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Draw the center point
        cv2.circle(image, centre, 7, (0, 0, 255), -1)
        # Draw current orientation arrow
        cv2.arrowedLine(image, centre, arrow_end_axis, (0, 255, 255), 4, tipLength=0.3)
        # Draw goal orientation arrow
        cv2.arrowedLine(image, centre, arrow_end_goal, (0, 255, 0), 4, tipLength=0.3)

        cv2.putText(
                image,
                f"{'Current'}",
                arrow_end_axis,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        
        cv2.putText(
                image,
                f"{'Goal'}",
                arrow_end_goal,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )



        return image

class JustSpin(RotationTask):
    def __init__(
        self,
        env_config: AUVEnvironmentConfig,
        auv_config: BoxfishConfig,
    ):
        super().__init__(env_config, auv_config)
        self.total_time = 0
        self.elapsed_num = 0

    def _reset(self):
        self.auv.level()
        time.sleep(3)

        self.camera.flash_cam_buffer()
        marker_poses = self._get_poses()
        for i in range(5):
            if 21 not in marker_poses["outter_wheel"] and i == 4:
                input("Marker 21 not detected, HELP HELP HELP")
            else:
                print("Object marker detected")
                break

        self.chosen_marker_id = self._choose_goal(marker_poses["inner_wheel"])

    def sample_action(self):
        """
        Samples a random action for the AUV.
        control_action = ["roll"(float)]
        Returns:
            list: [roll_value (float)]
        """
        roll_min = self.auv.min_values[0]
        roll_max = self.auv.max_values[0]
        roll_value = random.uniform(roll_min, roll_max)
        
        return [roll_value]

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

        # end_time = time.time()

        self.step_counter += 1
        # moving this over early so it doesn't overwrite marker detection issues
        truncated = self.step_counter >= self.episode_horizon

        # self.auv.move([0, 0, 0, action, 0, 0])  # config limit to -0.5 - 0.5
        self.auv.spin_by_angle(action)  # change config limit to -180 to 180

        self.current_environment_info = self._get_environment_info()
        

        # if gripper marker not detected then need human to check
        if self.missed_marker_num > 10:
            truncated = True
            self.auv.stop()  # Stop the AUV
            print("Inner or outer wheel not detected")


        state = self._environment_info_to_state(self.current_environment_info)
        image = self._render_environment(state, self.current_environment_info)

        if self.display:
            cv2.imshow("State Image", image)
            cv2.waitKey(10)

        self.reward, done = self._reward_function(
            self.previous_state, state
        )
        print("Reward:", self.reward)

        self.previous_environment_info = self.current_environment_info
        self.previous_state = state
        
        # start_time = time.time()
        # elapsed_time = end_time - start_time
        # elapsed_time = end_time - start_time
        # self.total_time += elapsed_time
        # self.elapsed_num+=1
        # average_time = self.total_time/self.elapsed_num
        # print(f"Average time per step: {average_time:.4f} seconds over {self.elapsed_num} steps")
        # print(f"Elapsed time: {elapsed_time:.4f} seconds")
        
        return state, self.reward, done, truncated, self.current_environment_info
    
    def _environment_info_to_state(self, environment_info):
        state = []
        # MMMM...
        state.append(environment_info["auv_pose"]["IMUQ"].x)
        state.append(environment_info["auv_pose"]["IMUQ"].y)
        state.append(environment_info["auv_pose"]["IMUQ"].z)
        # state.append(environment_info["auv_pose"]["IMUQ"].w) #?

        
        quat = [environment_info["auv_pose"]["IMUQ"].x, environment_info["auv_pose"]["IMUQ"].y, environment_info["auv_pose"]["IMUQ"].z, environment_info["auv_pose"]["IMUQ"].w]
        euler = self.quaternion_to_euler(quat, degrees=True)
        state.append(euler[3]) # roll
        # state.append(euler[4]) # pitch
        # state.append(euler[5]) # yaw


        # state.append(environment_info["auv_pose"]["IMUQ"].roll_rate)
        # state.append(environment_info["auv_pose"]["IMUQ"].pitch_rate)
        # state.append(environment_info["auv_pose"]["IMUQ"].yaw_rate)
        # state.append(environment_info["auv_pose"]["IMUQ"].x_accel)
        # state.append(environment_info["auv_pose"]["IMUQ"].y_accel)
        # state.append(environment_info["auv_pose"]["IMUQ"].z_accel)

        # for id in self.gripper_marker_ids:
        #     state += self._pose_to_state(environment_info["gripper_poses"][id])
        # state += self._pose_to_state(environment_info["inner_wheel_poses"][self.chosen_marker_id])
        
        state.append(self._get_yaw(environment_info["inner_wheel_poses"][self.chosen_marker_id]))
        state.append(self._get_target_yaw(environment_info["outter_wheel_poses"]))

        return np.array(state)
    

    def _choose_goal(self, inner_wheel_poses):
        # pick which one of the inner wheel poses has a yaw closest to 45
        closest_marker_id = None
        closest_yaw_diff = float('inf')
        target_yaw = 45  # degrees
        for marker_id, pose in inner_wheel_poses.items():
            yaw = self._get_yaw(pose)
            yaw_diff = abs(yaw - target_yaw)
            if yaw_diff < closest_yaw_diff:
                closest_yaw_diff = yaw_diff
                closest_marker_id = marker_id

        return closest_marker_id
    

class FiveMarkerSpin(JustSpin):
    def __init__(
        self,
        env_config: AUVEnvironmentConfig,
        auv_config: BoxfishConfig,
    ):
        super().__init__(env_config, auv_config)
        self.total_time = 0
        self.elapsed_num = 0


    def _choose_goal(self, inner_wheel_poses):
        while True:
            choice = random.choice(list(inner_wheel_poses.keys()))
            if choice < 6: # occationally other ones like 10 gets detected which creates detections issues cuz it's not stable and it's impossible to reach. not just picking between 1-6 cuz they're not always all detected
                break

        return choice