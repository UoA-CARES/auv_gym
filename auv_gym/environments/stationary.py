from abc import abstractmethod

import cv2
import auv_gym.tools.utils as utils
from auv_gym.environments.environment import Environment
import logging
import numpy as np
import time


from auv_gym.tools.configurations import AUVEnvironmentConfig
from auv_gym.tools.pid_controller import PIDController
from boxfish_lib.boxfish_configuration import BoxfishConfig
from boxfish_lib.vision.STagDetector import STagDetector


class StationaryCubeTask(Environment):
    def __init__(
        self,
        env_config: AUVEnvironmentConfig,
        auv_config: BoxfishConfig,
    ):
        super().__init__(env_config, auv_config)

        self.gripper_marker_ids = auv_config.gripper_marker_ids

        self.object_marker_detector = STagDetector(marker_size=env_config.object_marker_size, library_hd=11)
        self.gripper_marker_detector = STagDetector(marker_size=env_config.gripper_marker_size, library_hd=11)
        

    def _reset(self):
        for actions_taken in reversed(self.actions_taken):
            marker_poses = self._get_poses()
            if len(marker_poses["object"]) is None:
                print("No object marker detected backing up")
                # take a step back
                self.auv.move([*-np.array(actions_taken), 0, 0, 0])
            else:
                print("Object marker detected, proceed to centering")
                break
                    
        # now the object is in view, use the expert controller to center and reverse
        # self.expert_controller_home(marker_poses)
        self.pid_controller()


    def get_distances(self, marker_poses):
        left_marker = marker_poses["gripper"][7]["position"]
        right_marker = marker_poses["gripper"][8]["position"]
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

        state += self._pose_to_state(environment_info["gripper_poses"][7])
        state += self._pose_to_state(environment_info["gripper_poses"][8])

        state += self._pose_to_state(environment_info["object_pose"])

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
        environment_info["object_pose"] = marker_poses["object"]
        
        return environment_info

    def _get_marker_poses(self, must_see_ids):
        for _ in range(10):
            logging.debug(f"Attempting to Detect markers: {must_see_ids}")
            frame = cv2.rotate(self.camera.get_frame(0.6), cv2.ROTATE_180) if self.is_inverted else self.camera.get_frame(0.6)
            marker_poses = {}

            gripper_size_markers = self.gripper_marker_detector.get_marker_poses(
                frame,
                self.camera.camera_matrix,
                self.camera.camera_distortion,
                display=True,
            )
            # Only add the gripper markers that are in the must_see_ids
            for marker_id in gripper_size_markers:
                if marker_id in must_see_ids:
                    marker_poses[marker_id] = gripper_size_markers[marker_id]

            # print("marker_poses", marker_poses)


            object_size_markers = self.object_marker_detector.get_marker_poses(
                frame,
                self.camera.camera_matrix,
                self.camera.camera_distortion,
                display=True,
            )
            for marker_id in object_size_markers:
                if marker_id not in must_see_ids:
                    marker_poses[marker_id] = object_size_markers[marker_id]
            # print("marker_poses after object", marker_poses)

            # This will check that all the mandatory markers are detected correctly
            if all(ids in marker_poses for ids in must_see_ids): #if all gripper marker are detected
                if len(marker_poses) >= len(must_see_ids): # if any object marker is detected
                    break

            print("not seeing the gripper and object markers")

        return marker_poses
    
    def _get_poses(self):
        """
        Gets the current state of the environment using the markers.

        Returns:
        dict : A dictionary containing the poses of the gripper and object markers.

        gripper left: marker 7
        gripper right: marker 8
        object: X-Y-Z-RPY Object marker 1-6
        """
        poses = {}

        # num_gripper_markers = 2

        # marker_ids = [id for id in range(1, num_gripper_markers + 1)]

        

        marker_poses = self._get_marker_poses(must_see_ids=self.gripper_marker_ids)

        print(marker_poses[4])
        input("pause")
        
        poses["gripper"] = {i: marker_poses[i] for i in self.gripper_marker_ids}

        poses["object"] = self._get_cube_pose(marker_poses)

        return poses
    

    def _get_cube_pose(self, marker_poses):
        """
        Calculate the center point of a cube base on the detected markers.
        Args:
            marker_poses (dict): A dictionary containing the poses of the detected markers.
        Returns:
            dict: A dictionary containing the position and orientation of the cube.
        """
        cube_ids = [1,2,3,4,5,6]
        detected_ids = [ids for ids in marker_poses]

        cube_marker_ids = [id for id in cube_ids if id in detected_ids]

        if len(cube_marker_ids) == 0:
            # If no cube marker detected, return a default pose assuming the cube has been dropped
            return None
        else:
            # Calculate the cube centers for the marker IDs present in both cube_ids and detected_ids
            cube_centers = [self._calculate_cube_center(marker_poses[id]["position"], marker_poses[id]["r_vec"])
            for id in cube_marker_ids]

            # Calculate the final cube center by averaging
            cube_centers = np.array(cube_centers)
            cube_center = np.mean(cube_centers, axis=0)

        return {'position': cube_center, 'orientation': [1.0, 1.0, 1.0]}
    
    
    def _calculate_cube_center(self, marker_position, r_vec, cube_size=50):
        """
        Calculate the center point of a cube given the position and orientation of one face.  
        Args:
            marker_position (numpy.ndarray): A 1D array of length 3 representing the x, y, z coordinates of the center of the face.
            r_vec (numpy.ndarray): A 1D array of length 3 representing the row, pitch, yaw angles (in radians) of the face.
            cube_size (int): The size of the cube (default is 50).
        Returns:
            numpy.ndarray: A 1D array of length 3 representing the x, y, z coordinates of the center of the cube.
        """

        # Calculate the rotation matrix from the Rodrigues vector
        rotation_matrix, _ = cv2.Rodrigues(r_vec)

        # Calculate the offset from the face center to the cube center
        offset = np.dot(rotation_matrix, np.array([0, 0, cube_size / 2]))

        # Calculate the cube center
        cube_center = marker_position - offset

        return cube_center

    def _choose_goal(self):
        pass
    

    def _gpt_reward_function(state, k1=10.0, R_bonus=10.0, margin=0.01):
        """
        Computes the reward for placing object A between markers B and C.

        Parameters:
        - state: List or array in the format [a, b, c, d, xB, yB, xC, yC, xA, yA]
        - k1: Sharpness of the proximity reward
        - R_bonus: Bonus reward for placing object within the claw
        - margin: Safety margin to ensure the object is comfortably inside the claw gap

        Returns:
        - total_reward: The computed reward
        """
        # Unpack state
        _, _, _, _, xB, yB, xC, yC, xA, yA = state
        
        # Positions
        p_B = np.array([xB, yB])
        p_C = np.array([xC, yC])
        p_A = np.array([xA, yA])
        
        # Compute claw center
        p_center = (p_B + p_C) / 2.0
        d_A_center = np.linalg.norm(p_A - p_center)
        
        # Compute distance between claws
        d_BC = np.linalg.norm(p_B - p_C)
        
        # Proximity reward
        R_proximity = np.exp(-k1 * d_A_center)
        
        # Inside claw bonus
        if d_A_center < 0.5 * d_BC - margin:
            R_inside = R_bonus
        else:
            R_inside = 0.0
        
        # Total reward
        total_reward = R_proximity + R_inside
        
        return total_reward


    def _pose_to_state(self, pose):
        state = []
        position = pose["position"]
        state.append(position[0])  # X
        state.append(position[1])  # Y
        return state



    def pid_controller(self, tolerance=50):  # Add tolerance parameter
        """
        PID-based controller for the boxfish
        
        Args:
            tolerance: Distance tolerance - stops when all axes are within this range
        """
        # PID gains - tune these values based on your system
        # Start with these values and adjust based on performance:
        # - Increase Kp for faster response but watch for oscillation
        # - Increase Ki to eliminate steady-state error
        # - Increase Kd to reduce overshoot and improve stability
        
        setpoints = {'x': 50, 'y': -30, 'z': 50}  # Target distances
        
        pid_controllers = {
            'x': PIDController(kp=0.0005, ki=0.0002, kd=0.0002, setpoint=setpoints['x']),  # Forward/backward
            'y': PIDController(kp=0.0003, ki=0.0002, kd=0.0002, setpoint=setpoints['y']), # Up/down (inverted)
            'z': PIDController(kp=0.0005, ki=0.0002, kd=0.0002, setpoint=setpoints['z'])  # Left/right (inverted)
        }
        
        # Set output limits to prevent excessive thrust
        for controller in pid_controllers.values():
            controller.set_output_limits(-0.05, 0.05)  # Adjust based on your thrust limits
            controller.set_integral_limits(-10.0, 10.0)  # Prevent integral windup
        
        # Control loop
        try:
            while True:
                distances = self.pid_get_distances() 
                
                # Check if all axes are within tolerance
                errors = {
                    'x': abs(distances[0] - setpoints['x']),
                    'y': abs(distances[1] - setpoints['y']),
                    'z': abs(distances[2] - setpoints['z'])
                }
                
                max_error = max(errors.values())
                
                if max_error <= tolerance:
                    print(f"Target reached! All axes within tolerance ({tolerance})")
                    print(f"Final distances: x={distances[0]:.1f}, y={distances[1]:.1f}, z={distances[2]:.1f}")
                    print(f"Final errors: x={errors['x']:.1f}, y={errors['y']:.1f}, z={errors['z']:.1f}")
                    # Stop all thrust and exit
                    self.auv.stop()
                    break
                
                # Calculate PID outputs for each axis
                thrust_x = pid_controllers['x'].update(distances[0])  # Forward/backward
                thrust_y = pid_controllers['y'].update(distances[1])  # Up/down
                thrust_z = pid_controllers['z'].update(distances[2])  # Left/right
                
                # Map to thrust vector [forward, sideways, vertical] = [z, x, y]
                thrust = [-thrust_z, -thrust_x, thrust_y]
                
                # Apply thrust
                self.auv.move([*thrust, 0, 0, 0])
                print(f"Distances: x={distances[0]:.1f}, y={distances[1]:.1f}, z={distances[2]:.1f}")
                print(f"Errors: x={errors['x']:.1f}, y={errors['y']:.1f}, z={errors['z']:.1f}, max={max_error:.1f}")
                print(f"Thrust: forward={thrust[0]:.3f}, sideways={thrust[1]:.3f}, vertical={thrust[2]:.3f}")
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.05)  # 20Hz control loop
                
        except KeyboardInterrupt:
            print("\nController stopped by user")
            # Stop all thrust
            self.auv.stop()

    def pid_get_distances(self):
        markers_poses =self._get_poses()

        left_marker= markers_poses["gripper"][7]["position"]
        right_marker = markers_poses["gripper"][8]["position"]
        object_marker = markers_poses["object"]["position"]

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

        num_gripper_markers = 2

        # account for auv pose and IMU quaternion in state
        base_index = 4 # x, y, z, w

        for i in range(0, num_gripper_markers):
            x = state[base_index + i * 2]
            y = state[base_index + i * 2 + 1]


            position = [
                x,
                y,
                environment_state["gripper_poses"][self.gripper_marker_ids[i]]["position"][2],
            ]

            marker_pixel = utils.position_to_pixel(
                position,
                self.camera.camera_matrix,
            )
            cv2.circle(image, marker_pixel, 9, (0, 255, 0), -1)

            cv2.putText(
                image,
                f"{self.gripper_marker_ids[i]}",
                marker_pixel,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        object_color = (0, 255, 0)

        # Draw object's current position
        current_object_pose = environment_state["object_pose"]["position"]
        current_object_pixel = utils.position_to_pixel(
            current_object_pose,
            self.camera.camera_matrix,
        )
        # Circle size now reflects the "Close enough" to goal tolerance
        cv2.circle(image, current_object_pixel, self.noise_tolerance, object_color, -1)

        cv2.putText(
            image,
            "Current",
            (current_object_pixel[0]+self.noise_tolerance, current_object_pixel[1]+self.noise_tolerance), # Text location adjusted for circle size
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            object_color,
            2,
        )

        return image
