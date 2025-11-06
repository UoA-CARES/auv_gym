 def _reward_function(self, previous_environment_info, current_environment_info):
        """
        Computes the reward based on the target goal and the change in yaw.

        Returns:
            reward: Dependedent on the reward function.
                -Function 1: Distance-to-Goal reward function
                -Function 2: Delta Difference to goal reward function
                -Function 3: Combined Reward Function
            done: True if the goal is reached.
        """
        self.goal_reward = 300
        Precision_tolerance = 15
        touch_reward = 0
        touch_threshold = 1
        done = False
        logging.debug(previous_environment_info['poses']['object']['orientation'])
        
        previous_yaw = previous_environment_info['poses']['object']['orientation'][2]
        previous_yaw_diff = self.rotation_min_difference(self.goal[0], previous_yaw)
        current_yaw = current_environment_info['poses']['object']['orientation'][2]
        current_yaw_diff = self.rotation_min_difference(self.goal[0], current_yaw)

        #Touch-based reward
        if self.touch_config == True:
            print("Getting touch data in reward function")
            print("Max values after step: ", self.tactile_server.max_values)
            # Do reward based on touch sensor values
            for i in range(self.num_sensors):
                delta_touch = self.tactile_server.max_values[i] - self.sensor_baselines[i]
                if delta_touch < touch_threshold:
                    continue
                else:
                    touch_reward += 1
                    print("Touch Reward: ", touch_reward)
            # Reset the max values after each step
            self.tactile_server.max_values = self.sensor_baselines

        ##### Function 1
        # # Distance-to-Goal reward function
        # reward = round(-current_yaw_diff, 2)
        # # Reward set ot 0 if no cube no move
        # if abs(current_yaw_diff - previous_yaw_diff)<5:
        #     reward = 0
        #     print(reward)
        #     return reward, done
        #####

        ##### Function 2
        # # Delta Difference to goal reward function
        # delta = ((previous_yaw_diff - current_yaw_diff)/previous_yaw_diff) * 100
        # reward = round(delta, 2)
        # if reward < -100:
        #     reward = -100
        # # Negatively rewards for not moving the cube
        # if abs(delta) < 1:
        #     reward = -10
        #####
           
        ##### Function 3
        # Combined Reward Function
        A = 0.1 # Distance Coeffecient
        B = 1 # Delta Coefficient
        # Delta
        delta = ((previous_yaw_diff - current_yaw_diff)/previous_yaw_diff) * 100
        delta_reward = round(delta, 2)
        if delta_reward < -100:
            delta_reward = -100
        elif abs(delta_reward) < 1:
            delta_reward = 0
        # Distance
        distance_reward = round((-current_yaw_diff+180), 2)
        if abs(current_yaw_diff - previous_yaw_diff) < 1:
            distance_reward = 0
        reward = round((A*distance_reward) + (B*delta_reward), 2)
        #####

        if current_yaw_diff <= Precision_tolerance:
            logging.info("----------Reached the Goal!----------")
            reward = self.goal_reward
        print(f"Reward: ",reward)
        return reward, done