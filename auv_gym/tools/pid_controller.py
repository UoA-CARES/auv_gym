import numpy as np
import time
from time import sleep
from boxfish_lib.vision.STagDetector import STagDetector
from boxfish_lib.vision.camera import Camera
import numpy as np
import cv2

# from environments.stationary import _get_poses


gripper_ids = [18, 19]  # IDs for the gripper markers

thrust_order = [1,2,0]

class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.setpoint = setpoint  # Target value
        
        # Internal state
        self.previous_error = 0
        self.integral = 0
        self.previous_time = None  # Initialize as None
        
        # Optional limits
        self.output_min = None
        self.output_max = None
        self.integral_min = None
        self.integral_max = None
    
    def set_output_limits(self, min_val, max_val):
        """Set output limits to prevent actuator saturation"""
        self.output_min = min_val
        self.output_max = max_val
    
    def set_integral_limits(self, min_val, max_val):
        """Set integral limits to prevent windup"""
        self.integral_min = min_val
        self.integral_max = max_val
    
    def reset(self):
        """Reset the controller state"""
        self.previous_error = 0
        self.integral = 0
        self.previous_time = None
    
    def update(self, current_value, dt=None):
        """
        Calculate PID output based on current value
        
        Args:
            current_value: Current measured value
            dt: Time step (if None, will calculate from system time)
        
        Returns:
            PID controller output
        """
        current_time = time.time()
        
        # Calculate time step if not provided
        if dt is None:
            if self.previous_time is None:
                dt = 0.01  # Default small time step for first iteration
                self.previous_time = current_time  # Initialize previous_time
            else:
                dt = current_time - self.previous_time
                self.previous_time = current_time
        else:
            self.previous_time = current_time
        
        # Calculate error
        error = self.setpoint - current_value
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integral += error * dt
        
        # Apply integral limits if set
        if self.integral_min is not None and self.integral_max is not None:
            self.integral = max(min(self.integral, self.integral_max), self.integral_min)
        
        i_term = self.ki * self.integral
        
        # Derivative term
        if dt > 0:
            derivative = (error - self.previous_error) / dt
        else:
            derivative = 0
        
        d_term = self.kd * derivative
        
        # Calculate output
        output = p_term + i_term + d_term
        
        # Apply output limits if set
        if self.output_min is not None and self.output_max is not None:
            output = max(min(output, self.output_max), self.output_min)
        
        # Store error for next iteration
        self.previous_error = error
        
        return output


# def pid_controller():
#     """
#     PID-based controller for the AUV
#     """
#     # PID gains - tune these values based on your system
#     # Start with these values and adjust based on performance:
#     # - Increase Kp for faster response but watch for oscillation
#     # - Increase Ki to eliminate steady-state error
#     # - Increase Kd to reduce overshoot and improve stability
    
#     pid_controllers = {
#         'x': PIDController(kp=0.8, ki=0.1, kd=0.2, setpoint=500),  # Forward/backward
#         'y': PIDController(kp=0.8, ki=0.1, kd=0.2, setpoint=-500), # Up/down (inverted)
#         'z': PIDController(kp=0.8, ki=0.1, kd=0.2, setpoint=-500)  # Left/right (inverted)
#     }
    
#     # Set output limits to prevent excessive thrust
#     for controller in pid_controllers.values():
#         controller.set_output_limits(-0.15, 0.15)  # Adjust based on your thrust limits
#         controller.set_integral_limits(-10.0, 10.0)  # Prevent integral windup
    
#     # Control loop
#     try:
#         while True:
#             distances = get_distances() 

#             # Calculate PID outputs for each axis
#             thrust_x = pid_controllers['x'].update(distances[0])  # Forward/backward
#             thrust_y = pid_controllers['y'].update(distances[1])  # Up/down
#             thrust_z = pid_controllers['z'].update(distances[2])  # Left/right
            
#             # Map to thrust vector [forward, sideways, vertical] = [z, x, y]
#             thrust = [thrust_z, thrust_x, thrust_y]
            
#             # Apply thrust
#             auv.steer(*thrust, 0, 0, 0)
            
#             print(f"Distances: x={distances[0]:.1f}, y={distances[1]:.1f}, z={distances[2]:.1f}")
#             print(f"Thrust: forward={thrust[0]:.3f}, sideways={thrust[1]:.3f}, vertical={thrust[2]:.3f}")
            
#             # Small delay to prevent overwhelming the system
#             time.sleep(0.05)  # 20Hz control loop
            
#     except KeyboardInterrupt:
#         print("\nController stopped by user")
#         # Stop all thrust
#         auv.steer(0, 0, 0, 0, 0, 0)

# # Simplified version with automatic tuning helper
# def auto_tune_pid_controller(auv):
#     """
#     A version that starts with conservative gains and provides tuning guidance
#     """
#     print("Starting PID controller with conservative gains...")
#     print("Tuning tips:")
#     print("- If response is too slow: increase Kp")
#     print("- If oscillating: decrease Kp, increase Kd")
#     print("- If steady-state error: increase Ki")
#     print("- Monitor for integral windup")
    
#     # Start with conservative gains
#     pid_controllers = {
#         'x': PIDController(kp=0.3, ki=0.02, kd=0.1, setpoint=500),
#         'y': PIDController(kp=0.3, ki=0.02, kd=0.1, setpoint=-500),
#         'z': PIDController(kp=0.3, ki=0.02, kd=0.1, setpoint=-500)
#     }
    
#     for controller in pid_controllers.values():
#         controller.set_output_limits(-0.5, 0.5)  # Start with limited thrust
#         controller.set_integral_limits(-5.0, 5.0)
    
#     # Control loop with performance monitoring
#     error_history = {'x': [], 'y': [], 'z': []}
#     iteration = 0
    
#     try:
#         while True:
#             distances = get_distances()
            
#             # Calculate thrust for each axis
#             thrust_x = pid_controllers['x'].update(distances[0])
#             thrust_y = pid_controllers['y'].update(distances[1])
#             thrust_z = pid_controllers['z'].update(distances[2])
            
#             thrust = [thrust_z, thrust_x, thrust_y]
#             boxfish.steer(*thrust, 0, 0, 0)
            
#             # Track errors for tuning analysis
#             errors = {
#                 'x': abs(500 - distances[0]),
#                 'y': abs(-500 - distances[1]),
#                 'z': abs(-500 - distances[2])
#             }
            
#             for axis, error in errors.items():
#                 error_history[axis].append(error)
#                 if len(error_history[axis]) > 100:  # Keep last 100 samples
#                     error_history[axis].pop(0)
            
#             # Print status every 20 iterations
#             if iteration % 20 == 0:
#                 avg_errors = {axis: np.mean(history) if history else 0 
#                             for axis, history in error_history.items()}
#                 print(f"Average errors - X: {avg_errors['x']:.1f}, Y: {avg_errors['y']:.1f}, Z: {avg_errors['z']:.1f}")
#                 print(f"Current thrust: {[f'{t:.3f}' for t in thrust]}")
            
#             iteration += 1
#             time.sleep(0.05)
            
#     except KeyboardInterrupt:
#         print("\nController stopped. Final performance summary:")
#         for axis, history in error_history.items():
#             if history:
#                 print(f"{axis.upper()} axis - Avg error: {np.mean(history):.1f}, Std: {np.std(history):.1f}")
#         boxfish.steer(0, 0, 0, 0, 0, 0)


# # Usage:
# pid_controller()  # Run the main PID controller
# # auto_tune_pid_controller()  # Run with tuning assistance   
