from abc import ABC

class Environment(ABC):
    def hi(self):
        print("Hello from Environment")

class testEnv(Environment):
    def __init__(self):
        super().__init__()

    def hi(self):
        print("This is a test method in testEnv")

a = testEnv()
a.hi()
# import os

# from dataclasses import dataclass

# @dataclass()
# class Person:
#     name: str = ""
#     age: int = 0

# p = Person()

# p.name = "Alice"
# p.age = 30

# print(p)  # Person(name='Alice', age=30)


# from dataclasses import dataclass
# @dataclass()
# class IMU_Quaternion:
#     x: float = 5.0
#     y: float = 0.0
#     z: float = 0.0
#     w: float = 0.0
#     roll_rate: float = 0.0  # deg/min
#     pitch_rate: float = 0.0  # deg/min
#     yaw_rate: float = 0.0  # deg/min
#     x_accel: float = 0.0  # g
#     y_accel: float = 0.0  # g
#     z_accel: float = 0.0  # g
#     imu_q_data_valid: bool = False


# def update_IMU_quaternion():
#     """
#     Update the IMU_Q data class with the latest IMU quaternion data
#     """
#     IMU_Q = IMU_Quaternion()

#     IMU_Q.x = message[1]
#     IMU_Q.y = message[2]
#     IMU_Q.z = message[3]
#     IMU_Q.w = message[4]
#     IMU_Q.roll_rate = message[5]
#     IMU_Q.pitch_rate = message[6]
#     IMU_Q.yaw_rate = message[7]
#     IMU_Q.x_accel = message[8]
#     IMU_Q.y_accel = message[9]
#     IMU_Q.z_accel = message[10]
#     IMU_Q.imu_q_data_valid = True if message[11][0] == "A" else False

#     return IMU_Q


# message = [0,-0.00186,-0.00165,0.23272,0.97267,-0.02,-0.01,-0.06,0.002,-0.001,0.021,"A*B7215A7B"]
# current_info = {}


# print(current_info)
# # print(current_info["IMUQ"].x)  # Accessing the x attribute of IMU_Q

# current_info["IMUQ"] = update_IMU_quaternion() # dictionary fine?

# print(current_info)
# print(current_info["IMUQ"].x)  # Accessing the x attribute of IMU_Q


# message = [0,-0.00189,-0.00158,0.23309,0.97258,-0.01,0.01,-0.02,0.004,-0.002,0.021,"F*B7215A7B"]


# print(current_info)
# print(current_info["IMUQ"].x)  # Accessing the x attribute of IMU_Q

# current_info["IMUQ"] = update_IMU_quaternion() # dictionary fine?

# print(current_info)
# print(current_info["IMUQ"].x)  # Accessing the x attribute of IMU_f

####################################################################3

# from numpy import array

# marker_poses = {
#     5: {
#         "position": array([-58.07996666, 32.6119634, 343.47226231]),
#         "orientation": [181.21332268660473, 1.1820904612626135, 341.11097488085085],
#         "r_vec": array([[-3.07489823, 0.51184593, 0.02630519]]),
#     },
#     6: {
#         "position": array([-25.77131106, 10.27230043, 341.24471996]),
#         "orientation": [188.42776034989564, 357.4082444490566, 70.33801926856701],
#         "r_vec": array([[-2.4450115, -1.72889158, 0.07170825]]),
#     },
#     3: {
#         "position": array([-62.70799275, -7.55394334, 340.50320327]),
#         "orientation": [193.288159403439, 336.6137346102452, 339.79352560962195],
#         "r_vec": array([[-2.88289959, 0.44228967, -0.65368798]]),
#     },
#     4: {
#         "position": array([9.3818177, -11.21627118, 339.82926937]),
#         "orientation": [185.89128587727035, 14.900480128940671, 73.81071741052142],
#         "r_vec": array([[2.4429231, 1.80897649, -0.41177783]]),
#     },
#     1: {
#         "position": array([-43.48746966, -67.30028667, 336.67660563]),
#         "orientation": [182.64544349541129, 4.733150433692452, 359.8886134568725],
#         "r_vec": array([[3.18508335, -0.00613541, -0.13156187]]),
#     },
#     2: {
#         "position": array([39.22830717, -66.52764949, 337.71091298]),
#         "orientation": [171.09558989052385, 4.707236745900978, 3.6902549704630987],
#         "r_vec": array([[-3.29491865, -0.11670141, 0.12717456]]),
#     },
# }


# a = {i: marker_poses[i]["position"] for i in [5,6]}

# # b = dict([i, marker_poses[i]] for i in [5,6])

# print(a)
# print(b)
# print(a == b)

##########################################################

# from environments.reward_config import _reward_function
# class StationaryCubeTask():
#     def __init__(
#         self,
#     ):

#         self.unction = _reward_function



# test = StationaryCubeTask()


# print(test.unction([0,0,0,0,0,0,0,0,0,0], [1,2,3,4,5,6,7,8,9,10])) # Example usage

# #################################################################

# print("Current folder:", os.path.dirname(os.path.abspath(__file__)))