# Common
eps = 10 ** 5  # Keep five decimal.

# for test in real-world experiments, where the workspace is a square with 6.8m*4.5m
DT1 = 0.1
NEAR_GOAL_THRESHOLD1 = 0.08
SAMPLE_SIZE1 = 0.08

# for test in real-world experiments, where the workspace is a square with 6.8m*4.5m
DT2 = 0.1
NEAR_GOAL_THRESHOLD2 = 0.5
SAMPLE_SIZE2 = NEAR_GOAL_THRESHOLD2 * 1.5

# for simulation in large-scale scenarios, where the workspace is a square with 6.8km*5.0km
DT = 0.1
NEAR_GOAL_THRESHOLD = 3.0
SAMPLE_SIZE = NEAR_GOAL_THRESHOLD * 1.5


envs_type = {
    0: "small", 1: "large_maze", 2: "large", 3: "large_corridor", 4: "symmetric", 5: "large_obs",
    6: "small_indoor", 7: "small_blocks", 8: "128", 9: "256", 10: "indoor_csc", 11: "exploration_test",
    12: "large_obs_tp"
}

# USE_ROS = True
# USE_REAL_WORLD = True  # true 就是实物实验，false就是gazebo仿真
USE_ROS = False
USE_REAL_WORLD = False  # true 就是实物实验，false就是gazebo仿真
USE_TEMP = False
# MAP_OROGIN_POS_YAW_IN_REAL_WORDL = [5.379, 6.630, -1.57044]  # 实物实验中的地图原点在实际的点云map中的位置，三个值分别表示x y yaw
# MAP_OROGIN_POS_YAW_IN_REAL_WORDL = [6.6070-6.66, 6.630, -1.57044]  # 实物实验中的地图原点在实际的点云map中的位置，三个值分别表示x y yaw
# MAP_OROGIN_POS_YAW_IN_REAL_WORDL = [6.6070-6.5, 6.58, -1.57044]  # 实物实验中的地图原点在实际的点云map中的位置，三个值分别表示x y yaw
MAP_OROGIN_POS_YAW_IN_REAL_WORDL = [6.415, 1.905, 3.14159]  # 实物实验中的地图原点在实际的点云map中的位置，三个值分别表示x y yaw ditu

MAP_Z = 0.0
ROBOT_NUM = 7
