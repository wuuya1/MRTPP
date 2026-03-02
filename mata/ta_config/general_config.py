import os
import copy
import json
import time
import math
import random
import itertools
import subprocess
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mamp.envs import env
from mamp.planner.jps import JPS
from mamp.agents.agent import Agent
from mamp.agents.target import Target
from mamp.configs.config import envs_type
from mamp.planner.ovsPlanner import OVSPlanner
from mamp.policies.orcaPolicy import ORCAPolicy
from mamp.tools.utils import get_boundaries, l2norm, circle_polygon_intersect, enclosing_circle, turning_distance, \
    path_length
from mamp.tools.vector import Vector2


def set_agent_parameters(agent_num):
    """
    car_like_robot: 轴距H: 0.18; 长宽: 0.3*0.2; 最大转向角30度; 最小转弯半径为H/tan(30)
    """

    agent_type1 = {'radius': 2.5, 'pref_speed': 1.5, 'max_speed': 2.0, 'max_angular': np.pi / 6, 'min_speed': 0.2}
    car_like_robot = {'radius': 0.18, 'pref_speed': 0.08, 'max_speed': 0.12, 'max_angular': np.pi / 6,
                      'min_speed': 0.05}
    agent_type3 = {'radius': 5.0, 'pref_speed': 3., 'max_speed': 5., 'max_angular': np.pi / 6, 'min_speed': 2.}
    agents_type = []
    for i in range(agent_num):
        agents_type.append(car_like_robot)
    return agents_type


class AgentSimple(object):
    def __init__(self, start_pos, goal_pos, radius, idx, swarm_num, max_cap, max_range, env_type, exit_area, task_area):
        # agent's physical information
        self.max_load_ = max_cap
        self.max_endurance_ = max_range
        self.tasks = []
        self.group = 0
        self.env = env.Env(env_type)
        self.swarm_num_ = swarm_num
        self.ending_area_ = exit_area
        self.task_area_ = task_area
        self.is_exit_area = True if len(exit_area) > 0 else False
        self.range_x = self.env.xDim
        self.range_y = self.env.yDim
        self.min_x_, self.max_x_, self.min_y_, self.max_y_ = get_boundaries(task_area)
        self.min_xe_, self.max_xe_, self.min_ye_, self.max_ye_ = get_boundaries(exit_area)

        self.ta_pos_ = np.array(start_pos)
        self.pos_global_frame = np.array(start_pos[:2], dtype='float64')
        self.goal_global_frame = np.array(goal_pos[:2], dtype='float64')

        self.radius = radius
        self.turning_radius_ = 0.45
        self.id_ = idx


def build_targets(targets_pos, radius):
    # build targets
    targets = []
    target_radius = radius
    tar_num = len(targets_pos)
    for i in range(tar_num):
        targets.append(Target(pos=targets_pos[i][:2], shape_dict={'shape': 'circle', 'feature': target_radius},
                              idx=i, tid=str(i)))
    return targets


def write_env_cfg(rob_rad, environ, agents_pos, tasks_pos, name='/path_info_by_gps.json', scale=1.0):
    radius = rob_rad
    turning_radius = 3 * radius
    inflation = min(radius / 5, 1.2)
    dist_u2t, path_u2t = info_U2T(agents_pos, tasks_pos, scale, environ, radius, inflation)
    dist_t2t, path_t2t = info_T2T(tasks_pos, scale, environ, radius, inflation)
    data = {'dist_u2t': dist_u2t, 'path_u2t': path_u2t, 'dist_t2t': dist_t2t, 'path_t2t': path_t2t, 'turning': {}}

    if len(tasks_pos) < 50:
        all_permutations = list(itertools.permutations(list(range(len(tasks_pos))), 3))
        for permutation in all_permutations:
            idx0, idx1, idx2 = permutation
            path_01 = path_t2t[idx0][idx1]
            path_12 = path_t2t[idx1][idx2]
            p0, p1, p2 = path_01[-2], path_01[-1], path_12[1]
            distance = abs(turning_distance(p0, p1, p2, turning_radius) - l2norm(path_01[-1], path_12[1])) / scale
            data['turning'][str((idx0, idx1, idx2))] = distance
        dist_matrix = [[0. for _ in range(len(tasks_pos))] for _ in range(len(agents_pos))]
        for i in range(len(agents_pos)):
            for j in range(len(tasks_pos)):
                path_te = path_u2t[i][j]
                p0, p1, p2 = path_te[-2], path_te[-1], path_te[-2]
                distance = abs(turning_distance(p0, p1, p2, turning_radius) - l2norm(path_te[-1], path_te[-2])) / scale
                distance += dist_u2t[i][j]
                dist_matrix[i][j] = float(round(distance, 5))
        data['dist_t2u'] = dist_matrix
    else:
        data['dist_t2u'] = dist_u2t

    save_dir = os.path.dirname(os.path.realpath(__file__)) + '/path_info/'
    info_str = json.dumps(data, indent=4)
    with open(save_dir + name, 'w') as json_file:
        json_file.write(info_str)
    json_file.close()


def read_json(name='/path_infoa8.json'):
    """
    读取路径信息
    """
    save_dir = os.path.dirname(os.path.realpath(__file__)) + '/path_info/'
    with open(save_dir + name, 'r') as file:
        data = json.load(file)
    dist_u2t, path_u2t = data['dist_u2t'], data['path_u2t']
    dist_t2t, path_t2t = data['dist_t2t'], data['path_t2t']
    dist_t2u, turning = data['dist_t2u'], data['turning']

    return dist_u2t, path_u2t, dist_t2t, path_t2t, dist_t2u, turning


def build_obj_pos_large(environment, ag_num=100, tar_num=300, robot_radius=2.5, scl=1000.):
    x_range, y_range = environment.xDim, environment.yDim
    task_area = [[0.0, 0.0], [x_range+150, 0.0], [x_range+150, y_range+50], [0.0, y_range+50], [0.0, 0.0]]
    robs_nest = [[x_range, 0.0], [x_range + 150, 0.0], [x_range + 150, y_range+50], [x_range, y_range+50],
                 [x_range, 0.0]]
    # A example for 10 robots and 100 targets
    robs_start = [[6094.7, 3172.89, 0], [6091.3, 3601.63, 1], [6093.23, 880.49, 2], [6090.04, 670.04, 3], [6094.41, 3055.18, 4], [6090.09, 1343.46, 5], [6094.91, 3006.55, 6], [6091.08, 906.28, 7], [6094.76, 2677.22, 8], [6090.56, 1505.46, 9]]
    tasks_pos = [[1214.5, 1107.75, 0], [1318.99, 1447.47, 1], [4797.97, 2604.41, 2], [2568.19, 881.22, 3], [1210.0, 2966.42, 4], [4396.25, 2144.72, 5], [55.94, 3721.94, 6], [50.59, 3020.45, 7], [2488.19, 2490.43, 8], [3521.48, 3959.98, 9], [4519.05, 1051.77, 10], [3567.33, 2706.94, 11], [3430.24, 2087.26, 12], [4178.89, 195.11, 13], [5543.31, 3120.31, 14], [4648.81, 1998.76, 15], [240.59, 3421.15, 16], [2060.05, 1645.66, 17], [5834.39, 3429.75, 18], [3288.45, 164.31, 19], [1261.2, 2598.74, 20], [1230.95, 33.92, 21], [4634.84, 3822.95, 22], [3507.06, 599.66, 23], [3200.15, 2355.91, 24], [5223.16, 123.38, 25], [2697.76, 3656.24, 26], [2412.52, 562.55, 27], [5047.66, 3929.87, 28], [3376.0, 1077.64, 29], [1680.27, 363.9, 30], [4078.91, 272.64, 31], [4918.34, 2763.93, 32], [5077.57, 524.01, 33], [1707.73, 1410.98, 34], [2813.84, 3082.02, 35], [2564.15, 761.73, 36], [615.15, 351.87, 37], [112.94, 2632.54, 38], [1986.11, 3440.9, 39], [1106.84, 68.53, 40], [443.0, 662.12, 41], [1519.39, 1242.67, 42], [355.84, 3974.17, 43], [49.98, 2902.59, 44], [5019.38, 652.13, 45], [258.41, 2065.99, 46], [1029.2, 2108.67, 47], [5400.93, 3903.86, 48], [5597.89, 2659.31, 49], [4325.01, 54.81, 50], [2012.84, 1027.77, 51], [244.13, 3002.79, 52], [5131.65, 3155.92, 53], [5552.61, 2883.48, 54], [3453.97, 1966.37, 55], [2598.05, 2366.06, 56], [1845.58, 3434.44, 57], [4667.87, 2496.2, 58], [2782.87, 1590.85, 59], [1237.92, 3074.25, 60], [4382.21, 1984.03, 61], [1892.49, 2220.31, 62], [2991.68, 3073.91, 63], [5088.91, 413.45, 64], [3209.21, 1182.69, 65], [2811.02, 2640.32, 66], [4400.89, 1025.69, 67], [2138.53, 690.23, 68], [5788.16, 3738.38, 69], [4639.2, 999.45, 70], [1097.2, 2617.92, 71], [5625.55, 2164.41, 72], [2640.29, 1363.68, 73], [365.8, 558.7, 74], [941.28, 3021.29, 75], [3744.43, 3620.15, 76], [196.9, 563.05, 77], [130.43, 1865.35, 78], [3514.78, 421.31, 79], [3026.38, 2330.74, 80], [16.44, 1905.38, 81], [1726.61, 2191.61, 82], [5707.36, 2744.03, 83], [2297.9, 3092.39, 84], [5769.11, 1938.53, 85], [3125.51, 3678.75, 86], [1836.31, 90.3, 87], [2586.79, 3649.15, 88], [3833.88, 455.94, 89], [1581.75, 26.15, 90], [4908.52, 2343.86, 91], [3425.2, 164.41, 92], [2758.84, 1802.93, 93], [682.81, 1228.84, 94], [4288.84, 2734.89, 95], [5873.95, 3149.19, 96], [1143.35, 2849.22, 97], [4580.53, 1768.4, 98], [3270.95, 3282.81, 99]]
    robs_start = robs_start[:ag_num]
    tasks_pos = tasks_pos[:tar_num]
    agents = []
    max_load = [15 for _ in range(ag_num)]
    max_range = [15000. for _ in range(ag_num)]
    for i in range(ag_num):
        agents.append(AgentSimple(robs_start[i], robs_start[i], robot_radius, i, ag_num,
                                  max_load[i], max_range[i], envs_type[5], robs_nest, task_area))

    return agents, task_area, np.array(tasks_pos), scl


def build_agents_and_tasks(robots, tasks_pos, tar_ids, envs):
    # build agents
    agents = []
    x_range, y_range = (-0.1, 6.7), (0., 6.6)
    robots_num = len(robots)
    agents_para = set_agent_parameters(robots_num)
    for i in range(len(robots)):
        start_pos = robots[i].pos_global_frame
        robot = Agent(start_pos=start_pos, goal_pos=Vector2(start_pos[0], start_pos[1]),
                      radius=robots[i].radius, pref_speed=agents_para[i]['pref_speed'],
                      max_speed=agents_para[i]['max_speed'], max_angular=agents_para[i]['max_angular'],
                      min_speed=agents_para[i]['min_speed'], policy=ORCAPolicy, planner=OVSPlanner, dt=0.1,
                      start_yaw=0.0, max_cap=robots[i].max_load_, max_range=robots[i].max_endurance_,
                      expansion=agents_para[i]['radius'], ta_policy=OVSPlanner, swarm_num=robots_num, scale=1,
                      task_area=robots[i].task_area_, exit_area=robots[i].ending_area_)
        inflation = min(robot.radius_ / 5, 1.2)
        robot.id_ = i
        robot.planner_.init_env(envs.poly_obs, x_range, y_range, robot.radius_, inflation=inflation)
        agents.append(robot)
    # build targets
    targets = []
    tar_radius = robots[0].radius * 2.5
    tar_num = len(tasks_pos)
    for i in range(tar_num):
        targets.append(Target(pos=tasks_pos[i][:2], shape_dict={'shape': 'circle', 'feature': tar_radius},
                              idx=i, tid=str(i)))

    for i in range(len(agents)):
        for j in tar_ids[i]:
            agents[i].targets_.append(targets[j])
    return agents, targets


def info_U2T(agents_pos, tasks_pos, scale, environ, radius, inflation):
    x_range, y_range = (-0.3, environ.xDim + 100.0), (0, environ.yDim)
    poly_obs = environ.poly_obs

    dist_matrix = [[0. for _ in range(len(tasks_pos))] for _ in range(len(agents_pos))]
    path_matrix = [[[] for _ in range(len(tasks_pos))] for _ in range(len(agents_pos))]
    planner = OVSPlanner()
    planner.init_env(poly_obs, x_range, y_range, radius, inflation=inflation)
    for ag_idx, agpos in enumerate(agents_pos):
        print('agent', ag_idx)
        tuple_agpos = tuple(agpos)
        for tar_idx, t_pos in enumerate(tasks_pos):
            print('agent', ag_idx, 'target', tar_idx)
            tuple_tpos = tuple(t_pos)
            planner.set_start_and_goal(tuple_agpos, tuple_tpos)
            path = planner.global_search()
            # if ag_idx == 0:
            # planner.plot_path(path)
            path_result = [[float(round(po[0], 2)), float(round(po[1], 2))] for po in path]
            path_matrix[ag_idx][tar_idx] = path_result
            dist_matrix[ag_idx][tar_idx] = float(round(planner.length / scale, 5))
    return dist_matrix, path_matrix


def info_T2T(tasks_pos, scale, environ, radius, inflation):
    x_range, y_range = (-0.3, environ.xDim + 100.0), (0, environ.yDim)
    poly_obs = environ.poly_obs
    dist_matrix = [[0. for _ in range(len(tasks_pos))] for _ in range(len(tasks_pos))]
    path_matrix = [[[] for _ in range(len(tasks_pos))] for _ in range(len(tasks_pos))]
    planner = OVSPlanner()
    planner.init_env(poly_obs, x_range, y_range, radius, inflation=inflation)
    for i, i_pos in enumerate(tasks_pos):
        print('task', i)
        tuple_ipos = tuple(i_pos)
        for j, j_pos in enumerate(tasks_pos):
            print('task', i, j)
            tuple_jpos = tuple(j_pos)
            planner.set_start_and_goal(tuple_ipos, tuple_jpos)
            path = planner.global_search()
            # planner.plot_path(path)
            path_result = [[float(round(po[0], 2)), float(round(po[1], 2))] for po in path]
            path_matrix[i][j] = path_result
            dist_matrix[i][j] = float(round(planner.length / scale, 5))
    return dist_matrix, path_matrix


def generate_targets_pos(envs, robot_radius, tar_radius=10.0, tar_num=500, scale=1.0):
    x_range, y_range = envs.xDim, envs.yDim
    poly_obs = envs.poly_obs

    targets_pos = []
    iters = 10000
    combined_radius = tar_radius + robot_radius
    while iters > 0:
        iters -= 1
        space = 5.0
        x = round(np.random.uniform(260.0, 5940), 20)
        y = round(np.random.uniform(scale * space * combined_radius, 3950), 20)
        pos_center = [round(x, 2), round(y, 2)]
        is_intersect = False
        for poly_ob in poly_obs:
            if len(poly_ob.vertices_pos) > 4:
                vertices = np.array(poly_ob.vertices_pos)
                pos, radius = enclosing_circle(vertices)
                if l2norm(pos, pos_center) < radius+combined_radius:
                    is_intersect = True
                    break
            if circle_polygon_intersect(pos_center, combined_radius, poly_ob.vertices_pos):
                is_intersect = True
                break
        if targets_pos:
            for tar_pos in targets_pos:
                if l2norm(tar_pos, pos_center) < combined_radius + combined_radius:
                    is_intersect = True
                    break
        if not is_intersect:
            targets_pos.append(pos_center)
        print(len(targets_pos), '------------------------current iters-------------------------', iters)
        if len(targets_pos) == tar_num:
            break
    print(len(targets_pos))
    print(targets_pos)
    targets = build_targets(targets_pos, tar_radius)
    env.plot_env(poly_obs, targets)
    return targets_pos


def generate_robots_pos(envs, robot_radius, rob_num=20):
    x_range, y_range = envs.xDim, envs.yDim
    poly_obs = envs.poly_obs
    robs_pos = []
    iters = 10000
    inflation = 2.5
    combined_radius = robot_radius * inflation
    while iters > 0:
        iters -= 1
        x = round(np.random.uniform(x_range + 18 * robot_radius, x_range + 19 * robot_radius), 2)
        y = round(np.random.uniform(20 * combined_radius, y_range - 20 * combined_radius), 2)
        pos_center = [x, y]
        is_intersect = False
        for poly_ob in poly_obs:
            if circle_polygon_intersect(pos_center, combined_radius, poly_ob.vertices_pos):
                is_intersect = True
                break
        if robs_pos:
            for rob_pos in robs_pos:
                if l2norm(rob_pos, pos_center) < combined_radius + combined_radius:
                    is_intersect = True
                    break
        if not is_intersect:
            robs_pos.append(pos_center)
        print(len(robs_pos), '------------------------current iters-------------------------', iters)
        if len(robs_pos) == rob_num:
            break
    print(len(robs_pos))
    print(robs_pos)
    robots = build_targets(robs_pos, robot_radius)
    env.plot_env(poly_obs, robots)
    return robs_pos


def build_cost_path_matrix(robs_info, tasks_pos, scale):
    u2t_dist_matrix = [[0. for _ in range(len(tasks_pos))] for _ in range(len(robs_info))]
    u2t_path_matrix = [[[] for _ in range(len(tasks_pos))] for _ in range(len(robs_info))]
    for ag_idx, ag in enumerate(robs_info):
        # print('agent', ag_idx)
        tuple_agpos = tuple(ag.pos_global_frame)
        for tar_idx, t_pos in enumerate(tasks_pos):
            # print('agent', ag_idx, 'target', tar_idx)
            tuple_tpos = tuple(t_pos)
            path = [tuple_agpos, tuple_tpos]

            path_result = [[float(round(po[0], 2)), float(round(po[1], 2))] for po in path]
            u2t_path_matrix[ag_idx][tar_idx] = path_result
            length = l2norm(tuple_agpos, tuple_tpos)
            u2t_dist_matrix[ag_idx][tar_idx] = float(round(length / scale, 5))

    t2t_dist_matrix = [[0. for _ in range(len(tasks_pos))] for _ in range(len(tasks_pos))]
    t2t_path_matrix = [[[] for _ in range(len(tasks_pos))] for _ in range(len(tasks_pos))]
    for i, i_pos in enumerate(tasks_pos):
        # print('task', i)
        tuple_ipos = tuple(i_pos)
        for j, j_pos in enumerate(tasks_pos):
            # print('task', i, j)
            tuple_jpos = tuple(j_pos)
            path = [tuple_ipos, tuple_jpos]
            path_result = [[float(round(po[0], 2)), float(round(po[1], 2))] for po in path]
            t2t_path_matrix[i][j] = path_result
            length = l2norm(tuple_ipos, tuple_jpos)
            t2t_dist_matrix[i][j] = float(round(length / scale, 5))
    t2u_dist_matrix = u2t_dist_matrix
    return u2t_dist_matrix, u2t_path_matrix, t2t_dist_matrix, t2t_path_matrix, t2u_dist_matrix, 0.5


def generate_tsp_file(rob, assignment, tasks_pos, dist_u2t, dist_t2t, filename):
    """
    生成基于显式距离矩阵的TSP文件
    """
    num_targets = len(assignment)
    num_nodes = 1 + num_targets

    # 计算距离矩阵（放大1000倍取整）
    distance_matrix = []
    for i in range(num_nodes):
        row = []
        for j in range(num_nodes):
            if i > 0 and j > 0:
                i_idx = int(tasks_pos[assignment[i - 1]][2])
                j_idx = int(tasks_pos[assignment[j - 1]][2])
                dist = dist_t2t[i_idx][j_idx]
            elif i == 0 and j > 0:
                i_idx = int(rob.ta_pos_[2])
                j_idx = int(tasks_pos[assignment[j - 1]][2])
                dist = dist_u2t[i_idx][j_idx]
            elif j == 0 and i > 0:
                i_idx = int(tasks_pos[assignment[i - 1]][2])
                j_idx = int(rob.ta_pos_[2])
                dist = dist_u2t[j_idx][i_idx]
            else:
                dist = 0.0
            dist_int = int(round(dist * 1000))
            row.append(str(dist_int))
        distance_matrix.append(row)

    # 写入TSP文件
    with open(filename, "w") as f:
        f.write("NAME : test1\n")
        f.write("TYPE : TSP\n")
        f.write(f"DIMENSION : {num_nodes}\n")
        f.write("EDGE_WEIGHT_TYPE : EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT : FULL_MATRIX\n")
        f.write("EDGE_WEIGHT_SECTION\n")

        # 写入矩阵数据
        for row in distance_matrix:
            f.write(" ".join(row) + "\n")

        f.write("EOF\n")


def generate_lkh_parameter_file(tsp_filename, param_filename, output_filename):
    """
    Generates a parameter file for LKH to solve the TSP.
    """
    param_data = f"""
    PROBLEM_FILE = {tsp_filename}
    OUTPUT_TOUR_FILE = {output_filename}
    RUNS = 1
    """
    with open(param_filename, "w") as f:
        f.write(param_data.strip())


def solve_with_lkh(param_filename):
    """
    Solves the TSP using LKH and returns the optimal path from the solution.
    """
    # Running the LKH solver with the generated parameter file
    command = f"LKH {param_filename}"
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Check for errors in LKH execution
    if result.returncode != 0:
        print(f"Error executing LKH: {result.stderr.decode()}")
        return None

    # print(f"LKH completed successfully. Output stored in {param_filename}")
    return True


def parse_solution_file(solution_file):
    """
    Parses the solution file from LKH and extracts the optimal tour.
    """
    indices = []
    with open(solution_file, 'r') as f:
        lines = f.readlines()

        # Skip lines until we reach the TOUR_SECTION part
        for line in lines:
            if line.strip() == "TOUR_SECTION":
                break

        # Now, start reading the tour section
        for line in lines:
            if line.strip() == "-1":
                break  # End of the tour section
            try:
                indices.append(int(line.strip()) - 2)  # Convert to 0-indexed and corresponding tasks_pos
            except ValueError:
                continue  # Skip lines that cannot be converted to an integer
    indices.pop(0)
    return indices


def lkh_solve_tsp(robots, tasks_pos, dist_u2t, dist_t2t):
    filename = "../../draw/resource/lkh_test.tsp"
    param_filename = "../../draw/resource/lkh_test.par"
    output_filename = "../../draw/resource/lkh_test.tour"

    # routes = []
    for rob in robots:
        assign = rob.p
        if len(assign) > 1:
            generate_tsp_file(rob, assign, tasks_pos, dist_u2t, dist_t2t, filename)
            generate_lkh_parameter_file(filename, param_filename, output_filename)
            if not solve_with_lkh(param_filename):      # Call LKH to solve the TSP
                print("Failed to solve TSP.")
                break

            # Parse the solution file and get the optimal tour
            indices = parse_solution_file(output_filename)
        elif len(assign) == 1:
            indices = [0]
        else:
            indices = []
        route = []
        for index in indices:
            route.append(assign[index])
        rob.p = route


def task_assign_scheme(robs_info, tasks_pos, scale, ta_policy, name=None, use_lkh=False):
    task_num = len(tasks_pos)
    robot_num = len(robs_info)
    if name is not None:  # 使用全局规划器计算避障路径和路径成本
        dist_u2t, path_u2t, dist_t2t, path_t2t, dist_t2u, turning = read_json(name)
    else:
        dist_u2t, path_u2t, dist_t2t, path_t2t, dist_t2u, turning = build_cost_path_matrix(robs_info, tasks_pos, scale)
    robots = [ta_policy() for _ in range(robot_num)]
    # starts = []
    for i, robot in enumerate(robots):
        robot.init_agent_tainfo(robs_info[i], tasks_pos, scale, dist_u2t, path_u2t, dist_t2t, path_t2t, dist_t2u,
                                turning)
        # starts.append(robot.pos)

    # score_time = 0.0
    # judge_time = 0.0
    ts = time.time()
    # phase1: 初始化形成智能体的任务样本集
    for ag in robots:
        ag.form_task_sample()
    # t0 = time.time()
    # judge_time += t0 - ts
    # phase2: 任务分配
    itea = 0
    while True:
        itea += 1
        # phase2.1: 更新智能体的wa*, ja*
        # t1 = time.time()
        for agent in robots:
            agent.update_wa_ja()
        # t2 = time.time()
        # score_time += t2 - t1
        # phase2.2: 智能体广播自己的id, wa*以及ja*信息
        message_pool = [agent.send_message() for agent in robots]
        # phase2.3: 智能体接收广播的id, wa*以及ja*信息
        for agent in robots:
            agent.receive_message(message_pool)
        # phase2.4: 最大一致性解决分配冲突，且进行任务分配
        for agent in robots:
            a_star, j_star_a_star = agent.consensus()
            agent.allocation_task(a_star, j_star_a_star)
        assigned_num = 0
        converged = []
        for agent in robots:
            assigned_num += len(agent.p)
            if agent.converged:
                converged.append(True)
        if assigned_num == task_num or sum(converged) == len(robots):
            break

        # print("-----------------iterations------------------", itea)

    if use_lkh:
        lkh_solve_tsp(robots, tasks_pos, dist_u2t, dist_t2t)

    te = time.time()

    for robot in robots:
        robot.path = []
        for j in range(len(robot.p)):
            if j == 0:  # first
                robot.path += robot.path_u2t[0][robot.p[j]]
            else:
                robot.path.pop()
                robot.path += robot.path_t2t[robot.p[j - 1]][robot.p[j]]
            if j == len(robot.p) - 1:  # last
                robot.path.pop()
                robot.path += robot.path_t2e[0][robot.p[j]]

    path_list = []
    assigned_num = 0
    dist_costs = []
    dist_straight = []
    rewards = 0
    all_path = []
    for i in range(len(robots)):
        rewards += robots[i].reward_score()
        dist_costs.append((robots[i].dist_cost + robots[i].dist_end) * robots[i].scale)
        # dist_costs.append(path_length(robots[i].path))

        path_list.append(robots[i].p)
        all_path.append(robots[i].path)
        dist_straight.append(path_length(robots[i].path))
        assigned_num += len(robots[i].p)
    return path_list, rewards, te - ts, dist_costs, all_path


def main():
    rob_rad = 2.5
    environ = env.Env(envs_type[5])
    agents_pos = generate_robots_pos(environ, rob_rad, rob_num=10)
    tars_pos = generate_targets_pos(environ, rob_rad, tar_radius=30.0, tar_num=100, scale=1.0)

    # U: 10; T: 100
    # robots_pos = [[6094.7, 3172.89], [6091.3, 3601.63], [6093.23, 880.49], [6090.04, 670.04], [6094.41, 3055.18], [6090.09, 1343.46], [6094.91, 3006.55], [6091.08, 906.28], [6094.76, 2677.22], [6090.56, 1505.46]]
    # tars_pos = [[1214.5, 1107.75], [1318.99, 1447.47], [4797.97, 2604.41], [2568.19, 881.22], [1210.0, 2966.42], [4396.25, 2144.72], [55.94, 3721.94], [50.59, 3020.45], [2488.19, 2490.43], [3521.48, 3959.98], [4519.05, 1051.77], [3567.33, 2706.94], [3430.24, 2087.26], [4178.89, 195.11], [5543.31, 3120.31], [4648.81, 1998.76], [240.59, 3421.15], [2060.05, 1645.66], [5834.39, 3429.75], [3288.45, 164.31], [1261.2, 2598.74], [1230.95, 33.92], [4634.84, 3822.95], [3507.06, 599.66], [3200.15, 2355.91], [5223.16, 123.38], [2697.76, 3656.24], [2412.52, 562.55], [5047.66, 3929.87], [3376.0, 1077.64], [1680.27, 363.9], [4078.91, 272.64], [4918.34, 2763.93], [5077.57, 524.01], [1707.73, 1410.98], [2813.84, 3082.02], [2564.15, 761.73], [615.15, 351.87], [112.94, 2632.54], [1986.11, 3440.9], [1106.84, 68.53], [443.0, 662.12], [1519.39, 1242.67], [355.84, 3974.17], [49.98, 2902.59], [5019.38, 652.13], [258.41, 2065.99], [1029.2, 2108.67], [5400.93, 3903.86], [5597.89, 2659.31], [4325.01, 54.81], [2012.84, 1027.77], [244.13, 3002.79], [5131.65, 3155.92], [5552.61, 2883.48], [3453.97, 1966.37], [2598.05, 2366.06], [1845.58, 3434.44], [4667.87, 2496.2], [2782.87, 1590.85], [1237.92, 3074.25], [4382.21, 1984.03], [1892.49, 2220.31], [2991.68, 3073.91], [5088.91, 413.45], [3209.21, 1182.69], [2811.02, 2640.32], [4400.89, 1025.69], [2138.53, 690.23], [5788.16, 3738.38], [4639.2, 999.45], [1097.2, 2617.92], [5625.55, 2164.41], [2640.29, 1363.68], [365.8, 558.7], [941.28, 3021.29], [3744.43, 3620.15], [196.9, 563.05], [130.43, 1865.35], [3514.78, 421.31], [3026.38, 2330.74], [16.44, 1905.38], [1726.61, 2191.61], [5707.36, 2744.03], [2297.9, 3092.39], [5769.11, 1938.53], [3125.51, 3678.75], [1836.31, 90.3], [2586.79, 3649.15], [3833.88, 455.94], [1581.75, 26.15], [4908.52, 2343.86], [3425.2, 164.41], [2758.84, 1802.93], [682.81, 1228.84], [4288.84, 2734.89], [5873.95, 3149.19], [1143.35, 2849.22], [4580.53, 1768.4], [3270.95, 3282.81]]

    robots_pos = agents_pos
    tars_pos = tars_pos
    write_env_cfg(rob_rad, environ, robots_pos, tars_pos, name='/large_u10t100.json', scale=1000)
    for k in range(len(robots_pos)):
        robots_pos[k].append(k)
    for k in range(len(tars_pos)):
        tars_pos[k].append(k)
    print(robots_pos)
    print(tars_pos)


if __name__ == "__main__":
    main()
    # a = 10
