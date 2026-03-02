import copy
import math
import numpy as np
import pandas as pd
import mamp.tools.rvo_math as rvo_math
from mamp.tools.vector import Vector2
from shapely.geometry import Polygon
from mamp.tools.utils import l2norm, get_boundaries, mod2pi, norm, pi_2_pi, is_intersect_polys, path_length
from mamp.tools.utils import robot_pose_to_outline, line_intersect_segment, point_line_dist, seg_is_intersec
from draw.plt2d import plt_visulazation, plot_half_planes1
from mamp.configs.config import USE_REAL_WORLD, USE_ROS, USE_TEMP


class Agent(object):
    def __init__(self, start_pos, goal_pos, radius, pref_speed, max_speed, min_speed, max_angular, policy, planner,
                 dt=0.1, start_yaw=np.pi, max_cap=0, max_range=0, expansion=0, ta_policy=None, swarm_num=0, scale=100,
                 task_area=None, exit_area=None):
        self.is_agent_ = True
        self.is_obstacle_ = False
        self.is_target_ = False

        # agent's physical information
        self.id_ = 0
        self.tid_ = {}
        self.radius_ = radius
        self.pref_speed_ = pref_speed
        self.max_speed_ = max_speed
        self.min_speed_ = min_speed
        self.max_angular_ = max_angular
        self.L = radius
        self.turning_radius_ = max(self.L / math.tan(max_angular), 0.45)  #
        self.cons_ang_ = max_speed / self.turning_radius_
        self.max_accel_ = 2.0
        self.time_step_ = dt
        self.neighbor_dist_ = 300.0
        self.max_neighbors_ = 10
        self.time_horizon_ = 10.0  # radius=2.5, 取值12
        self.time_horizon_obst_ = 10.0
        self.percept_dist_ = self.neighbor_dist_
        self.comm_range_ = 600.0
        self.max_load_ = max_cap
        self.max_endurance_ = max_range

        # agent's physical states
        self.initial_pos_ = Vector2(start_pos[0], start_pos[1])
        self.initial_heading_ = start_yaw
        self.position_ = Vector2(start_pos[0], start_pos[1])
        self.heading_yaw_ = self.initial_heading_
        self.outline_ = Polygon(robot_pose_to_outline(self.position_, self.heading_yaw_, self.radius_))
        self.goal_ = goal_pos
        self.goal_heading_ = -math.pi / 2
        self.velocity_ = Vector2()
        self.velocity_unicycle_ = np.array([0.0, 0.0], dtype='float64')
        self.pref_velocity_ = Vector2()
        self.agent_neighbors_ = []  # (float, Agent)
        self.obstacle_neighbors_ = []  # (float, Obstacle)
        self.neighbor_inflated_obs_ = []  # (inflated PolyObstacle)
        self.neighbor_obs_ = []  # (PolyObstacle)
        self.now_goal_ = None
        self.is_reach_goal_ = False
        self.is_travel_done_ = False
        self.is_collision_ = False
        self.is_out_of_max_time_ = False
        self.is_out_boundary_ = False
        self.stuck_num_ = 0

        self.swarm_state_ = {}
        self.swarm_assigned_scheme = {}
        self.send_message_ = []      # [target.id, other.id]

        self.policy_ = policy()
        self.new_velocity_ = Vector2()
        self.new_velocity_unicycle_ = np.array([0.0, 0.0], dtype='float64')
        self.planner_ = planner()
        self.ta_policy_ = ta_policy()
        self.scale_ = scale
        self.targets_ = []
        self.all_targets_ = {}
        self.visited_targets_ = []
        self.next_targets_idx = 0
        self.ta_pos_ = start_pos
        self.swarm_num_ = swarm_num

        self.path_ = []
        self.path_node_ = []  # 全局路径点序列
        self.max_travel_dist_ = self.to_goal_dist() if self.max_endurance_ == 0 else 1.05 * self.max_endurance_

        self.task_area_ = task_area
        self.ending_area_ = exit_area
        self.is_ending_area_ = True if len(exit_area) > 0 else False
        self.min_x_, self.max_x_, self.min_y_, self.max_y_ = get_boundaries(task_area)
        self.min_xe_, self.max_xe_, self.min_ye_, self.max_ye_ = get_boundaries(exit_area)

        self.reach_goal_threshold_ = self.radius_ * 0.3
        self.dist_update_now_goal = -1.0
        self.sampling_size_ = self.reach_goal_threshold_ * 4.5
        self.in_ending_area_ = False
        self.ending_step_ = 0

        self.travel_time = 0.0
        self.travel_dist = 0.0
        self.step_num = 0
        self.max_yaw_rate = 0.0
        self.max_path_plan_cost = 0.0
        self.solve_vel_cost = 0.0
        self.solve_ta_cost = 0.0
        self.all_ta_rewards = 0.0
        self.ta_initial_distance = 0.0
        self.completed_task_num = 0
        self.straight_path_length = rvo_math.l2norm(self.initial_pos_, self.goal_)  # For computing Distance Rate.
        self.desire_steps = int(self.straight_path_length / (pref_speed * dt))  # For computing Time Rate.
        self.history_pos = []
        self.history_speed = []

        self.ANIMATION_COLUMNS_ = ['pos_x', 'pos_y', 'yaw', 'vel_x', 'vel_y']
        self.history_info = pd.DataFrame(columns=self.ANIMATION_COLUMNS_)

    def send_message(self):
        return self.id_, self.position_, self.now_goal_, self.velocity_, self.ta_policy_.p, self.path_

    def receive_message(self, receive_info):
        for agent_info in receive_info:
            self.swarm_state_[agent_info[0]] = {}
            self.swarm_state_[agent_info[0]]['pos'] = agent_info[1]
            self.swarm_state_[agent_info[0]]['goal'] = agent_info[2]
            self.swarm_state_[agent_info[0]]['vel'] = agent_info[3]
            self.swarm_state_[agent_info[0]]['tars_id'] = agent_info[4]
            self.swarm_state_[agent_info[0]]['path'] = agent_info[5]

    def to_goal_dist(self):
        if self.path_:
            max_run_dist = 0.0
            for i in range(len(self.path_) - 1):
                max_run_dist += l2norm(self.path_[i], self.path_[i + 1])
            max_run_dist = max_run_dist * 6
        else:
            max_run_dist = 6 * rvo_math.l2norm(self.position_, self.goal_)
        return max_run_dist

    def compute_neighbors(self, kd_tree):
        """
        Computes the neighbors of this agent.
        """
        rangeSq = rvo_math.square(max(self.time_horizon_obst_ * self.max_speed_ + self.radius_, self.neighbor_dist_))
        self.obstacle_neighbors_ = []
        kd_tree.compute_obstacle_neighbors(self, rangeSq)

        self.agent_neighbors_ = []
        if self.max_neighbors_ > 0:
            rangeSq = rvo_math.square(self.neighbor_dist_)
            kd_tree.compute_agent_neighbors(self, rangeSq)

    def get_neighbor_inflated_obs(self):
        poly_id_set = set()
        self.neighbor_inflated_obs_ = []
        self.neighbor_obs_ = []
        for obj in self.obstacle_neighbors_:
            if obj[1].ob_id != -1 and obj[1].ob_id not in poly_id_set:
                poly_id_set.add(obj[1].ob_id)
                self.neighbor_inflated_obs_.append(self.planner_.inflated_obs[obj[1].ob_id])
                self.neighbor_obs_.append(self.planner_.poly_obs[obj[1].ob_id])

    def dynamics_constraints(self):
        new_vel = np.array([self.new_velocity_.x, self.new_velocity_.y])
        new_speed = norm(new_vel)
        new_yaw = mod2pi(math.atan2(new_vel[1], new_vel[0]))
        yaw_current = mod2pi(self.heading_yaw_)
        delta_theta = pi_2_pi(new_yaw - yaw_current)
        new_angular = delta_theta / self.time_step_

        if USE_ROS:
            if new_angular != 0 and new_speed / abs(new_angular) < self.turning_radius_:
                new_speed = min(new_speed, self.pref_speed_)
            if self.policy_.turning_left and not self.policy_.turning_right:
                delta_theta = 100.
            elif self.policy_.turning_right and not self.policy_.turning_left:
                delta_theta = -100.
            self.new_velocity_unicycle_ = np.array([new_speed, delta_theta])
        else:
            # 最小转弯半径约束
            if new_angular != 0 and new_speed / abs(new_angular) < self.turning_radius_:
                new_speed = min(new_speed, self.pref_speed_)
                omega = new_speed / self.turning_radius_
                if abs(delta_theta-np.pi) > 1e-3:
                    if self.policy_.turning_left and not self.policy_.turning_right:
                        new_angular = omega
                    elif self.policy_.turning_right and not self.policy_.turning_left:
                        new_angular = -omega
                    elif new_angular >= 0:
                        new_angular = omega
                    else:
                        new_angular = -omega
                else:
                    new_speed = self.velocity_unicycle_[0]
                    new_angular = self.velocity_unicycle_[1]
            new_yaw = new_angular * self.time_step_ + yaw_current

            if self.id_ == 6 and self.step_num >= 14500:
                print(delta_theta, new_angular)
                print(new_vel.tolist(), [new_speed * math.cos(new_yaw), new_speed * math.sin(new_yaw)])
                print('pref_velocity_', rvo_math.norm(self.pref_velocity_), 'new_speed', new_speed)
                # plot_half_planes1(self.pref_velocity_, self.new_velocity_, self.velocity_)
            self.new_velocity_ = Vector2(new_speed * math.cos(new_yaw), new_speed * math.sin(new_yaw))
            self.new_velocity_unicycle_ = np.array([new_speed, new_angular])
            # 靠近终点时停止
            new_position = self.position_ + self.time_step_ * self.new_velocity_
            if self.obstacle_neighbors_:
                obstacle1 = self.obstacle_neighbors_[0][1].point_
                obstacle2 = self.obstacle_neighbors_[0][1].next_.point_
                distance = rvo_math.dist_point_line_segment(obstacle1, obstacle2, new_position)
                if distance < 2 * self.planner_.inflation + self.radius_ and self.is_reach_goal_:
                    self.new_velocity_ = Vector2(0.0, 0.0)
                    self.new_velocity_unicycle_ = np.array([0.0, 0.0])

    def compute_new_velocity(self):
        """
        Computes the new velocity of this agent.
        """
        if self.id_ == 2:
            a = 0
        self.new_velocity_ = self.policy_.find_next_action(self)
        self.dynamics_constraints()

    def set_now_goal(self):
        self.now_goal_ = np.array(self.path_.pop(), dtype='float64')
        self.path_node_.append(self.now_goal_[:])

    def replanning_path(self):
        if self.now_goal_ is not None:
            pos = np.array([self.position_.x, self.position_.y])
            pos_next = self.now_goal_[:2]
            if is_intersect_polys(pos, pos_next, self.neighbor_obs_):
                # print(self.now_goal_, self.path_)
                paths = [self.path_ + [self.now_goal_, [self.position_.x, self.position_.y]]]
                agents = [self]
                for i in range(len(self.agent_neighbors_)):
                    other = self.agent_neighbors_[i][1]
                    agents.append(other)
                    paths.append([[other.position_.x, other.position_.y]])
                # plt_visulazation(paths, agents, [], self.planner_.poly_obs, self.task_area_, self.ending_area_)
                s_start = (self.goal_.x, self.goal_.y)  # path是逆序，此处起止位置互换正好不用再次逆序
                s_goal = (self.position_.x, self.position_.y)
                self.planner_.set_start_and_goal(s_start, s_goal)  # 起止位置必须是tuple型
                path = self.planner_.global_search()
                if path:
                    path[-1] = (path[-1][0], path[-1][1], 0)
                    path[0] = (self.goal_.x, self.goal_.y, 0)
                    self.path_ = path
                    self.path_node_.pop()
                    self.set_now_goal()
                # print(path)
                paths = [self.path_]
                agents = [self]
                for i in range(len(self.agent_neighbors_)):
                    other = self.agent_neighbors_[i][1]
                    agents.append(other)
                    paths.append([[other.position_.x, other.position_.y]])
                # plt_visulazation(paths, agents, [], self.planner_.poly_obs, self.task_area_, self.ending_area_)
                # if self.step_num >= 54:
                #     print(self.id_, self.now_goal_)
                #     plot_half_planes1(self.new_velocity_, self.velocity_, self.pref_velocity_)

    def is_in_ending_area(self):
        if self.is_ending_area_:
            min_xe, max_xe = self.min_xe_ + 2 * self.radius_, self.max_xe_ - 2 * self.radius_
            min_ye, max_ye = self.min_ye_ + 2 * self.radius_, self.max_ye_ - 2 * self.radius_
            is_in = min_xe < self.position_.x < max_xe and min_ye < self.position_.y < max_ye
        else:
            is_in = False
        return is_in

    def update_now_goal_from_path(self):
        self.replanning_path()
        if self.path_:
            if self.now_goal_ is None: self.set_now_goal()
            pos = np.array([self.position_.x, self.position_.y])
            dist2now_goal = l2norm(pos, self.now_goal_)
            while dist2now_goal <= self.percept_dist_:
                if self.path_:
                    pos_next = self.path_[-1]
                    is_near_now_goal = dist2now_goal <= self.dist_update_now_goal
                    if len(self.now_goal_) == 3:
                        if is_near_now_goal:
                            self.set_now_goal()
                            dist2now_goal = l2norm(pos, self.now_goal_)
                        else:
                            break
                    elif is_near_now_goal:
                        self.path_node_.pop()
                        self.set_now_goal()
                        dist2now_goal = l2norm(pos, self.now_goal_)
                    elif not is_intersect_polys(pos, pos_next, self.neighbor_inflated_obs_):
                        # if len(self.now_goal_) == 3 and self.now_goal_[2] == 2:
                        #     break
                        self.path_node_.pop()
                        self.set_now_goal()
                        dist2now_goal = l2norm(pos, self.now_goal_)
                    else:
                        break
                else:
                    self.now_goal_ = np.array([self.goal_.x, self.goal_.y, 0])
                    break
            # if dist2now_goal > self.percept_dist_:
            #     intersect_obs = is_intersect_polys(pos, self.now_goal_, self.neighbor_inflated_obs_)
            #     now_goal_dir = np.array(self.now_goal_[:2]) - pos
            #     vel = np.array([self.velocity_.x, self.velocity_.y])
            #     is_back = np.dot(vel, now_goal_dir) < 0
            #     if (intersect_obs or is_back) and len(self.now_goal_) < 2:
            #         self.set_now_goal()
        else:
            self.now_goal_ = np.array([self.goal_.x, self.goal_.y, 0])
        direction = Vector2(self.now_goal_[0], self.now_goal_[1]) - self.position_
        if len(self.now_goal_) == 3 and self.now_goal_[2] == 2 and direction @ self.velocity_ < 0:
            self.path_node_.pop()
            while True:
                self.now_goal_ = np.array(self.path_.pop(), dtype='float64')
                if len(self.now_goal_) == 2 or (len(self.now_goal_) == 3 and self.now_goal_[2] != 2):
                    self.path_node_.append(self.now_goal_[:])
                    break

    def set_preferred_velocity(self):
        """
        Set the preferred velocity to be a vector of unit magnitude (speed) in the direction of the goal.
        """
        # if self.targets_ and self.swarm_state_: self.pairwise_optimization()
        now_goal = Vector2(self.now_goal_[0], self.now_goal_[1])
        goal_vector = now_goal - self.position_
        distSq = rvo_math.abs_sq(goal_vector)
        c1 = self.in_ending_area_ and len(self.path_) == 0
        if (self.is_reach_goal_ or c1) and distSq > self.min_speed_ ** 2:   # 终点降低速度
            goal_vector = min(self.min_speed_, math.sqrt(distSq)) * rvo_math.normalize(goal_vector)
        elif distSq > self.max_speed_ ** 2 and (len(self.now_goal_) == 3 and self.now_goal_[2] == 2):  # 转弯降低速度
            goal_vector = self.max_speed_ * rvo_math.normalize(goal_vector)
        elif distSq > self.max_speed_ ** 2 or len(self.now_goal_) == 2 or (
                len(self.now_goal_) == 3 and self.now_goal_[2] != 0):
            goal_vector = self.max_speed_ * rvo_math.normalize(goal_vector)
        self.pref_velocity_ = goal_vector

    def path_t2t2e_dist(self, targets, nbr=None):
        targets_path_length = 0.0

        path_t2t = self.ta_policy_.path_t2t
        for i in range(len(targets)-1):
            targets_path_length += path_length(path_t2t[targets[i].id][targets[i+1].id])

        path_t2e = self.ta_policy_.path_t2e if nbr is None else nbr.ta_policy_.path_t2e
        targets_path_length += path_length(path_t2e[0][targets[-1].id])

        return targets_path_length

    # 最小化最大距离
    def pairwise_optimization(self):
        self_pos = (self.position_.x, self.position_.y)
        self_tar_pos = (self.targets_[0].position_.x, self.targets_[0].position_.y)
        self_cur_path = self.path_ + [self.now_goal_, self_pos]
        self_cur_dist = path_length(self_cur_path)
        path_t2t = copy.deepcopy(self.ta_policy_.path_t2t)
        for i in range(len(self.agent_neighbors_)):
            nbr = self.agent_neighbors_[i][1]
            nbr_id = nbr.id_
            nbr_targets = [self.all_targets_[idx] for idx in self.swarm_state_[nbr_id]['tars_id']]
            nbr_now_goal = self.swarm_state_[nbr_id]['goal']
            if not nbr_targets or nbr_now_goal is None: continue
            nbr_pos = (self.swarm_state_[nbr_id]['pos'].x, self.swarm_state_[nbr_id]['pos'].y)
            if seg_is_intersec(self_pos, self.now_goal_, nbr_pos, nbr_now_goal):
                nbr_tar_pos = (nbr_targets[0].position_.x, nbr_targets[0].position_.y)
                nbr_cur_path = [nbr_pos, nbr_now_goal] + self.swarm_state_[nbr_id]['path']
                nbr_cur_dist = path_length(nbr_cur_path)
                self.planner_.set_start_and_goal(self_pos, nbr_tar_pos)    # 本机当前位置到邻居目标位置路径
                self_new_path1 = self.planner_.global_search()
                self_new_dist1 = self.planner_.length
                self.planner_.set_start_and_goal(nbr_pos, self_tar_pos)    # 邻居当前位置到本机目标位置路径
                nbr_new_path1 = self.planner_.global_search()
                nbr_new_dist1 = self.planner_.length
                self_new_dist = self_new_dist1 + self.path_t2t2e_dist(nbr_targets)
                nbr_new_dist = nbr_new_dist1 + self.path_t2t2e_dist(self.targets_, nbr)
                cost_cur, cost_new = max(self_cur_dist, nbr_cur_dist), max(self_new_dist, nbr_new_dist)
                print('cost_cur, cost_new:', cost_cur, cost_new)

                if cost_cur > cost_new:    # 交换之后最大路程更短
                    self.targets_, nbr_targets = nbr_targets, self.targets_
                    self.ta_policy_.p = [tar.id for tar in self.targets_]
                    self.swarm_state_[self.id_]['tars_id'] = self.ta_policy_.p

                    self_new_path = []
                    self_new_path1.pop(0)
                    self_new_path1[-1] = self_new_path1[-1] + (0,)
                    self_new_path += self_new_path1
                    for k in range(0, len(self.targets_)-1):
                        path_seg = copy.deepcopy(path_t2t[self.targets_[k].id][self.targets_[k+1].id][:])
                        path_seg.pop(0)
                        path_seg[-1].append(0)
                        self_new_path += path_seg
                    path_seg = copy.deepcopy(self.ta_policy_.path_t2e[0][self.targets_[-1].id])
                    path_seg.pop(0)
                    path_seg[-1].append(0)
                    self_new_path += path_seg
                    self_new_path.reverse()
                    self.path_ = self_new_path
                    self.path_node_.pop()
                    # plt_visulazation([self.path_+[self_pos]], [self], self.targets_,
                    #                  self.planner_.poly_obs, self.task_area_, self.ending_area_)
                    self.set_now_goal()
                    # break

    def pairwise_allocation(self):
        if self.next_targets_idx > 0:
            # print(self.id_)
            if self.send_message_:
                self.send_message_ = []  # 一轮之后一定会有相应的机器人接收抛出的任务
            now_goal = Vector2(self.now_goal_[0], self.now_goal_[1])
            tar_pos = self.targets_[self.next_targets_idx].position_
            agent_tar_id = self.targets_[self.next_targets_idx].id
            if self.velocity_ @ self.pref_velocity_ < 0 and rvo_math.l2norm(now_goal, tar_pos) < rvo_math.EPSILON:
                for i in range(len(self.agent_neighbors_)):
                    other = self.agent_neighbors_[i][1]
                    other_tar_idx = other.next_targets_idx
                    if other_tar_idx < 0:
                        continue
                    max_j = len(other.targets_)
                    other_last_tar_id = other.targets_[-1].id
                    t2t_path = other.ta_policy_.path_t2t[other_last_tar_id][agent_tar_id]
                    p0 = Vector2(t2t_path[0][0], t2t_path[0][1])
                    p1 = Vector2(t2t_path[1][0], t2t_path[1][1])
                    if len(other.targets_) > 1:
                        other_previous_id = other.targets_[-2].id
                        t2t_path1 = other.ta_policy_.path_t2t[other_previous_id][other_last_tar_id]
                        p_last = Vector2(t2t_path1[-2][0], t2t_path1[-2][1])
                    else:
                        p_last = other.position_
                    vec1 = p0 - p_last
                    vec2 = p1 - p0
                    cj_score = vec1 @ vec2
                    if cj_score < 0:
                        continue
                    agent_path0 = self.path_ + [self.now_goal_[:2], [self.position_.x, self.position_.y]]
                    agent_distance0 = self.travel_dist + path_length(agent_path0)
                    other_path0 = other.path_ + [other.now_goal_[:2], [other.position_.x, other.position_.y]]
                    other_distance0 = other.travel_dist + path_length(other_path0)
                    last_tar_id = self.targets_[self.next_targets_idx-1].id
                    agent_tar_ids = [self.targets_[k].id for k in range(len(self.targets_))]
                    for k in range(self.next_targets_idx+1):
                        agent_tar_ids.pop(0)

                    if agent_tar_ids:
                        agent_path1 = copy.deepcopy(self.ta_policy_.path_t2t[last_tar_id][agent_tar_ids[0]])
                        agent_path1.pop(0)
                        agent_path1[-1].append(0)
                        for k in range(1, len(agent_path1)):
                            agent_path1.pop()
                            path_seg = copy.deepcopy(self.ta_policy_.path_t2t[agent_tar_ids[k - 1]][agent_tar_ids[k]][:])
                            path_seg[0].append(0)
                            path_seg[-1].append(0)
                            agent_path1 += path_seg
                        agent_path1.pop()
                        path_seg = copy.deepcopy(self.ta_policy_.path_t2e[0][agent_tar_ids[-1]])
                        path_seg[0].append(0)
                        path_seg[-1].append(0)
                        agent_path1 += path_seg
                    else:
                        agent_path1 = copy.deepcopy(self.ta_policy_.path_t2e[0][last_tar_id])
                        agent_path1.pop(0)
                        agent_path1[-1].append(0)

                    agent_path1.reverse()
                    agent_path1.append([self.position_.x, self.position_.y])
                    agent_distance1 = self.travel_dist + path_length(agent_path1)

                    other_tar_ids = [other.targets_[k].id for k in range(len(other.targets_))]
                    other_tar_ids.append(agent_tar_id)
                    last_tar_id = other.targets_[other.next_targets_idx-1].id
                    for k in range(other.next_targets_idx):
                        other_tar_ids.pop(0)

                    other_path1 = copy.deepcopy(other.ta_policy_.path_t2t[last_tar_id][other_tar_ids[0]][:])
                    while True:
                        node = other_path1.pop(0)
                        if l2norm(node, other.now_goal_[:2]) < rvo_math.EPSILON:
                            other_path1.insert(0, node)
                            break
                    other_path1[-1].append(0)
                    for k in range(1, len(other_tar_ids)):
                        other_path1.pop()
                        path_seg = copy.deepcopy(other.ta_policy_.path_t2t[other_tar_ids[k - 1]][other_tar_ids[k]])
                        path_seg[0].append(0)
                        path_seg[-1].append(0)
                        other_path1 += path_seg
                    other_path1.pop()
                    path_seg = copy.deepcopy(other.ta_policy_.path_t2e[0][other_tar_ids[-1]])
                    path_seg[0].append(0)
                    path_seg[-1].append(0)
                    other_path1 += path_seg
                    other_path1.reverse()
                    other_path1.append([other.position_.x, other.position_.y])
                    other_distance1 = other.travel_dist + path_length(other_path1)
                    c1 = agent_distance0 + other_distance0 > agent_distance1 + other_distance1
                    if c1:
                        self.send_message_ = [self.targets_.pop(self.next_targets_idx), max_j, other.id_]
                        if self.next_targets_idx == len(self.targets_):
                            self.next_targets_idx -= 1
                        self.path_ = agent_path1
                        self.path_.pop()
                        self.set_now_goal()
                        break

            for i in range(len(self.agent_neighbors_)):
                other = self.agent_neighbors_[i][1]
                if other.send_message_ and other.send_message_[2] == self.id_:
                    self.targets_.insert(other.send_message_[1], other.send_message_[0])
                    tar_ids = [self.targets_[k].id for k in range(len(self.targets_))]
                    for k in range(self.next_targets_idx):
                        tar_ids.pop(0)
                    last_tar_id = self.targets_[self.next_targets_idx - 1].id
                    path = copy.deepcopy(self.ta_policy_.path_t2t[last_tar_id][tar_ids[0]][:])
                    while True:
                        node = path.pop(0)
                        if l2norm(node, self.now_goal_[:2]) < rvo_math.EPSILON:
                            path.insert(0, node)
                            break
                    path[-1].append(0)
                    for k in range(1, len(tar_ids)):
                        path.pop()
                        path_seg = copy.deepcopy(self.ta_policy_.path_t2t[tar_ids[k - 1]][tar_ids[k]])
                        path_seg[0].append(0)
                        path_seg[-1].append(0)
                        path += path_seg
                    path.pop()
                    path_seg = copy.deepcopy(self.ta_policy_.path_t2e[0][tar_ids[-1]])
                    path_seg[0].append(0)
                    path_seg[-1].append(0)
                    path += path_seg
                    path.reverse()
                    self.path_ = path
                    self.set_now_goal()
                    break

    def insert_agent_neighbor(self, agent, rangeSq):
        """
        Inserts an agent neighbor into the set of neighbors of this agent.

        Args:
            agent (Agent): A pointer to the agent to be inserted.
            rangeSq (float): The squared range around this agent.
        """
        if self != agent:
            distSq = rvo_math.abs_sq(self.position_ - agent.position_)

            if distSq < rangeSq:
                if len(self.agent_neighbors_) < self.max_neighbors_:
                    self.agent_neighbors_.append((distSq, agent))

                i = len(self.agent_neighbors_) - 1
                while i != 0 and distSq < self.agent_neighbors_[i - 1][0]:
                    self.agent_neighbors_[i] = self.agent_neighbors_[i - 1]
                    i -= 1

                self.agent_neighbors_[i] = (distSq, agent)

                if len(self.agent_neighbors_) == self.max_neighbors_:
                    rangeSq = self.agent_neighbors_[len(self.agent_neighbors_) - 1][0]
        return rangeSq

    def insert_obstacle_neighbor(self, obstacle, rangeSq):
        """
        Inserts a static obstacle neighbor into the set of neighbors of this agent.

        Args:
            obstacle (Obstacle): The number of the static obstacle to be inserted.
            rangeSq (float): The squared range around this agent.
        """
        nextObstacle = obstacle.next_
        distSq = rvo_math.dist_sq_point_line_segment(obstacle.point_, nextObstacle.point_, self.position_)

        if distSq < rangeSq:
            self.obstacle_neighbors_.append((distSq, obstacle))

            i = len(self.obstacle_neighbors_) - 1

            while i != 0 and distSq < self.obstacle_neighbors_[i - 1][0]:
                self.obstacle_neighbors_[i] = self.obstacle_neighbors_[i - 1]
                i -= 1

            self.obstacle_neighbors_[i] = (distSq, obstacle)

    def to_vector(self):
        """ Convert the agent's attributes to a single global state vector. """
        global_state_dict = {
            'pos_x': self.position_.x,
            'pos_y': self.position_.y,
            'vel_x': self.velocity_.x,
            'vel_y': self.velocity_.y,
            'yaw': self.heading_yaw_,
        }
        animation_columns_dict = {}
        for key in self.ANIMATION_COLUMNS_:
            animation_columns_dict.update({key: global_state_dict[key]})

        self.history_info = pd.concat([self.history_info, pd.DataFrame([animation_columns_dict])], ignore_index=True)

    def check_visited_target(self):
        if self.targets_ and rvo_math.l2norm(self.position_, self.targets_[0].position_) < self.dist_update_now_goal:
            if not self.targets_[0].is_visited:
                self.targets_[0].is_visited = True
                self.visited_targets_.append(self.targets_.pop(0))

    def check_task(self):
        idx = self.next_targets_idx
        if idx != 0 and idx >= len(self.targets_) - 1:
            pass
        elif self.targets_ and rvo_math.l2norm(self.position_, self.targets_[idx].position_) < self.dist_update_now_goal:
            if not self.targets_[idx].is_visited:
                self.targets_[idx].is_visited = True
                if self.next_targets_idx + 1 < len(self.targets_):
                    self.next_targets_idx += 1

    def update_state(self):
        """
        Updates the state of this agent.
        """
        if self.id_ == 2:
            a = 10
        # self.dynamics_constraints()

        speed = self.new_velocity_unicycle_[0]
        heading = pi_2_pi(self.heading_yaw_ + self.new_velocity_unicycle_[1] * self.time_step_)
        dx = speed * math.cos(heading) * self.time_step_
        dy = speed * math.sin(heading) * self.time_step_
        self.velocity_ = self.new_velocity_
        self.velocity_unicycle_ = self.new_velocity_unicycle_
        self.position_ += Vector2(dx, dy)
        self.heading_yaw_ = heading
        self.outline_ = Polygon(robot_pose_to_outline(self.position_, self.heading_yaw_, self.radius_))
        self.history_pos.append([self.position_.x, self.position_.y])
        self.in_ending_area_ = self.is_in_ending_area()

        length = np.sqrt(dx ** 2 + dy ** 2)
        if length < 1e-3:
            self.stuck_num_ += 1
        if self.stuck_num_ == 300:
            self.is_out_of_max_time_ = True
        if self.is_reach_goal_ and len(self.path_) == 0:  # 进入终止区域后不再计算旅行时间和距离
            travel_time = 0.0
            length = 0.0
        else:
            travel_time = self.time_step_
        self.travel_dist += length
        self.travel_time += travel_time
        self.max_yaw_rate = max(abs(self.new_velocity_unicycle_[1]), self.max_yaw_rate)
        # self.check_task()
        self.check_visited_target()
        self.to_vector()

    def update_position_from_pose(self, position, euler, speed, save=True):
        if not self.is_collision_:      # not self.is_reach_goal_ and
            position_old = self.position_
            # self.position_ = Vector2(position[0], position[1])
            self.heading_yaw_ = euler[2]
            LENGTH = 0.30  # [m]
            BACK_TO_WHEEL = 0.05  # [m]
            l_norm = (LENGTH / 2) - BACK_TO_WHEEL
            x = position[0] + l_norm * math.cos(self.heading_yaw_)
            y = position[1] + l_norm * math.sin(self.heading_yaw_)
            self.position_ = Vector2(x, y)
            self.outline_ = Polygon(robot_pose_to_outline(self.position_, self.heading_yaw_, self.radius_))
            self.velocity_ = Vector2(speed * np.cos(euler[2]), speed * np.sin(euler[2]))
            self.velocity_unicycle_ = np.array([speed, euler[2]])
            self.history_pos.append(position)
            self.in_ending_area_ = self.is_in_ending_area()

            dist_goal = rvo_math.l2norm(self.position_, self.goal_)
            length = rvo_math.l2norm(position_old, self.position_)
            if speed < 1e-5 and dist_goal < self.reach_goal_threshold_:
                travel_time = 0.0
                length = 0.0
            else:
                travel_time = self.time_step_
            self.travel_dist += length
            self.travel_time += travel_time
            # self.check_task()
            self.check_visited_target()
        if save:
            self.to_vector()
