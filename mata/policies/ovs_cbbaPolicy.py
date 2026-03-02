import time
import copy
import numpy as np
from mamp.envs import env
from draw.plt2d import plt_visulazation
from mamp.tools.utils import l2norm, path_length
from mamp.configs.config import envs_type
from mata.ta_config.general_config import build_agents_and_tasks, read_json, build_cost_path_matrix
from mata.ta_config.general_config import build_obj_pos_large


class CBBAPolicy(object):
    def __init__(self):
        self.str = 'vescbba'
        self.task_num = -1
        self.agent_num = -1
        self.robs_nest = None
        self.obstacles = []
        self.scale = -1
        self.test_num = 0

        # Agent information
        self.id = -1
        self.vel = 1.0

        self.assigned_result = []
        self.agents_assigned_scheme = {}

        # Local Winning Agent List
        self.z = []
        # Local Winning Bid List
        self.y = []
        # Bundle
        self.b = []
        # Path
        self.p = []
        self.path = []
        # Maximum Task Number
        self.L_t = -1
        # Maximum Distance
        self.L_dist = -1.0
        self.dist_end = 0.0
        self.dist_cost = 0.0
        # Local Clock
        self.time_step = 0
        # Time Stamp List
        self.s = {}

        # This part can be modified depend on the problem
        self.pos = None  # Agent State (Position)
        self.c = []  # Initial Score (Euclidean Distance)

        self.tasks_pos = []

        # score function parameters
        self.Lambda = 0.95
        self.c_bar = None

        self.Y = None

        # distance info
        self.x_range, self.y_range = -1.0, -1.0
        self.dis_u2t = None
        self.path_u2t = None
        self.dis_t2t = None
        self.path_t2t = None
        self.dis_t2e = None
        self.path_t2e = None
        self.dist_turning = None

    def init_agent_tainfo(self, agent, tasks_pos, scale, dist_u2t, path_u2t, dist_t2t, path_t2t, dist_t2u, turning):
        # agent info
        self.id = agent.id_
        self.vel = 1.0
        self.L_t = agent.max_load_
        self.L_dist = agent.max_endurance_
        self.agent_num = agent.swarm_num_
        self.pos = agent.ta_pos_
        self.robs_nest = agent.ending_area_
        self.x_range, self.y_range = (agent.min_x_, agent.max_x_), (agent.min_y_, agent.max_y_)
        self.assigned_result = [[] for _ in range(self.agent_num)]
        self.agents_assigned_scheme = {}

        # task info
        self.tasks_pos = tasks_pos
        self.task_num = len(tasks_pos)
        self.c_bar = np.ones(self.task_num)

        # Local Winning Agent List
        self.z = np.ones(self.task_num, dtype=np.int16) * self.id
        # Local Winning Bid List
        self.y = np.array([0 for _ in range(self.task_num)], dtype=np.float64)
        # Time Stamp List
        self.s = {a: self.time_step for a in range(self.agent_num)}
        self.c = np.zeros(self.task_num)  # Initial Score (Euclidean Distance)

        # environment info
        self.obstacles = []
        self.scale = scale
        self.dis_u2t = np.zeros((1, self.task_num))
        self.path_u2t = [[[] for _ in range(self.task_num)]]
        self.dis_t2t = np.zeros((self.task_num, self.task_num))
        self.path_t2t = [[[] for _ in range(self.task_num)] for _ in range(self.task_num)]
        self.dis_t2e = np.zeros((1, self.task_num))
        self.path_t2e = [[[] for _ in range(self.task_num)]]
        self.dist_turning = turning

        for i in range(1):
            ag_idx = int(self.pos[2])
            for j in range(len(self.tasks_pos)):
                tar_idx = int(self.tasks_pos[j][2])
                self.dis_u2t[i][j] = dist_u2t[ag_idx][tar_idx]
                self.dis_t2e[i][j] = dist_t2u[ag_idx][tar_idx]
                self.path_u2t[i][j] = copy.deepcopy(path_u2t[ag_idx][tar_idx])
                self.path_t2e[i][j] = path_u2t[ag_idx][tar_idx]
                self.path_t2e[i][j].reverse()

        for i in range(len(self.tasks_pos)):
            tar_iid = int(self.tasks_pos[i][2])
            for j in range(len(self.tasks_pos)):
                tar_jid = int(self.tasks_pos[j][2])
                self.dis_t2t[i][j] = dist_t2t[tar_iid][tar_jid]
                self.path_t2t[i][j] = path_t2t[tar_iid][tar_jid]

    def tau(self, j):
        # Estimate time agent will take to arrive at task j's location
        # This function can be used in later
        pass

    def send_message(self):
        """
        Return local winning bid list
        [output]
        y: winning bid list (list:task_num)
        z: winning agent list (list:task_num)
        s: Time Stamp List (Dict:{agent_id:update_time})
        """
        return self.y.tolist(), self.z.tolist(), self.s

    def receive_message(self, Y):
        self.Y = Y

    def build_bundle(self):
        """
        Construct bundle and path list with local information
        """
        J = [j for j in range(self.task_num)]

        while len(self.b) < self.L_t:
            # Calculate S_p for constructed path list
            S_p = 0  # The total reward obtained by agent i while executing the task along path p_i
            if len(self.p) > 0:
                distance_j = 0
                distance_j += self.dis_u2t[0][self.p[0]]
                S_p += (self.Lambda ** (distance_j / self.vel)) * self.c_bar[self.p[0]]
                for p_idx in range(len(self.p) - 1):
                    distance_j += self.dis_t2t[self.p[p_idx]][self.p[p_idx + 1]]
                    S_p += (self.Lambda ** (distance_j / self.vel)) * self.c_bar[self.p[p_idx + 1]]

            # Calculate c_ij for each task j
            best_pos = {}
            for j in J:
                c_list = []
                if j in self.b:  # If already in bundle list
                    self.c[j] = 0  # Minimum Score
                else:
                    for n in range(len(self.p) + 1):
                        p_temp = copy.deepcopy(self.p)
                        p_temp.insert(n, j)
                        c_temp = 0
                        distance_j = 0
                        distance_j += self.dis_u2t[0][p_temp[0]]
                        c_temp += (self.Lambda ** (distance_j / self.vel)) * self.c_bar[p_temp[0]]
                        if len(p_temp) > 1:
                            for p_loc in range(len(p_temp) - 1):
                                distance_j += self.dis_t2t[p_temp[p_loc]][p_temp[p_loc + 1]]
                                c_temp += (self.Lambda ** (distance_j / self.vel)) * self.c_bar[p_temp[p_loc + 1]]

                        c_jn = c_temp - S_p
                        f_dist = distance_j * self.scale + self.dis_t2e[0][p_temp[-1]] * self.scale
                        if f_dist > self.L_dist and self.L_dist < 1000:  # Maximum travel distance constraint
                            c_jn = 0.001
                        c_list.append(c_jn)

                    max_idx = int(np.argmax(c_list))
                    c_j = c_list[max_idx]
                    self.c[j] = c_j
                    best_pos[j] = max_idx

            h = (self.c > self.y)
            if sum(h) == 0:
                break
            self.c[~h] = 0
            J_i = int(np.argmax(self.c))
            n_J = best_pos[J_i]

            self.b.append(J_i)
            self.p.insert(n_J, J_i)

            self.y[J_i] = self.c[J_i]
            self.z[J_i] = self.id

    def update_task(self):
        """
        [input]
        Y: winning bid lists from neighbors (dict:{neighbor_id:(winning bid_list, winning agent list, time stamp list)})
        time: for simulation,
        """

        old_p = copy.deepcopy(self.p)

        id_list = list(self.Y.keys())
        id_list.insert(0, self.id)

        # Update time list
        for id in list(self.s.keys()):
            if id in id_list:
                self.s[id] = self.time_step
            else:
                s_list = []
                for neighbor_id in id_list[1:]:
                    s_list.append(self.Y[neighbor_id][2][id])
                if len(s_list) > 0:
                    self.s[id] = max(s_list)

        # Update Process
        for j in range(self.task_num):
            for k in id_list[1:]:
                y_k = self.Y[k][0]
                z_k = self.Y[k][1]
                s_k = self.Y[k][2]

                z_ij = self.z[j]
                z_kj = z_k[j]
                y_kj = y_k[j]

                i = self.id
                y_ij = self.y[j]

                # Rule Based Update
                # Rule 1~4
                if z_kj == k:
                    # Rule 1
                    if z_ij == self.id:
                        if y_kj > y_ij:
                            self.__update(j, y_kj, z_kj)
                        elif abs(y_kj - y_ij) < np.finfo(float).eps:  # Tie Breaker
                            if k < self.id:
                                self.__update(j, y_kj, z_kj)
                        else:
                            self.__leave()
                    # Rule 2
                    elif z_ij == k:
                        self.__update(j, y_kj, z_kj)
                    # Rule 3
                    elif z_ij != -1:
                        m = z_ij
                        if (s_k[m] > self.s[m]) or (y_kj > y_ij):
                            self.__update(j, y_kj, z_kj)
                        elif abs(y_kj - y_ij) < np.finfo(float).eps:  # Tie Breaker
                            if k < self.id:
                                self.__update(j, y_kj, z_kj)
                    # Rule 4
                    elif z_ij == -1:
                        self.__update(j, y_kj, z_kj)
                    else:
                        raise Exception("Error while updating")
                # Rule 5~8
                elif z_kj == i:
                    # Rule 5
                    if z_ij == i:
                        self.__leave()
                    # Rule 6
                    elif z_ij == k:
                        self.__reset(j)
                    # Rule 7
                    elif z_ij != -1:
                        m = z_ij
                        if s_k[m] > self.s[m]:
                            self.__reset(j)
                    # Rule 8
                    elif z_ij == -1:
                        self.__leave()
                    else:
                        raise Exception("Error while updating")
                # Rule 9~13
                elif z_kj != -1:
                    m = z_kj
                    # Rule 9
                    if z_ij == i:
                        if (s_k[m] >= self.s[m]) and (y_kj > y_ij):
                            self.__update(j, y_kj, z_kj)
                        elif (s_k[m] >= self.s[m]) and (abs(y_kj - y_ij) < np.finfo(float).eps):  # Tie Breaker
                            if m < self.id:
                                self.__update(j, y_kj, z_kj)
                    # Rule 10
                    elif z_ij == k:
                        if s_k[m] > self.s[m]:
                            self.__update(j, y_kj, z_kj)
                        else:
                            self.__reset(j)
                    # Rule 11
                    elif z_ij == m:
                        if s_k[m] > self.s[m]:
                            self.__update(j, y_kj, z_kj)
                    # Rule 12
                    elif z_ij != -1:
                        n = z_ij
                        if (s_k[m] > self.s[m]) and (s_k[n] > self.s[n]):
                            self.__update(j, y_kj, z_kj)
                        elif (s_k[m] > self.s[m]) and (y_kj > y_ij):
                            self.__update(j, y_kj, z_kj)
                        elif (s_k[m] > self.s[m]) and (abs(y_kj - y_ij) < np.finfo(float).eps):  # Tie Breaker
                            if m < n:
                                self.__update(j, y_kj, z_kj)
                        elif (s_k[n] > self.s[n]) and (self.s[m] > s_k[m]):
                            self.__update(j, y_kj, z_kj)
                    # Rule 13
                    elif z_ij == -1:
                        if s_k[m] > self.s[m]:
                            self.__update(j, y_kj, z_kj)
                    else:
                        raise Exception("Error while updating")
                # Rule 14~17
                elif z_kj == -1:
                    # Rule 14
                    if z_ij == i:
                        self.__leave()
                    # Rule 15
                    elif z_ij == k:
                        self.__update(j, y_kj, z_kj)
                    # Rule 16
                    elif z_ij != -1:
                        m = z_ij
                        if s_k[m] > self.s[m]:
                            self.__update(j, y_kj, z_kj)
                    # Rule 17
                    elif z_ij == -1:
                        self.__leave()
                    else:
                        raise Exception("Error while updating")
                else:
                    raise Exception("Error while updating")

        n_bar = len(self.b)
        # Get n_bar
        for n in range(len(self.b)):
            b_n = self.b[n]
            if self.z[b_n] != self.id:
                n_bar = n
                break

        b_idx1 = copy.deepcopy(self.b[n_bar + 1:])

        if len(b_idx1) > 0:
            self.y[b_idx1] = 0
            self.z[b_idx1] = -1

        if n_bar < len(self.b):
            del self.b[n_bar:]

        self.p = []
        for task in self.b:
            self.p.append(task)

        self.time_step += 1

        converged = False
        if old_p == self.p:
            converged = True

        return converged

    def __update(self, j, y_kj, z_kj):
        """
        Update values
        """
        self.y[j] = y_kj
        self.z[j] = z_kj

    def __reset(self, j):
        """
        Reset values
        """
        self.y[j] = 0
        self.z[j] = -1  # -1 means "none"

    def __leave(self):
        """
        Do nothing
        """
        pass

    def reward_score(self):
        return self.dist_cost


def task_assign_cbba(robs_info, tasks_pos, scale, ta_policy, name=None):
    task_num = len(tasks_pos)
    robot_num = len(robs_info)
    if name is not None:  # Use the global planner to compute the collision-free path and the path cost.
        dist_u2t, path_u2t, dist_t2t, path_t2t, dist_t2u, turning = read_json(name)
    else:
        dist_u2t, path_u2t, dist_t2t, path_t2t, dist_t2u, turning = build_cost_path_matrix(robs_info, tasks_pos, scale)

    robots = [ta_policy() for _ in range(robot_num)]
    starts = []
    for i, robot in enumerate(robots):
        robot.init_agent_tainfo(robs_info[i], tasks_pos, scale, dist_u2t, path_u2t, dist_t2t, path_t2t, dist_t2u,
                                turning)
        starts.append(robot.pos)

    # Network Initialize
    G = np.ones((robot_num, robot_num))  # Fully connected network

    t = 0  # Iteration number
    max_time = 600.0
    # bundle_list = [[] for _ in range(robot_num)]
    # path_list = [[] for _ in range(robot_num)]
    ts = time.time()
    while True:
        converged_list = []  # Converged List

        # Phase 1: Auction Process
        for robot in robots:
            # select task by local information
            robot.build_bundle()
        # Communicating: Send winning bid list to neighbors (depend on env)
        Y = None
        message_pool = [robot.send_message() for robot in robots]

        for robot_id, robot in enumerate(robots):
            # Receive winning bid list from neighbors
            g = G[robot_id]

            connected, = np.where(g == 1)
            connected = list(connected)
            connected.remove(robot_id)

            if len(connected) > 0:
                Y = {neighbor_id: message_pool[neighbor_id] for neighbor_id in connected}
            else:
                Y = None

            robot.receive_message(Y)

        # Phase 2: Consensus Process
        for robot in robots:
            # Update local information and decision
            if Y is not None:
                converged = robot.update_task()
                converged_list.append(converged)

        t += 1
        t2 = time.time()
        if sum(converged_list) == robot_num or t2 - ts > max_time:
            break
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
        robot.dist_cost = path_length(robot.path) / scale
        robot.dist_end = 0.0

    path_list = []
    assigned_num = 0
    dist_costs = []
    rewards = 0
    all_path = []
    for i in range(len(robots)):
        dist_costs.append((robots[i].dist_cost + robots[i].dist_end) * scale)
        rewards += robots[i].reward_score()
        path_list.append(robots[i].p)
        all_path.append(robots[i].path)
        assigned_num += len(robots[i].p)
    # print("assigned_num: ", assigned_num)
    return path_list, rewards, te - ts, dist_costs, all_path


def main():
    # large blocks scenario
    envs = env.Env(envs_type[5])
    agents, task_area, tars_pos, scl = build_obj_pos_large(envs, ag_num=10, tar_num=100)
    name = '/large_u10t100.json'

    tar_ids, res, allocost, dists, all_paths = task_assign_cbba(agents, tars_pos, scl, CBBAPolicy, name)

    # print("Time used [sec]: ", allocost)
    # print("all_rewards: ", res)
    print("max travel distance: ", max(dists), dists)
    print("tar_id_list: ", tar_ids)

    # plot
    all_length = 0.0
    for k in range(len(all_paths)):
        if len(all_paths[k]) == 0:
            all_paths[k].append(agents[k].pos_global_frame)
        length = 0.0
        for n in range(len(all_paths[k]) - 1):
            length += l2norm(all_paths[k][n], all_paths[k][n + 1])
        all_length += length
    print(all_length)
    assigned_num = 0
    empty_num = 0
    for tars in tar_ids:
        assigned_num += len(tars)
        if len(tars) == 0:
            empty_num += 1
    print("Time used [sec]: ", allocost)
    # print("all_rewards: ", res)
    # print("sum all distance: ", sum(dists))
    # print("all path length: ", all_length)
    print("assigned_num", assigned_num)
    print("empty num: ", empty_num)

    robs_nest = agents[0].ending_area_
    robs, tasks = build_agents_and_tasks(agents, tars_pos, tar_ids, envs)

    plt_visulazation(all_paths, robs, tasks, envs.poly_obs, task_area, robs_nest)


if __name__ == "__main__":
    main()
