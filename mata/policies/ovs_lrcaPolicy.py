"""
@ Author: Gang Xu
@ Date: 2024.02.29
@ Function:
"""
import copy
import numpy as np
from mamp.envs import env
from mamp.tools.utils import l2norm, seg_is_intersec, dis_UT, dis_TT, dist_sector
from draw.plt2d import plt_visulazation
from mamp.configs.config import envs_type
from mata.ta_config.general_config import build_agents_and_tasks, task_assign_scheme, build_obj_pos_large


class LRCAPolicy(object):
    def __init__(self):
        self.str = 'ovslrca'
        self.Na = []  # Agent a's own task sample set
        self.Wa = []  # Marginal utilities of all tasks in Na
        self.Ta = []  # Task set assigned to agent a
        self.p = []  # Assignment result for this agent, storing task IDs in visiting order
        self.current_rewards = 0.
        self.current_distance = 0.
        self.dist_end = 0.0
        self.dist_cost = 0.0
        self.path = []
        self.wa_star = None  # Maximum marginal utility of agent a
        self.ja_star = None  # Task ID corresponding to the maximum marginal utility
        self.assigned_result = []  # Assignment results of all active agents in the environment, storing task IDs in visiting order
        self.agents_assigned_scheme = {}
        self.converged = False
        self.Lambda = 0.95
        self.receive_info = None

        # agent info
        self.id = -1
        self.vel = 1.0
        self.L_t = -1
        self.L_dist = -1.
        self.agent_num = -1
        self.pos = None
        self.robs_nest = None
        self.x_range, self.y_range = -1., -1.

        # task info
        self.task_num = -1
        self.tasks_pos = []
        self.c_bar = None

        # environment info
        self.obstacles = []
        self.scale = -1
        self.dis_u2t = None  # Initialize matrix information
        self.path_u2t = None
        self.dis_t2t = None  # Initialize matrix information
        self.dis_t2t1 = None
        self.dis_sector = None
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

        # environment info
        self.obstacles = []
        self.scale = scale
        self.dis_u2t = np.zeros((1, self.task_num))  # Initialize matrix information
        self.path_u2t = [[[] for _ in range(self.task_num)]]
        self.dis_t2t = np.zeros((self.task_num, self.task_num))  # Initialize matrix information
        self.dis_t2t1 = dis_TT(tasks_pos)
        self.dis_sector = dist_sector(agent.turning_radius_)
        self.path_t2t = [[[] for _ in range(self.task_num)] for _ in range(self.task_num)]
        self.dis_t2e = np.zeros((1, self.task_num))  # Initialize matrix information
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

    def send_message(self):
        return self.id, self.ja_star, self.wa_star, self.pos, self.p

    def receive_message(self, receive_info):
        self.receive_info = receive_info
        for agent_info in receive_info:
            self.agents_assigned_scheme[agent_info[0]] = agent_info[4]

    def form_task_sample(self):
        self.Na = []  # Agent a's own task sample set
        self.Wa = []  # Marginal utilities of all tasks in Na
        self.Ta = []  # Task set assigned to agent a
        self.p = []  # Assignment result for this agent, storing task IDs in visiting order
        self.path = []
        self.wa_star = None  # Maximum marginal utility of agent a
        self.ja_star = None  # Task ID corresponding to the maximum marginal utility
        self.converged = False
        self.assigned_result = [[] for _ in range(self.agent_num)]  # Assignment results of all active agents, storing task IDs in visiting order

        # Phase 1: Initialize the agent's task sample set
        for j in range(self.task_num):
            self.Na.append(j)
        for j in self.Na:
            w_aj = self.score_scheme(j)
            self.Wa.append(w_aj)
        self.sort_Wa_Na()
        self.wa_star = self.Wa[0]
        self.ja_star = self.Na[0]

    def update_wa_ja(self):
        # Phase 2.1: Update the agent's wa*, ja*
        if len(self.Na) > 0 and len(self.Ta) < self.L_t:
            while self.wa_star == 0.0:
                wa1 = self.score_scheme(self.Na[0])
                if len(self.Na) == 1:  # No need to compare if only the last task remains
                    self.wa_star = wa1
                    self.ja_star = self.Na[0]
                elif wa1 >= self.Wa[1]:
                    self.wa_star = wa1
                    self.ja_star = self.Na[0]
                else:
                    # Resort Wa and Na
                    self.Wa = []
                    for j in self.Na:
                        w_aj = self.score_scheme(j)
                        self.Wa.append(w_aj)
                    self.sort_Wa_Na()

    def allocation_task(self, a_star, j_star_a_star):
        # Phase 2.3: Allocate the task or remove the current task j
        if a_star == self.id:
            self.Ta.append(self.ja_star)
            self.p.append(self.ja_star)
            idx = self.Na.index(self.ja_star)
            self.Na.pop(idx)
            self.Wa.pop(idx)
            self.wa_star = 0.0
            if len(self.Na) == 0:  # Last one
                self.converged = True
        else:
            if j_star_a_star in self.Na:
                idx = self.Na.index(j_star_a_star)
                self.Na.pop(idx)  # Na and Wa are one-to-one corresponding
                self.Wa.pop(idx)  # Na and Wa are one-to-one corresponding
                if self.Na:
                    if idx == 0:
                        self.ja_star = self.Na[0]
                        self.wa_star = self.Wa[0]
                        # self.wa_star = 0.0
                else:
                    self.converged = True

    def score_scheme(self, j):
        S_p = 0  # Total reward obtained by agent a while executing tasks along path p
        if len(self.p) > 0:
            distance_j = 0
            distance_j += self.dis_u2t[0][self.p[0]]
            S_p += (self.Lambda ** distance_j)
            for p_idx in range(len(self.p) - 1):
                distance_j += self.dis_t2t[self.p[p_idx]][self.p[p_idx + 1]]
                S_p += (self.Lambda ** distance_j)

        # Reward after adding the task
        p_temp = copy.deepcopy(self.p)
        p_temp.append(j)
        c_temp = 0
        distance_j = 0
        distance_j += self.dis_u2t[0][p_temp[0]]
        c_temp += (self.Lambda ** distance_j)
        if len(p_temp) > 1:
            for p_loc in range(len(p_temp) - 1):
                distance_j += self.dis_t2t[p_temp[p_loc]][p_temp[p_loc + 1]]
                c_temp += (self.Lambda ** distance_j)

        # Compute marginal utility
        w_aj = c_temp - S_p
        return w_aj

    def reward_score(self):
        if len(self.p) > 0:
            distance_j = 0
            distance_j += self.dis_u2t[0][self.p[0]]
            for p_idx in range(len(self.p) - 1):
                if p_idx >= 1:
                    dist = self.dis_t2t[self.p[p_idx]][self.p[p_idx + 1]]
                    distance_j += dist
                else:
                    distance_j += self.dis_t2t[self.p[p_idx]][self.p[p_idx + 1]]
            self.dist_cost = distance_j
            self.dist_end = self.dis_t2e[0][self.p[-1]]
        return self.dist_cost

    def sort_Wa_Na(self):
        """
        Function: sort in decreasing order
        """
        Wa = np.array(self.Wa)
        Na = np.array(self.Na)
        self.Wa.sort(reverse=True)  # Sort in descending order
        self.Na = list(Na[np.argsort(-Wa)])  # Sort Na in descending order according to Wa values

    def local_dist(self, ag_pos, taID):
        distance_j = 0.
        if len(taID) > 0:
            distance_j += l2norm(ag_pos, self.tasks_pos[taID[0]]) / self.scale
            for p_idx in range(len(taID) - 1):
                distance_j += self.dis_t2t1[taID[p_idx]][taID[p_idx + 1]]
        return distance_j

    def review_strategy(self, receive_info):
        a_star = receive_info[0][0]
        j_star_a_star = receive_info[0][1]
        a_par = None
        a_par_ta = None
        a_par_pos = None
        for i in range(1, len(self.receive_info)):
            cond1 = j_star_a_star == receive_info[i][1]
            # The premise for checking for intersection is that
            # the robot must be assigned to at least one task for it to be possible.
            cond2 = len(self.assigned_result[receive_info[i][0]]) > 0
            cond3 = (receive_info[0][2] - receive_info[i][2]) < 0.024  # 0.024-0.03
            if cond1 and cond2 and cond3:
                a_par = receive_info[i][0]
                a_par_ta = copy.deepcopy(self.assigned_result[a_par])
                a_par_pos = receive_info[i][3]
                break  # Only consider the second-highest bidding neighbor.
        if a_par is None:
            return a_star, j_star_a_star
        else:
            a_star_pos = receive_info[0][3]
            a_star_ta = copy.deepcopy(self.assigned_result[a_star])
            p1 = self.tasks_pos[a_star_ta[-1]] if len(a_star_ta) > 0 else a_star_pos
            p2 = self.tasks_pos[j_star_a_star]
            is_cross = False
            idx = 0
            for j in range(len(a_par_ta)):
                if j == 0:
                    p3 = a_par_pos
                    p4 = self.tasks_pos[a_par_ta[j]]
                else:
                    p3 = self.tasks_pos[a_par_ta[j - 1]]
                    p4 = self.tasks_pos[a_par_ta[j]]
                is_cross = seg_is_intersec(p1, p2, p3, p4)
                if is_cross:
                    idx = j
                    break
            if is_cross:
                a_star_seg1 = a_star_ta[0:]
                a_star_seg2 = [j_star_a_star]
                a_par_seg1 = a_par_ta[:idx]
                a_par_seg2 = a_par_ta[idx:]
                a_star_corssover = a_star_seg1 + a_par_seg2
                a_par_corssover = a_par_seg1 + a_star_seg2
                d1 = self.local_dist(a_star_pos, a_star_ta + [j_star_a_star]) + self.local_dist(a_par_pos, a_par_ta)
                d2 = self.local_dist(a_star_pos, a_star_corssover) + self.local_dist(a_par_pos, a_par_corssover)
                if d2 < d1:
                    self.assigned_result[a_star] = a_star_corssover
                    self.assigned_result[a_par] = a_par_seg1  # j_star_a_star will be appended later
                    if a_star == self.id:
                        self.p = copy.deepcopy(a_star_corssover)
                        self.Ta = copy.deepcopy(a_star_corssover)
                        self.wa_star = 0.0
                        self.update_wa_ja()
                    elif a_par == self.id:
                        self.p = copy.deepcopy(a_par_seg1)
                        self.Ta = copy.deepcopy(a_par_seg1)
                        self.wa_star = 0.0
                        self.update_wa_ja()
                    a_star = a_par
                    j_star_a_star = j_star_a_star

            return a_star, j_star_a_star

    def consensus(self):
        receive_info = sorted(self.receive_info, key=lambda x: x[2], reverse=True)
        a_star, j_star_a_star = self.review_strategy(receive_info)
        self.assigned_result[a_star].append(j_star_a_star)
        return a_star, j_star_a_star


def main():
    # Large blocks scenario
    envs = env.Env(envs_type[5])
    agents, task_area, tars_pos, scl = build_obj_pos_large(envs, ag_num=10, tar_num=100)
    name = '/large_u10t100.json'

    tar_ids, res, allocost, dists, all_paths = task_assign_scheme(agents, tars_pos, scl, LRCAPolicy, name)

    # print("Time used [sec]: ", allocost)
    # print("all_rewards: ", res)
    print("max travel distance: ", max(dists))
    # print("tar_id_list: ", tar_ids)

    # Plot
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
