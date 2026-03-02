"""
A_star 2D
@author: huiming zhou
"""

import math
import time
import heapq
import numpy as np
import matplotlib.pyplot as plt
from mamp.envs import env, env_visual
from mamp.configs.config import envs_type
from mamp.tools.utils import l2norm, smooth, path_length, normalize, unit_normal_vector


class AStar:
    """AStar set the cost + heuristics as the priority
    """
    def __init__(self, s_start, s_goal, heuristic_type, obs_grid, poly_obs, x_range, y_range, res=1.0, rob_radius=0.5):
        self.rob_radius = rob_radius
        self.s_start_f = s_start
        self.s_goal_f = s_goal
        self.s_start = (round(s_start[0] / res) * res, round(s_start[1] / res) * res)
        self.s_goal = (round(s_goal[0] / res) * res, round(s_goal[1] / res) * res)
        self.heuristic_type = heuristic_type
        self.x_range = x_range
        self.y_range = y_range
        self.res = res

        self.u_set = [(-1 * res, 0), (-1 * res, 1 * res), (0, 1 * res), (1 * res, 1 * res),
                      (1 * res, 0), (1 * res, -1 * res), (0, -1 * res), (-1 * res, -1 * res)]

        self.obstacles = obs_grid

        self.OPEN = []  # priority queue / OPEN set
        self.CLOSED = []  # CLOSED set / VISITED order
        self.PARENT = dict()  # recorded parent
        self.g = dict()  # cost to come
        self.path = []
        self.inflated_obs = []
        self.scale_polygon_vertices(poly_obs)
        self.length = 0.0
        self.check_time = 0.0
        self.searc_time = 0.0

    def scale_polygon_vertices(self, polygons):
        """Compute the vertices of the Minkowski sum of the polygon obstacle and the robot."""
        from mamp.agents.obstacle import Obstacle
        inflation = min(self.rob_radius / 5, 1.2)
        combinedRadius = self.rob_radius + 2 * inflation
        for i in range(len(polygons)):
            vertices_pos = []
            graph_pos = []
            for vertice in polygons[i].vertices_:
                p1 = normalize(vertice.previous_.point_ - vertice.point_)
                p2 = normalize(vertice.next_.point_ - vertice.point_)
                p1p2 = p2 - p1
                nLeft, nRight = unit_normal_vector(p1p2)
                n_scale = combinedRadius / math.sin(math.acos(p1.dot(p2)) / 2)
                vertices_pos.append(vertice.point_ + n_scale * nRight)  # The vertices of the obstacle are counter-clockwise, so the right side of the edge is the expanded side outward from the obstacle
                graph_pos.append(vertice.point_ + (n_scale + inflation) * nRight)
            self.inflated_obs.append(Obstacle(shape_dict={'shape': 'polygon'}, idx=polygons[i].id,
                                              tid='poly' + str(polygons[i].id), vertices=vertices_pos))

    def searching(self):
        """
        A_star Searching.
        :return: path, visited order
        """

        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        heapq.heappush(self.OPEN, (self.f_value(self.s_start), self.s_start))

        while self.OPEN:
            t1 = time.time()
            _, s = heapq.heappop(self.OPEN)
            self.CLOSED.append(s)
            t2 = time.time()
            self.searc_time += t2 - t1

            if s == self.s_goal:  # stop condition
                return self.extract_path(self.PARENT), self.CLOSED

            for s_n in self.get_neighbors(s):
                new_cost = self.g[s] + self.cost(s, s_n)

                if s_n not in self.g:
                    self.g[s_n] = math.inf

                if new_cost < self.g[s_n]:  # conditions for updating Cost
                    t1 = time.time()
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s
                    heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))
                    t2 = time.time()
                    self.searc_time += t2 - t1
                    if s_n == self.s_goal:           # goal found
                        break

        return [], []

    def get_neighbors(self, s):
        """
        Find neighbors of state s that are not in obstacles.
        :param s: state
        :return: neighbors
        """
        t1 = time.time()
        s_ns = [(s[0] + u[0], s[1] + u[1]) for u in self.u_set]
        t2 = time.time()
        self.searc_time += t2 - t1
        return s_ns

    def cost(self, s_start, s_goal):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicated!
        """

        t1 = time.time()
        if self.is_collision(s_start, s_goal):
            t2 = time.time()
            self.check_time += t2 - t1
            return math.inf
        t2 = time.time()
        self.check_time += t2 - t1
        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        """
        Check if the line segment (s_start, s_end) is in collision.
        :param s_start: start node
        :param s_end: end node
        :return: True: is collision / False: not collision
        """
        t1 = time.time()
        if s_start in self.obstacles or s_end in self.obstacles:
            t2 = time.time()
            self.check_time += t2 - t1
            return True

        t2 = time.time()
        self.check_time += t2 - t1
        return False

    def f_value(self, s):
        """
        f = g + h. (g: Cost to come, h: heuristic value)
        :param s: current state
        :return: f
        """

        return self.g[s] + self.heuristic(s)

    def extract_path(self, PARENT):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """

        path = [self.s_goal_f, self.s_goal]
        s = self.s_goal

        while True:
            s = PARENT[s]
            path.append(s)
            if s == self.s_start:
                break
        path.append(self.s_start_f)

        # self.path = path
        # self.length = path_length(path)
        self.path, self.length = smooth(path, self.inflated_obs)
        return self.path

    def heuristic(self, s):
        """
        Calculate heuristic.
        :param s: current node (state)
        :return: heuristic function value
        """

        heuristic_type = self.heuristic_type  # heuristic type
        goal = self.s_goal  # goal node

        if heuristic_type == "manhattan":
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:
            return math.hypot(goal[0] - s[0], goal[1] - s[1])

    def plot_path(self, path, visited):
        color = ['gainsboro', 'lightgray', 'silver', 'darkgray',
                 'bisque', 'navajowhite', 'moccasin', 'wheat',
                 'powderblue', 'skyblue', 'lightskyblue', 'cornflowerblue']

        obs_x = [x[0] for x in self.obstacles]
        obs_y = [x[1] for x in self.obstacles]
        plt.plot(obs_x, obs_y, "sk", zorder=3)

        px = [x[0] for x in path]
        py = [x[1] for x in path]
        plt.plot(px, py, linewidth=2, color='blue', zorder=3)
        plt.plot(self.s_start[0], self.s_start[1], "bs", zorder=3)
        plt.plot(self.s_goal[0], self.s_goal[1], "gs", zorder=3)

        # for x in visited:
        #     plt.plot(x[0], x[1], marker='s', color=color[0])
        plt.axis("equal")
        plt.show()


def main():
    import mamp.planner.random_positions as rand_pos

    # Scenario 8 small environment with 3 start and goal points
    # scenario_num = 8
    # radius = 0.1
    # res = 0.25
    # space = 0.5
    # agent_starts, agents_goal = rand_pos.small_env_positions()
    # agent_starts = agent_starts + [[1, 1], [7, 7], [7, 7], [7, 7]]
    # agents_goal = agents_goal + [[30, 30], [28, 20], [28, 26], [24, 29]]

    # Scenario 9 medium environment
    # scenario_num = 9
    # radius = 0.18
    # res = 0.5
    # space = 2.0
    # agent_starts, agents_goal = rand_pos.medium_env_positions()
    # agent_starts = agent_starts + [[50.0, 6.0], [205.0, 88.0], [148.0, 40.0], [232.0, 172.0]]
    # agents_goal = agents_goal + [[216, 200], [0, 208], [0, 208], [0, 208]]

    # Large-scale kilometer-level environment
    scenario_num = 5
    radius = 2.5
    res = 5
    space = 10.0
    agent_starts, agents_goal = rand_pos.large_env_positions()
    agent_starts = agent_starts + [[105.0, 160.0], [174.0, 3832.0], [496.0, 2819.0]]
    agents_goal = agents_goal + [[5955.0, 3865.0], [5816.0, 478.0], [5810.0, 478.0]]

    envs = env.Env(envs_type[scenario_num], is_grid_map=True)
    envs1 = env.Env(envs_type[scenario_num])
    envs2 = env_visual.Env(envs_type[scenario_num])
    x_range, y_range = (-space, envs.xDim + space), (-space, envs.yDim + space)
    obs_grid = envs.obs_grid
    poly_obs = envs2.poly_obs

    all_length = 0.
    all_time = 0.
    check_time = 0.0
    search_time = 0.0
    time_data = []
    dist_data = []
    ids = []
    nodes_num = []
    for i in range(len(agent_starts)):
        print('-----------------------------------idx------------------------------------', i)
        idx = 1000
        s_start, s_goal = (agent_starts[idx][0], agent_starts[idx][1]), (agents_goal[idx][0], agents_goal[idx][1])
        planner = AStar(s_start, s_goal, "euclidean", obs_grid, poly_obs, x_range, y_range, res=res, rob_radius=radius)
        t1 = time.time()
        path, visited = planner.searching()
        t2 = time.time()
        solve_time = t2 - t1
        all_length += planner.length
        all_time += solve_time
        check_time += planner.check_time
        search_time += planner.searc_time
        time_data.append(round(solve_time*1000, 2))
        dist_data.append(round(planner.length, 2))
        nodes_num.append(len(planner.CLOSED))
        print('solve time: ', solve_time)
        print('length: ', planner.length)
        path.reverse()
        print(len(path), path)
        if len(path) == 0:
            ids.append(i)
        info = "(Cost: " + str(round(solve_time, 5)) + " sec; " + str(round(planner.length, 2)) + " m)"
        if len(path) == 0:
            print(x_range, y_range)
        envs2.plot_env(planner, "Astar" + info)
    print('\n', '----------------------------------------data results---------------------------------------------')
    print(len(ids), ids)
    print('time_data: ', time_data)
    print('dist_data: ', dist_data, '\n')
    print('all_nodes_num:', sum(nodes_num), nodes_num)
    print('all solve time: ', all_time)
    print('all length: ', all_length)
    print('all check time: ', check_time)
    print('all search time: ', search_time)
    print('avg_nodes_num: ', sum(nodes_num) / len(agent_starts))
    print('average solve time: ', all_time / len(agent_starts))
    print('average length: ', all_length / len(agent_starts))
    print('average check time: ', check_time / len(agent_starts))
    print('average search time: ', search_time / len(agent_starts))


if __name__ == '__main__':
    main()
