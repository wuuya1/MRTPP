"""
RRT_star 2D
@author: huiming zhou
@optimized author: gang xu
"""

import math
import time
import numpy as np

from mamp.envs import env, env_visual
from mamp.configs.config import envs_type

from mamp.tools.utils import path_length, normalize, unit_normal_vector, is_intersect_polys


class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None
        self.cost = None


class RrtStar:
    def __init__(self, x_start, x_goal, poly_obs, x_range, y_range, step_len=1.0, goal_sample_rate=0.1,
                 search_radius=50.0, iter_max=10000, rob_radius=2.5):
        self.rob_radius = rob_radius
        self.s_start = x_start
        self.s_goal = x_goal
        self.s_start_f = x_start
        self.s_goal_f = x_goal
        self.s_start_node = Node(x_start)
        self.s_start_node.cost = 0.0
        self.s_goal_node = Node(x_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.iter_max = iter_max
        self.vertex = [self.s_start_node]
        self.vertex_coords = np.array([[x_start[0], x_start[1]]])  # Initial coordinate matrix
        self.path = []
        self.length = 0.0

        self.inflated_obs = []
        self.scale_polygon_vertices(poly_obs)

        self.x_range = x_range
        self.y_range = y_range

    def scale_polygon_vertices(self, polygons):
        """Compute the approximate Minkowski sum vertices of polygon obstacles and the robot."""
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
                vertices_pos.append(vertice.point_ + n_scale * nRight)  # The vertices of the obstacle are represented counter-clockwise, so the right side of the edge is the outward expansion of the obstacle
                graph_pos.append(vertice.point_ + (n_scale + inflation) * nRight)
            self.inflated_obs.append(Obstacle(shape_dict={'shape': 'polygon'}, idx=polygons[i].id,
                                              tid='poly' + str(polygons[i].id), vertices=vertices_pos))

    def planning(self):
        for k in range(self.iter_max):
            node_rand = self.generate_random_node(self.goal_sample_rate)
            node_near = self.nearest_neighbor(node_rand)
            node_new = self.new_state(node_near, node_rand)
            pos_near = [node_near.x, node_near.y]
            pos_new = [node_new.x, node_new.y]
            if node_new and not is_intersect_polys(pos_near, pos_new, self.inflated_obs):
                neighbor_index = self.find_near_neighbor(node_new)      # Index of nodes within the search radius and not intersecting with node_new
                self.vertex.append(node_new)
                self.vertex_coords = np.vstack([self.vertex_coords, [node_new.x, node_new.y]])

                if neighbor_index:
                    self.choose_parent(node_new, neighbor_index)
                    self.rewire(node_new, neighbor_index)
            index = self.search_goal_parent()
            if index is not None:
                path = self.extract_path(self.vertex[index])
                self.path, self.length = self.smooth(path)
                return self.path

        index = self.search_goal_parent()
        if index is None:
            index = len(self.vertex) - 1
        path = self.extract_path(self.vertex[index])
        self.path, self.length = self.smooth(path)
        return self.path

    def nearest_neighbor(self, target_node):
        target = np.array([target_node.x, target_node.y])
        dists = np.linalg.norm(self.vertex_coords - target, axis=1)
        return self.vertex[np.argmin(dists)]

    def new_state(self, node_start, node_goal):
        dist, theta = self.get_distance_and_angle(node_start, node_goal)

        dist = min(self.step_len, dist)
        node_new = Node((node_start.x + dist * math.cos(theta), node_start.y + dist * math.sin(theta)))

        node_new.parent = node_start

        return node_new

    def choose_parent(self, node_new, neighbor_index):
        costs = []
        for i in neighbor_index:
            if self.vertex[i].cost:
                cost = self.vertex[i].cost + math.hypot(node_new.x - self.vertex[i].x, node_new.y - self.vertex[i].y)
            else:
                cost = self.get_new_cost(self.vertex[i], node_new)
            costs.append(cost)

        cost_min_index = neighbor_index[int(np.argmin(costs))]
        node_new.parent = self.vertex[cost_min_index]

    def rewire(self, node_new, neighbor_index):
        for i in neighbor_index:
            node_neighbor = self.vertex[i]

            if self.cost(node_neighbor) > self.get_new_cost(node_new, node_neighbor):
                node_neighbor.parent = node_new

    def search_goal_parent(self):
        dist_list = [math.hypot(n.x - self.s_goal_node.x, n.y - self.s_goal_node.y) for n in self.vertex]
        node_index = [i for i in range(len(dist_list)) if dist_list[i] <= self.step_len]

        if len(node_index) > 0:
            cost_list = [dist_list[i] + self.cost(self.vertex[i]) for i in node_index
                         if not is_intersect_polys([self.vertex[i].x, self.vertex[i].y],
                                                   [self.s_goal_node.x, self.s_goal_node.y], self.inflated_obs)]
            if cost_list:
                return node_index[int(np.argmin(cost_list))]
            else:
                return None

        return None

    def get_new_cost(self, node_start, node_end):
        dist = math.hypot(node_end.x - node_start.x, node_end.y - node_start.y)

        return self.cost(node_start) + dist

    def generate_random_node(self, goal_sample_rate):
        delta = 0.5
        if np.random.random() > goal_sample_rate:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

        return self.s_goal_node

    def find_near_neighbor(self, node_new):
        n = len(self.vertex) + 1
        r = min(self.search_radius * math.sqrt((math.log(n) / n)), self.step_len)

        dist_table = [math.hypot(nd.x - node_new.x, nd.y - node_new.y) for nd in self.vertex]
        dist_table_index = [ind for ind in range(len(dist_table)) if dist_table[ind] <= r and
                            not is_intersect_polys([node_new.x, node_new.y],
                                                   [self.vertex[ind].x, self.vertex[ind].y], self.inflated_obs)]

        return dist_table_index

    def smooth(self, path):
        path_smooth = []
        route = path[:]
        path_smooth.append(path[0])
        while True:
            n = len(route)
            if n == 1:
                break
            for i in range(n - 1, 0, -1):
                if i == 1:
                    path_smooth.append((route[i][0], route[i][1]))
                    for j in range(i):
                        route.pop(0)
                    break
                else:
                    pos = np.array([route[0][0], route[0][1]])
                    pos_next = np.array([route[i][0], route[i][1]])
                    c1 = True
                    if i < n - 1:
                        p3 = np.array([route[i + 1][0], route[i + 1][1]])
                        c1 = np.dot(pos_next - pos, p3 - pos_next) > 0
                    if c1 and not is_intersect_polys(pos, pos_next, self.inflated_obs):
                        path_smooth.append((route[i][0], route[i][1]))
                        for j in range(i):
                            route.pop(0)
                        break
        return path_smooth, path_length(path_smooth)

    @staticmethod
    def cost(node_p):
        node = node_p
        cost = 0.0

        while node.parent:
            cost += math.hypot(node.x - node.parent.x, node.y - node.parent.y)
            node = node.parent

        return cost

    def extract_path(self, node_end):
        path = [[self.s_goal_node.x, self.s_goal_node.y]]
        node = node_end

        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)


def main():
    import mamp.planner.random_positions as rand_pos

    # Small scenario with 3 start and goal points
    scenario_num = 5
    radius = 2.5
    step_len = 200.0
    s_r = 500
    iter_max = 10000
    space = 10.0
    agent_starts, agents_goal = rand_pos.large_env_positions()
    agent_starts = agent_starts + [[105.0, 160.0], [174.0, 3832.0], [496.0, 2819.0]]
    agents_goal = agents_goal + [[5955.0, 3865.0], [5816.0, 478.0], [5810.0, 478.0]]

    envs = env.Env(envs_type[scenario_num])
    envs1 = env_visual.Env(envs_type[scenario_num])
    x_range, y_range = (-space, envs.xDim + space), (0, envs.yDim + space)
    poly_obs = envs.poly_obs

    all_length = 0.
    all_time = 0.
    check_time = 0.0
    search_time = 0.0
    time_data = []
    dist_data = []
    ids = []
    nodes_num = []
    for i in range(0, len(agent_starts)):
        print('-----------------------------------idx------------------------------------', i)
        idx = 1000
        s_start, s_goal = (agent_starts[idx][0], agent_starts[idx][1]), (agents_goal[idx][0], agents_goal[idx][1])
        rrt_star = RrtStar(s_start, s_goal, poly_obs, x_range, y_range, step_len=step_len, search_radius=s_r,
                           iter_max=iter_max, rob_radius=radius)
        t1 = time.time()
        path = rrt_star.planning()
        t2 = time.time()
        all_length += rrt_star.length
        solve_time = t2 - t1
        all_time += solve_time
        time_data.append(round(solve_time * 1000, 2))
        dist_data.append(round(rrt_star.length, 2))
        nodes_num.append(len(rrt_star.vertex))
        print('solve time: ', solve_time)
        print('length: ', rrt_star.length)
        path.reverse()
        print(len(path), path)
        if len(path) == 0:
            ids.append(i)
        info = "(Cost: " + str(round(solve_time, 5)) + " sec; " + str(round(rrt_star.length, 2)) + " m)"
        if len(path) == 0:
            print(x_range, y_range)
        envs1.plot_env(rrt_star, "RRT*" + info)
    print('----------------------------------------data results---------------------------------------------')
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
