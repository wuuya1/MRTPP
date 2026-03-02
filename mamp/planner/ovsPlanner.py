"""
@ Guidance Point Strategy path planning in 2D workspace
@ Author: Gang Xu
@ Date: 2026.01.12
@ Function: A fast path generation method
"""
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from mamp.envs import env, env_visual
from mamp.configs.config import envs_type
from draw.plt2d import draw_polygon_2d
from mamp.agents.obstacle import Obstacle, Vertex
from mamp.tools.utils import left_of, point_line_dist, normalize, pos_in_polygons, determin_between_line, path_length
from mamp.tools.utils import l2norm, unit_normal_vector, smooth, intersect_ploy_edges, is_intersect_polys


class OVSPlanner(object):
    def __init__(self):
        self.s_start = None  # Must be of type tuple
        self.s_goal = None  # Must be of type tuple
        self.s_start_f = None
        self.s_goal_f = None
        self.x_range = None
        self.y_range = None
        self.radius = -1
        self.turning_radius = -1
        self.epsilon = 1e-5
        self.inflation = 0.01
        self.path = []
        self.path_set = set()
        self.searched = dict()
        self.checked_obs = dict()
        self.global_guide_point = None
        self.length = 0.0
        self.guide_points = []
        self.path_verts = []
        self.poly_obs = None  # Obstacles represented by polygons and grid points
        self.inflated_obs = []  # Obstacles inflated considering robot size
        self.visible_graph = []  # Expanded points after obstacle inflation
        self.check_time = 0.0
        self.search_time = 0.0
        # self.init_time = 0.0

    def set_start_and_goal(self, s_start, s_goal):
        self.s_start = s_start
        self.s_goal = s_goal
        self.s_start_f = s_start
        self.s_goal_f = s_goal
        self.path = []
        self.path_set = set()
        self.path_verts = [Vertex(self.s_start)]
        self.searched = dict()
        self.checked_obs = dict()
        self.global_guide_point = None
        self.length = 0.0
        self.guide_points = []
        self.check_time = 0.0
        self.search_time = 0.0

    def init_env(self, obstacles, x_range, y_range, radius, inflation=0.01):
        self.x_range = x_range
        self.y_range = y_range
        self.poly_obs = obstacles
        self.inflation = inflation
        self.radius = radius

        agent_rad = self.radius + self.inflation
        combinedRadius = agent_rad + self.inflation
        for i in range(len(self.poly_obs)):
            self.scale_polygon_vertices(self.poly_obs[i], combinedRadius)
        for i in range(len(self.visible_graph)):
            vertices = self.visible_graph[i].vertices_
            for j in range(len(vertices)):
                if self.x_range[0] < vertices[j].point_[0] < self.x_range[1] \
                        and self.y_range[0] < vertices[j].point_[1] < self.y_range[1]:
                    continue
                else:
                    self.visible_graph[i].vertices_[j].is_disrupted = True
                    self.visible_graph[i].unreachable_vertices.append(self.visible_graph[i].vertices_[j])

    def scale_polygon_vertices(self, polygon, combinedRadius):
        """Compute the vertices of the Minkowski sum of the polygon obstacle and the robot."""
        vertices_pos = []
        graph_pos = []
        for vertice in polygon.vertices_:
            p1 = normalize(vertice.previous_.point_ - vertice.point_)
            p2 = normalize(vertice.next_.point_ - vertice.point_)
            p1p2 = p2 - p1
            nLeft, nRight = unit_normal_vector(p1p2)
            n_scale = combinedRadius / math.sin(math.acos(p1.dot(p2)) / 2)
            vertices_pos.append(vertice.point_ + n_scale * nRight)  # The vertices of the obstacle are counter-clockwise, so the right side of the edge is the expanded side outward from the obstacle
            graph_pos.append(vertice.point_ + (n_scale + self.inflation) * nRight)
        self.inflated_obs.append(Obstacle(shape_dict={'shape': 'polygon'}, idx=polygon.id,
                                          tid='poly' + str(polygon.id), vertices=vertices_pos))

        self.visible_graph.append(Obstacle(shape_dict={'shape': 'polygon'}, idx=polygon.id,
                                           tid='poly' + str(polygon.id), vertices=graph_pos))

    def global_search(self):
        pos = np.array(self.s_start)
        pos_next = np.array(self.s_goal)
        if pos_in_polygons(pos, self.inflated_obs) or pos_in_polygons(pos_next, self.inflated_obs):
            print("Solve failed, there may be in obstacle region. in begin")
            return []
        interseted_obs = self.check_all_intersect_obstacles(pos, pos_next, self.inflated_obs)
        path = [self.s_start]
        self.path_set.add(self.s_start)
        while True:
            # self.plot_path(path + [pos_next, self.s_goal])
            # self.plot_process(path + [pos_next, self.s_goal], interseted_obs)
            path_line, is_again_check = self.global_ovs_strategy(pos, pos_next, interseted_obs, path)
            if path_line is None:
                print("Solve Failed, there may not be a path to avoid collision. in while")
                return []
            if not is_again_check or not is_intersect_polys(path_line[0], path_line[1], self.inflated_obs):
                path_line.pop(0)  # Remove the duplicate starting node between the last element of path and the first element of path_line
                path += path_line
                self.path_set.add(path_line[-1])
                if path[-1] == self.s_goal:
                    # self.path = path
                    # self.length = path_length(path)
                    self.path, self.length = smooth(path, self.inflated_obs)
                    return self.path
                pos = np.array(path[-1])
                pos_next = np.array(self.s_goal)
            else:
                pos_next = np.array(path_line[-1])
            interseted_obs = self.check_all_intersect_obstacles(pos, pos_next, self.inflated_obs)

    def check_all_intersect_obstacles(self, pos, pos_next, sorted_obs):
        t1 = time.time()
        all_intersected_obs = []
        for obstacle in sorted_obs:
            if intersect_ploy_edges(pos, pos_next, obstacle.vertices_):
                all_intersected_obs.append(obstacle)
        t2 = time.time()
        self.check_time += t2 - t1
        return all_intersected_obs

    def cal_candidate_points(self, pos, pos_next, intersect_ob, path):
        """
        # Assume the gap between any polygon obstacles is large enough for the robot to pass through.
        """
        min_acceptable_dist = 0.5 * self.inflation
        vertices = self.visible_graph[intersect_ob.id].vertices_
        left_cands = []
        left_candsB = []
        right_cands = []
        right_candsB = []
        max_dist_left = 0.0
        max_dist_right = 0.0
        for vertice in vertices:
            if not vertice.convex_ or vertice.is_disrupted:  # Skip concave points and impassable points
                continue
            opt_p = vertice.point_
            condition1 = not intersect_ploy_edges(pos, opt_p, intersect_ob.vertices_)  # Line segment pos, opt_p does not intersect with the edges of the obstacle
            condition2 = l2norm(opt_p, pos) > min_acceptable_dist  # The guidance point's distance to the current position should not be less than the minimum distance to the polygon vertex
            condition3 = tuple(opt_p) not in self.path_set
            condition4 = np.dot(opt_p - pos, pos_next - opt_p) > 0
            v_in_left_side = left_of(pos, pos_next, opt_p)
            dist = point_line_dist(pos, pos_next, opt_p)
            if v_in_left_side > 0:
                if dist > max_dist_left:
                    max_dist_left = dist
                if condition1 and condition2 and condition3:
                    if condition4:
                        left_cands.append([opt_p, dist])
                    else:
                        left_candsB.append([opt_p, dist])
            elif v_in_left_side < 0:
                if dist > max_dist_right:
                    max_dist_right = dist
                if condition1 and condition2 and condition3:
                    if condition4:
                        right_cands.append([opt_p, dist])
                    else:
                        right_candsB.append([opt_p, dist])

        if len(left_cands) == 0 and len(right_cands) == 0:
            return left_candsB, right_candsB, max_dist_left, max_dist_right
        elif len(left_cands) > 0 and len(right_cands) == 0:
            return left_cands, right_candsB, max_dist_left, max_dist_right
        elif len(left_cands) == 0 and len(right_cands) > 0:
            return left_candsB, right_cands, max_dist_left, max_dist_right
        else:
            return left_cands, right_cands, max_dist_left, max_dist_right

    def cal_guidance_point(self, pos, pos_next, intersect_ob, path):
        left_cands, right_cands, max_d_left, max_d_right = self.cal_candidate_points(pos, pos_next, intersect_ob, path)
        if left_cands or right_cands:
            p_left = max(left_cands, key=lambda px: px[1]) if left_cands else []
            p_right = max(right_cands, key=lambda px: px[1]) if right_cands else []
            # At least one guidance point exists on either the left or the right
            if p_left and p_right:
                opt_ps = [p for p in [p_left, p_right] if
                          not intersect_ploy_edges(p[0], pos_next, intersect_ob.vertices_)]
                if opt_ps:
                    guide_point = min(opt_ps, key=lambda px: px[1])
                else:
                    p_left[1] = max_d_left
                    p_right[1] = max_d_right
                    guide_point = min([p_left, p_right], key=lambda px: px[1])
            elif p_left:
                guide_point = p_left
            else:
                guide_point = p_right
        else:
            guide_point = []
        return guide_point

    def global_ovs_strategy(self, pos, pos_next, interseted_obs, path):
        if interseted_obs:  # Obstacles blocking ahead
            vertex_points = []
            vertex_pointsB = []
            for intersect_ob in interseted_obs:
                vertex_point = self.cal_guidance_point(pos, pos_next, intersect_ob, path)
                if len(vertex_point) > 0:
                    if not is_intersect_polys(pos, vertex_point[0], self.inflated_obs):
                        vertex_points.append(vertex_point)
                    else:
                        vertex_pointsB.append(vertex_point)
            if vertex_points:
                gp = max(vertex_points, key=lambda px: px[1])[0]
                is_again_check = False
                return [(pos[0], pos[1]), (gp[0], gp[1])], is_again_check
            else:
                if vertex_pointsB:
                    gp = max(vertex_pointsB, key=lambda px: px[1])[0]
                    is_again_check = True
                    return [(pos[0], pos[1]), (gp[0], gp[1])], is_again_check
                else:
                    return None, None
        else:
            is_again_check = False
            return [(pos[0], pos[1]), (pos_next[0], pos_next[1])], is_again_check

    def plot_path(self, path, name="OVS"):
        fig = plt.figure(0)
        fig_size = (10 * 1.2, 7 * 1.2)
        fig.set_size_inches(fig_size[0], fig_size[1])
        ax = fig.add_subplot(1, 1, 1)
        colors = rcParams['axes.prop_cycle'].by_key()['color']  # Get default color cycle

        for i in range(len(self.poly_obs)):
            color = colors[i % len(colors)]
            draw_polygon_2d(ax, self.poly_obs[i].vertices_pos, fc_color=color, ec_color=color, alpha=1)
            draw_polygon_2d(ax, self.inflated_obs[i].vertices_pos, fc_color='none', ec_color=color, alpha=0.9)

        px = [x[0] for x in path]
        py = [x[1] for x in path]
        color = colors[1 % len(colors)]
        # for i in range(len(path)):
        #     ax.add_patch(plt.Circle((px[i], py[i]), radius=self.radius, fc=color, ec=color, linewidth=1))

        # spx = [x[0] for x in self.searched]
        # spy = [x[1] for x in self.searched]
        # ax.scatter(spx[:], spy[:], marker='*', color='green')
        plt.plot(px, py, linewidth=2, color='blue', zorder=3)
        plt.plot(self.s_start[0], self.s_start[1], marker='o', markersize=8, color='b', zorder=4)
        plt.plot(self.s_goal[0], self.s_goal[1], marker='*', markersize=8, color='r', zorder=4)
        ax.scatter(px[:], py[:], marker='o', color='red')

        plt.title(name)
        plt.axis("equal")
        plt.show()

    def plot_process(self, path, intersect_obs, name="OVS", use_dash=True):
        fig = plt.figure(0)
        fig_size = (12 * 1.2, 9 * 1.2)
        fig.set_size_inches(fig_size[0], fig_size[1])
        ax = fig.add_subplot(1, 1, 1)
        colors = rcParams['axes.prop_cycle'].by_key()['color']  # Get default color cycle
        for i in range(len(self.poly_obs)):
            draw_polygon_2d(ax, self.poly_obs[i].vertices_pos)
            draw_polygon_2d(ax, self.inflated_obs[i].vertices_pos, fc_color='none', alpha=0.8)

        px = [x[0] for x in path]
        py = [x[1] for x in path]
        color_rad = colors[1 % len(colors)]
        for i in range(len(path)):
            ax.add_patch(plt.Circle((px[i], py[i]), radius=self.radius, fc=color_rad, ec=color_rad, linewidth=1))

        # spx = [x[0] for x in self.searched]
        # spy = [x[1] for x in self.searched]
        # ax.scatter(spx[:], spy[:], marker='*', color='green')
        if intersect_obs:
            for i in range(len(intersect_obs)):
                idx = intersect_obs[i].id
                draw_polygon_2d(ax, self.poly_obs[idx].vertices_pos, fc_color=color_rad, ec_color=color_rad, alpha=0.9)
                draw_polygon_2d(ax, self.inflated_obs[idx].vertices_pos, fc_color='none', ec_color=color_rad, alpha=0.8)
        if use_dash:
            plt.plot(px, py, linewidth=1, linestyle=":", color='blue', zorder=3)
        else:
            plt.plot(px, py, linewidth=2, color='blue', zorder=3)
        plt.plot(self.s_start[0], self.s_start[1], marker='o', markersize=15, color='b', zorder=4)
        plt.plot(self.s_goal[0], self.s_goal[1], marker='*', markersize=15, color='r', zorder=4)
        for i in range(len(intersect_obs)):
            verts_x, verts_y = [], []
            for point in intersect_obs[i].vertices_:
                if point.convex_ and not point.is_disrupted:
                    verts_x.append(point.point_[0])
                    verts_y.append(point.point_[1])
            ax.scatter(verts_x, verts_y, marker='o', color=color_rad)

        plt.title(name)
        plt.axis("equal")
        plt.show()


def main():
    import mamp.planner.random_positions as rand_pos

    # Small scenario with 3 start and goal points
    scenario_num = 5
    radius = 2.5
    inflation = min(radius / 5, 1.2)
    agent_starts, agents_goal = rand_pos.large_env_positions()
    agent_starts = agent_starts + [[105.0, 160.0], [174.0, 3832.0], [496.0, 2819.0]]
    agents_goal = agents_goal + [[5955.0, 3865.0], [5816.0, 478.0], [5810.0, 478.0]]

    envs = env.Env(envs_type[scenario_num])
    envs1 = env_visual.Env(envs_type[scenario_num])
    space = 10.0
    x_range, y_range = (-space, envs.xDim + space), (-space, envs.yDim + space)
    poly_obs = envs.poly_obs

    all_length = 0.
    all_time = 0.
    check_time = 0.
    search_time = 0.
    time_data = []
    dist_data = []
    ids = []
    pos_num = len(agent_starts)
    ovs = OVSPlanner()
    ovs.init_env(poly_obs, x_range, y_range, radius, inflation=inflation)
    for i in range(0, pos_num):
        print('-----------------------------------idx------------------------------------', i)
        idx = 1000
        s_start, s_goal = (agent_starts[idx][0], agent_starts[idx][1]), (agents_goal[idx][0], agents_goal[idx][1])
        ovs.set_start_and_goal(s_start, s_goal)
        t1 = time.time()
        path = ovs.global_search()
        if len(path) == 0:
            ovs.set_start_and_goal(s_goal, s_start)
            path = ovs.global_search()
            path.reverse()
            ovs.s_start, ovs.s_goal = s_start, s_goal
        t2 = time.time()
        solve_time = t2 - t1
        all_length += ovs.length
        all_time += solve_time
        check_time += ovs.check_time
        search_time += ovs.search_time
        time_data.append(round(solve_time*1000, 2))
        dist_data.append(round(ovs.length, 2))
        print('solve time: ', solve_time)
        print('length: ', ovs.length)
        print(len(path), path)
        if len(path) == 0:
            ids.append(i)
        info = "(Cost: " + str(round(solve_time, 5)) + " sec; " + str(round(ovs.length, 2)) + " m)"
        if len(path) == 0:
            print(x_range, y_range)
        envs1.plot_env(ovs, "OVSPlanner" + info)
    print('\n', '----------------------------------------data results---------------------------------------------')
    print(len(ids), ids)
    print('time_data: ', time_data)
    print('dist_data: ', dist_data, '\n')
    print('all solve time: ', all_time)
    print('all length: ', all_length)
    print('all check time: ', check_time)
    print('all search time: ', search_time)
    print('average solve time: ', all_time / len(agent_starts))
    print('average length: ', all_length / len(agent_starts))
    print('average check time: ', check_time / len(agent_starts))
    print('average search time: ', search_time / len(agent_starts))


if __name__ == '__main__':
    main()
