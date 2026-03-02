"""
Env 2D
@author: Gang Xu
"""
import os
import json
import math
import random
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon, Point
from draw.plt2d import draw_polygon_2d
# from mamp.agents.obstacle import Obstacle
from mamp.agents.poly_obstacle import PolyObstacle, Vertex
from mamp.tools.utils import normalize, unit_normal_vector, enclosing_circle, circle_polygon_intersect, path_length


# from scipy.spatial import ConvexHull, minimum_enclosing_circle


class Env:
    def __init__(self, env_type, is_grid_map=False):
        self.xDim = 501  # size of background
        self.yDim = 501
        self.env_type = env_type
        self.is_grid_map = is_grid_map
        self.motions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                        (1, 0), (1, -1), (0, -1), (-1, -1)]
        self.poly_obs, self.obs_grid = self.read_obs_map()

    def read_obs_map(self):
        if self.env_type == "small":
            if self.is_grid_map:
                return [], self.obs_test_grid()
            else:
                return self.small_map(), []
        elif self.env_type == "large_maze":
            if self.is_grid_map:
                return [], self.poly_maze_map_grid()
            else:
                return self.poly_maze_map(), []
        elif self.env_type == "large":
            if self.is_grid_map:
                return [], self.obs_map_grid()
            else:
                return self.poly_obs_map(), []
        elif self.env_type == "large_corridor":
            if self.is_grid_map:
                return [], self.cooridor_grid()
            else:
                return self.corridor_map(), []
        elif self.env_type == "small_blocks":
            if self.is_grid_map:
                return [], self.small_cooridor_grid()
            else:
                return self.small_corridor_map(), []
        elif self.env_type == "symmetric":
            if self.is_grid_map:
                return [], self.symmetric_grid()
            else:
                return self.symmetrical_map(), []
        elif self.env_type == "large_obs":
            if self.is_grid_map:
                return [], self.large_obs_grid()
            else:
                return self.large_obs_map(), []
        elif self.env_type == "small_indoor":
            if self.is_grid_map:
                return [], self.small_indoor_grid()
            else:
                return self.small_indoor_map(), []
        elif self.env_type == "128":
            if self.is_grid_map:
                return [], self.small_128_grid()
            else:
                return self.small_128_map(), []
        elif self.env_type == "256":
            if self.is_grid_map:
                return [], self.small_256_grid()
            else:
                return self.small_256_map(), []
        else:
            if self.is_grid_map:
                return [], self.obs_test_grid()
            else:
                return [], self.obs_test_grid()

    def obs_test_grid(self):
        """
        小范围经典的测试场景，栅格绘制, 51*31
        """
        self.xDim, self.yDim = 50., 30.
        x, y = int(self.xDim + 1), int(self.yDim + 1)
        obs = set()

        for i in range(x):
            obs.add((i, 0))
        for i in range(x):
            obs.add((i, y - 1))

        for i in range(y):
            obs.add((0, i))
        for i in range(y):
            obs.add((x - 1, i))

        for i in range(10, 21):
            obs.add((i, 15))
        for i in range(15):
            obs.add((20, i))

        for i in range(15, 30):
            obs.add((30, i))
        for i in range(16):
            obs.add((40, i))

        return obs

    def poly_maze_map_grid(self):
        """
        大范围的二维迷宫场景，栅格绘制, 500*500
        """
        self.xDim, self.yDim = 500., 500.
        x, y = int(self.xDim + 1), int(self.yDim + 1)
        poly_maze = self.poly_maze_map()  # 大范围的二维迷宫场景，多边形绘制, 501m*501m
        obstacles = set()

        # boundary of environment
        for i in range(x):
            obstacles.add((i, 0))
            obstacles.add((i, y - 1))
        for i in range(y):
            obstacles.add((0, i))
            obstacles.add((x - 1, i))

        # obstacles
        for i in range(4, len(poly_maze)):
            for j in range(len(poly_maze[i].vertices_pos)):
                point = poly_maze[i].vertices_[j].point_
                next_point = poly_maze[i].vertices_[j].next_.point_
                if point[0] == next_point[0]:
                    min_y = min(point[1], next_point[1])
                    max_y = max(point[1], next_point[1])
                    for k in range(min_y, max_y + 1):
                        p = (point[0], k)
                        if p not in obstacles:
                            obstacles.add(p)
                elif point[1] == next_point[1]:
                    min_x = min(point[0], next_point[0])
                    max_x = max(point[0], next_point[0])
                    for k in range(min_x, max_x + 1):
                        p = (k, point[1])
                        if p not in obstacles:
                            obstacles.add(p)
        return obstacles

    def symmetric_grid(self):
        """
        大范围的对称二维场景，栅格绘制，801*201
        Initialize obstacles' positions
        :return: map of obstacles
        """
        self.xDim, self.yDim = 800., 200.
        return self.read_json(name='/symetric_env.json')

    def obs_map_grid(self):
        """
        大范围的二维场景，栅格绘制，681*501
        Initialize obstacles' positions
        :return: map of obstacles
        """
        obstacles = set()
        poly_gons = self.poly_obs_map()

        # boundary of environment
        for i in range(self.xDim):
            obstacles.add((i, 0))
            obstacles.add((i, self.yDim - 1))
        for i in range(self.yDim):
            obstacles.add((0, i))
            obstacles.add((self.xDim - 1, i))

        # obstacles
        for i in range(4, len(poly_gons)):
            for j in range(len(poly_gons[i].vertices_pos)):
                point = poly_gons[i].vertices_[j].point_
                next_point = poly_gons[i].vertices_[j].next_.point_
                if point[0] == next_point[0]:
                    min_y = min(point[1], next_point[1])
                    max_y = max(point[1], next_point[1])
                    for k in range(min_y, max_y + 1):
                        p = (point[0], k)
                        if p not in obstacles:
                            obstacles.add(p)
                elif point[1] == next_point[1]:
                    min_x = min(point[0], next_point[0])
                    max_x = max(point[0], next_point[0])
                    for k in range(min_x, max_x + 1):
                        p = (k, point[1])
                        if p not in obstacles:
                            obstacles.add(p)
        return obstacles

    def cooridor_grid(self):
        """
        Initialize obstacles' positions
        :return: map of obstacles
        """
        # self.xDim, self.yDim = 800., 200.
        #
        # obstacles = set()
        # poly_gons = self.corridor_map()
        # blocks = []
        # resolution = 1
        # combined_radius = 2.5 + 1.0
        # for x in range(0, int(self.xDim+1), resolution):
        #     for y in range(0, int(self.yDim+1), resolution):
        #         for ob in poly_gons:
        #             if circle_polygon_intersect((x, y), combined_radius, ob.vertices_pos):
        #                 obstacles.add((x, y))
        #                 blocks.append([x, y])
        #     print(x)
        # self.write_env_cfg(blocks, name='/corridor_env.json')

        obstacles = self.read_json(name='/corridor_env.json')
        return obstacles

    def small_cooridor_grid(self):
        """
        Initialize obstacles' positions
        :return: map of obstacles
        """
        # obstacles = set()
        # radius = 2.5
        # combined_radius = radius + 2 * min(radius / 5, 1.2)
        # polygons = self.small_corridor_map()
        # res = 0.25
        # xDim, yDim = int((self.xDim/res)+2), int((self.yDim/res)+2)
        # blocks = []
        # for x in range(0, xDim, 1):
        #     print(x)
        #     for y in range(0, yDim, 1):
        #         for ob in polygons:
        #             point = (x*res, y*res)
        #             if circle_polygon_intersect(point, combined_radius, ob.vertices_pos):
        #                 obstacles.add(point)
        #                 blocks.append(point)
        # self.write_env_cfg(blocks, res, name='/small_blocks_grid.json')

        obstacles = self.read_json(name='/small_blocks_grid.json')
        return obstacles

    def large_obs_grid(self):
        """
        大范围的对称二维场景，栅格绘制，6001*4001
        Initialize obstacles' positions
        :return: map of obstacles
        """
        # obstacles = set()
        # radius = 2.5
        # combined_radius = radius + 2 * min(radius / 5, 1.2)
        # polygons = self.large_obs_map()
        # res = 2
        # xDim, yDim = int((self.xDim / res) + 2), int((self.yDim / res) + 2)
        # blocks = []
        # for x in range(0, xDim, 1):
        #     print(x)
        #     for y in range(0, yDim, 1):
        #         for ob in polygons:
        #             point = (x * res, y * res)
        #             if circle_polygon_intersect(point, combined_radius, ob.vertices_pos):
        #                 obstacles.add(point)
        #                 blocks.append(point)
        # self.write_env_cfg(blocks, res, name='/large_obs_grid.json')

        obstacles = self.read_json(name='/large_obs_grid.json')
        return obstacles

    def small_128_grid(self):
        """
        :return: map of obstacles
        # """
        # obstacles = set()
        # combined_radius = combined_radius = 0.18 + 0.18/5 + 0.18/5
        # polygons = self.small_128_map()
        # res = 0.25
        # xDim, yDim = int((self.xDim/res)+2), int((self.yDim/res)+2)
        # blocks = []
        # for x in range(0, xDim, 1):
        #     print(x)
        #     for y in range(0, yDim, 1):
        #         for ob in polygons:
        #             point = (x*res, y*res)
        #             if circle_polygon_intersect(point, combined_radius, ob.vertices_pos):
        #                 obstacles.add(point)
        #                 blocks.append(point)
        # self.write_env_cfg(blocks, res, name='/128.json')

        obstacles = self.read_json(name='/128.json')
        return obstacles

    def small_256_grid(self):
        """
        :return: map of obstacles
        """
        scale = 16.0
        self.xDim, self.yDim = 16.0 * scale, 16.0 * scale

        # # 根据像素点提取栅格
        obstacles = set()
        blocks = []
        res = 0.25
        # save_dir = os.path.dirname(os.path.realpath(__file__)) + '/map/'
        # csv_path = save_dir + 'pixel_values.csv'  # 替换为CSV文件的实际路径
        # pixel_array = np.loadtxt(csv_path, delimiter=',')
        #
        # # 遍历二维数组，找到灰度值为0的像素点
        # for y in range(pixel_array.shape[0] - 1, -1, -4):
        #     for x in range(0, pixel_array.shape[1], 4):
        #         if pixel_array[y, x] == 0:
        #             # 计算坐标
        #             coord_x = x * 0.0625
        #             coord_y = (255 - y) * 0.0625
        #             point = (coord_x * scale, coord_y * scale)
        #             obstacles.add(point)
        #             blocks.append(point)
        #
        # self.write_env_cfg(blocks, 4.0, name='/256_256.json')

        obstacles1 = self.read_json(name='/256_256.json')  # 256_256.json栅格大小为4*4
        res = 0.25  # 分辨率为0.25m(256m*256m情况下)
        for point in obstacles1:
            coord_x, coord_y = point
            for i in range(int(4 / res)):
                for j in range(int(4 / res)):
                    p = (coord_x + i * res, coord_y + j * res)
                    obstacles.add(p)
                    blocks.append(p)
        self.write_env_cfg(blocks, res, name='/256.json')
        return obstacles

    def small_256_map(self):
        scale = 16.0
        self.xDim, self.yDim = 16.0 * scale, 16.0 * scale
        obstacles = []
        save_dir = os.path.dirname(os.path.realpath(__file__)) + '/map/'
        csv_path = save_dir + 'pixel_values.csv'  # 替换为CSV文件的实际路径
        pixel_array = np.loadtxt(csv_path, delimiter=',')

        # 初始化一个空列表，用于存储灰度值为0的像素点
        ob_pos = []

        # 遍历二维数组，找到灰度值为0的像素点
        points = []
        for y in range(pixel_array.shape[0] - 1, -1, -4):
            for x in range(0, pixel_array.shape[1], 4):
                if pixel_array[y, x] == 0:
                    # 计算坐标
                    coord_x = x * 0.0625
                    coord_y = (255 - y) * 0.0625
                    # 将坐标和形状信息添加到列表中
                    ob_pos.append([coord_x, coord_y, {'shape': 'rect', 'rect': (0.25, 0.25)}])
                    points.append([coord_x, coord_y])

        # print(points)
        # print(len(ob_pos))
        for i in range(len(ob_pos)):
            vertices = gen_polygon_vertices(ob_pos[i][:2], ob_pos[i][2])
            vertices = np.array(vertices) * scale
            polygon = Polygon(vertices)
            pos = (polygon.centroid.x, polygon.centroid.y)
            min_rect = polygon.minimum_rotated_rectangle
            width, height = min_rect.bounds[2] - min_rect.bounds[0], min_rect.bounds[3] - min_rect.bounds[1]
            radius = np.sqrt((width / 2) ** 2 + (height / 2) ** 2)
            obstacles.append(PolyObstacle(shape_dict={'shape': 'polygon', 'feature': radius}, idx=i,
                                          tid='obstacle' + str(i), vertices=vertices, pos=pos, ))

        return obstacles

    def small_indoor_grid(self):
        """
        :return: map of obstacles
        """
        obstacles = set()
        combined_radius = 0.18 + 0.18 / 5 + 0.18 / 5
        polygons = self.small_indoor_map()
        res = 0.1
        xDim, yDim = int((self.xDim / res) + 2), int((self.yDim / res) + 2)
        blocks = []
        for x in range(-15, xDim, 1):
            print(x)
            for y in range(0, yDim, 1):
                for ob in polygons:
                    point = (x * res, y * res)
                    if circle_polygon_intersect(point, combined_radius, ob.vertices_pos):
                        obstacles.add(point)
                        blocks.append(point)
        self.write_env_cfg(blocks, res, name='/small_indoor_grid.json')

        # obstacles = self.read_json(name='/small_indoor_grid.json')
        return obstacles

    def rect_vertices(self, min_x, min_y, max_x, max_y):
        max_x = min(max_x, self.xDim)
        max_y = min(max_y, self.yDim)
        rect_vertices = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
        print(rect_vertices)
        return rect_vertices

    def small_map(self):
        """
        小范围的二维场景，多边形绘制，50m*30m
        Initialize obstacles' positions
        :return: map of obstacles
        """
        self.xDim, self.yDim = 50., 30.
        x, y = self.xDim, self.yDim
        poly_obs = []
        obs_vertices = [self.rect_vertices(0, 0, x, 1), self.rect_vertices(x - 1, 1, x, y - 1),
                        self.rect_vertices(0, y - 1, x, y), self.rect_vertices(0, 1, 1, y - 1),
                        [[20, 0], [21, 0], [21, 15], [10, 15], [10, 14], [20, 14]],
                        [[40, 0], [41, 0], [41, 15], [40, 15]],
                        [[29, 15], [30, 15], [30, 30], [29, 30]]
                        ]
        for i, vertices in enumerate(obs_vertices):
            vertices = np.array(vertices)
            pos, radius = enclosing_circle(vertices)
            poly_obs.append(PolyObstacle(pos=pos, shape_dict={'shape': 'polygon', 'feature': radius},
                                         idx=i, tid='obstacle' + str(i), vertices=vertices))

        return poly_obs

    def large_obs_map(self):
        self.xDim, self.yDim = 6000., 4000.
        obstacles = []
        ob_num = 50
        obs_vertices = [
            [[4895.25, 1171.77], [4985.25, 1171.77], [4985.25, 1396.77], [5255.25, 1396.77], [5255.25, 1726.77],
             [4655.25, 1726.77], [4655.25, 1396.77], [4895.25, 1396.77]],
            [[295.81, 1496.17], [558.31, 1496.17], [558.31, 1578.67], [730.81, 1578.67], [730.81, 1841.17],
             [295.81, 1841.17]], [[759.24, 286.46], [1531.74, 286.46], [1531.74, 541.46], [759.24, 541.46]],
            [[1391.1, 2716.44], [2283.6, 2716.44], [2283.6, 2843.94], [1391.1, 2843.94]],
            [[2905.97, 3213.65], [3145.97, 3213.65], [3145.97, 3423.65], [2905.97, 3423.65]],
            [[326.38, 2362.35], [611.38, 2362.35], [611.38, 2647.35], [888.88, 2647.35], [888.88, 2909.85],
             [693.88, 2909.85], [693.88, 2722.35], [326.38, 2722.35]],
            [[1982.53, 1133.66], [2200.0299999999997, 1133.66], [2200.0299999999997, 1351.16], [1982.53, 1351.16]],
            [[4187.46, 3137.14], [4277.46, 3137.14], [4277.46, 3362.14], [4547.46, 3362.14], [4547.46, 3692.14],
             [3947.46, 3692.14], [3947.46, 3362.14], [4187.46, 3362.14]],
            [[3534.35, 1595.95], [3811.85, 1595.95], [3811.85, 1768.45], [3534.35, 1768.45]],
            [[4789.94, 1922.16], [5682.44, 1922.16], [5682.44, 2049.66], [4789.94, 2049.66]],
            [[3973.22, 1026.39], [4213.219999999999, 1026.39], [4213.219999999999, 1236.39], [3973.22, 1236.39]],
            [[972.65, 3391.37], [1212.65, 3391.37], [1212.65, 3601.37], [972.65, 3601.37]],
            [[4208.2, 2382.0], [4418.2, 2382.0], [4418.2, 2592.0], [4208.2, 2592.0]],
            [[3649.65, 2378.14], [3859.65, 2378.14], [3859.65, 2610.64], [3649.65, 2610.64]],
            [[4791.7, 2849.33], [5009.2, 2849.33], [5009.2, 3066.83], [4791.7, 3066.83]],
            [[5386.39, 284.32], [5671.39, 284.32], [5671.39, 569.3199999999999], [5948.89, 569.3199999999999],
             [5948.89, 831.8199999999999], [5753.89, 831.8199999999999], [5753.89, 644.3199999999999],
             [5386.39, 644.3199999999999]],
            [[3606.89, 569.86], [3824.39, 569.86], [3824.39, 787.36], [3606.89, 787.36]],
            [[2952.06, 309.33], [3042.06, 309.33], [3042.06, 534.3299999999999], [3312.06, 534.3299999999999],
             [3312.06, 864.3299999999999], [2712.06, 864.3299999999999], [2712.06, 534.3299999999999],
             [2952.06, 534.3299999999999]],
            [[4725.96, 395.35], [4920.96, 395.35], [4920.96, 762.85], [4725.96, 762.85]],
            [[2973.22, 2044.81], [3250.72, 2044.81], [3250.72, 2217.31], [2973.22, 2217.31]],
            [[2134.73, 1896.73], [2397.23, 1896.73], [2397.23, 1979.23], [2569.73, 1979.23], [2569.73, 2241.73],
             [2134.73, 2241.73]], [[2212.78, 3700.86], [2490.28, 3700.86], [2490.28, 3873.36], [2212.78, 3873.36]],
            [[5140.89, 3367.94], [5403.39, 3367.94], [5403.39, 3450.44], [5575.89, 3450.44], [5575.89, 3712.94],
             [5140.89, 3712.94]], [[1585.34, 815.21], [1780.34, 815.21], [1780.34, 1182.71], [1585.34, 1182.71]],
            [[2980.4, 1354.81], [3227.9, 1354.81], [3227.9, 1677.31], [2980.4, 1677.31]],
            [[2020.47, 233.6], [2267.9700000000003, 233.6], [2267.9700000000003, 556.1], [2020.47, 556.1]],
            [[177.27, 767.54], [372.27, 767.54], [372.27, 1135.04], [177.27, 1135.04]],
            [[4140.85, 1402.35], [4358.35, 1402.35], [4358.35, 1619.85], [4140.85, 1619.85]],
            [[1684.39, 3121.59], [1924.39, 3121.59], [1924.39, 3331.59], [1684.39, 3331.59]],
            [[1105.73, 755.16], [1323.23, 755.16], [1323.23, 972.66], [1105.73, 972.66]],
            [[3382.35, 3363.9], [3727.35, 3363.9], [3727.35, 3491.4], [3382.35, 3491.4]],
            [[1603.78, 1522.32], [1836.28, 1522.32], [1836.28, 1972.32], [1603.78, 1972.32]],
            [[347.8, 3104.73], [557.8, 3104.73], [557.8, 3314.73], [347.8, 3314.73]],
            [[5716.0, 1039.53], [5933.5, 1039.53], [5933.5, 1227.03], [5716.0, 1227.03]],
            [[3272.6, 3729.17], [3617.6, 3729.17], [3617.6, 3856.67], [3272.6, 3856.67]],
            [[5009.99, 2367.94], [5902.49, 2367.94], [5902.49, 2495.44], [5009.99, 2495.44]],
            [[1421.58, 2135.8], [1631.58, 2135.8], [1631.58, 2368.3], [1421.58, 2368.3]],
            [[2634.76, 2844.28], [3527.26, 2844.28], [3527.26, 2971.78], [2634.76, 2971.78]],
            [[5673.76, 2860.88], [5891.26, 2860.88], [5891.26, 3048.38], [5673.76, 3048.38]],
            [[2725.95, 3774.25], [2943.45, 3774.25], [2943.45, 3991.75], [2725.95, 3991.75]],
            [[309.29, 214.09], [526.79, 214.09], [526.79, 431.59000000000003], [309.29, 431.59000000000003]],
            [[823.77, 1236.7], [1168.77, 1236.7], [1168.77, 1364.2], [823.77, 1364.2]],
            [[334.06, 3494.08], [596.56, 3494.08], [596.56, 3576.58], [769.06, 3576.58], [769.06, 3839.08],
             [334.06, 3839.08]], [[3477.57, 1208.79], [3695.07, 1208.79], [3695.07, 1426.29], [3477.57, 1426.29]],
            [[2591.71, 1035.89], [2801.71, 1035.89], [2801.71, 1268.39], [2591.71, 1268.39]],
            [[4031.19, 659.31], [4308.6900000000005, 659.31], [4308.6900000000005, 831.81], [4031.19, 831.81]],
            [[1161.73, 1550.28], [1356.73, 1550.28], [1356.73, 1917.78], [1161.73, 1917.78]],
            [[5608.8, 1509.17], [5886.3, 1509.17], [5886.3, 1681.67], [5608.8, 1681.67]],
            [[971.8, 2242.23], [1189.3, 2242.23], [1189.3, 2459.73], [971.8, 2459.73]],
            [[1797.07, 3590.51], [2014.57, 3590.51], [2014.57, 3808.01], [1797.07, 3808.01]]]
        for i, vertices in enumerate(obs_vertices):
            vertices = np.array(vertices)
            # print(vertices.tolist())
            pos, radius = enclosing_circle(vertices)
            obstacles.append(PolyObstacle(pos=pos, shape_dict={'shape': 'polygon', 'feature': radius},
                                          idx=i, tid='obstacle' + str(i), vertices=vertices))
        return obstacles

    def small_indoor_map1(self):
        self.xDim, self.yDim = 6., 5.
        obstacles = []
        ob_pos = [
            [0.3, 0.98, {'shape': 'rect', 'rect': (0.6, 0.5)}],
            [0.0, 0.9, {'shape': 'rect', 'rect': (0.3, 1.0)}],
            [1.8, 0, {'shape': 'rect', 'rect': (0.1, 0.69)}],
            [2.1, 3.4, {'shape': 'concave', 'concave': (1.2, 0.9, 0.1)}],
            [2.95, 0.9, {'shape': 'circle', 'circle': 0.35}],
            [4.3, 0.6, {'shape': 'concave', 'concave': (0.6, 1.1, 0.1)}],
            [0.8, 4.63, {'shape': 'rect', 'rect': (0.23, 0.37)}],
            [1.5, 1.9, {'shape': 'rect', 'rect': (1.8, 0.1)}],
            [1.2, 2.7, {'shape': 'rect', 'rect': (1.8, 0.1)}],
            [4.2, 3.3, {'shape': 'rect', 'rect': (0.92, 0.1)}],
            # [3.6, 3.2, {'shape': 'rect', 'rect': (0.1, 0.23)}],
            [5.1, 2.4, {'shape': 'rect', 'rect': (0.23, 0.2)}],
            [4.7, 4.3, {'shape': 'rect', 'rect': (0.2, 0.69)}],
            [0.7, 3.6, {'shape': 'rect', 'rect': (0.23, 0.2)}],
        ]
        for i in range(len(ob_pos)):
            vertices = gen_polygon_vertices(ob_pos[i][:2], ob_pos[i][2])
            polygon = Polygon(vertices)
            pos = (polygon.centroid.x, polygon.centroid.y)
            min_rect = polygon.minimum_rotated_rectangle
            width, height = min_rect.bounds[2] - min_rect.bounds[0], min_rect.bounds[3] - min_rect.bounds[1]
            radius = np.sqrt((width / 2) ** 2 + (height / 2) ** 2)
            obstacles.append(PolyObstacle(pos=pos, shape_dict={'shape': 'polygon', 'feature': radius},
                                          idx=i, tid='obstacle' + str(i), vertices=vertices))

        obs_vertices = [[[-0.1, -0.1], [6.1, -0.1], [6.1, 0.0], [-0.1, 0.0]],
                        [[6.0, 0.0], [6.1, 0.0], [6.1, 5.1], [6.0, 5.1]],
                        [[-0.1, 5.0], [6.1, 5.0], [6.1, 5.1], [-0.1, 5.1]],
                        [[-0.1, 0.0], [0.0, 0.0], [0.0, 5.0], [-0.1, 5.0]],
                        ]
        for i, vertices in enumerate(obs_vertices):
            vertices = np.array(vertices)
            pos, radius = enclosing_circle(vertices)
            obstacles.append(PolyObstacle(pos=pos, shape_dict={'shape': 'polygon', 'feature': radius},
                                          idx=i, tid='obstacle' + str(i), vertices=vertices))
        return obstacles

    def small_128_map(self):
        self.xDim, self.yDim = 32.0, 32.0
        obstacles = []
        ob_pos = [
            [0.0, 3.75, {'shape': 'rect', 'rect': (2.5, 0.5)}],
            [2.5, 4.0, {'shape': 'rect', 'rect': (0.25, 4.0)}],
            [2.75, 3.75, {'shape': 'rect', 'rect': (2.5, 0.5)}],
            [5.25, 2.5, {'shape': 'rect', 'rect': (0.25, 5.5)}],
            [2.5, 0.0, {'shape': 'rect', 'rect': (0.25, 3.25)}],
            [2.75, 1.5, {'shape': 'rect', 'rect': (5.25, 0.5)}],
            [5.25, 0.0, {'shape': 'rect', 'rect': (0.25, 1.5)}]
        ]

        for i in range(len(ob_pos)):
            vertices = gen_polygon_vertices(ob_pos[i][:2], ob_pos[i][2])
            vertices = np.array(vertices) * 4
            polygon = Polygon(vertices)
            pos = (polygon.centroid.x, polygon.centroid.y)
            min_rect = polygon.minimum_rotated_rectangle
            width, height = min_rect.bounds[2] - min_rect.bounds[0], min_rect.bounds[3] - min_rect.bounds[1]
            radius = np.sqrt((width / 2) ** 2 + (height / 2) ** 2)
            obstacles.append(PolyObstacle(shape_dict={'shape': 'polygon', 'feature': radius}, idx=i,
                                          tid='obstacle' + str(i), vertices=vertices, pos=pos, ))

        return obstacles

    def small_indoor_map(self):
        self.xDim, self.yDim = 6.8, 6.6
        obstacles = []
        ob_pos = [
            [-0.5, 0.0, {'shape': 'rect', 'rect': (2.44, 1.53)}],
            [0.8, 2.58, {'shape': 'rect', 'rect': (0.43, 0.41)}],
            [2.3, -0.1, {'shape': 'concave', 'concave': (1.9, 3.1, 0.1)}],
            [1.2, 4.68, {'shape': 'circle', 'circle': 0.26}],
            [4.5, -0.1, {'shape': 'concave', 'concave': (1.1, 2.2, 0.1)}],
            [1.4, 6.13, {'shape': 'rect', 'rect': (0.23, 0.56)}],
            [5.3, 6.3, {'shape': 'rect', 'rect': (0.2, 0.4)}],
            [2.2, 4.5, {'shape': 'wenhao'}]]

        for i in range(len(ob_pos)):
            vertices = gen_polygon_vertices(ob_pos[i][:2], ob_pos[i][2])
            polygon = Polygon(vertices)
            pos = (polygon.centroid.x, polygon.centroid.y)
            min_rect = polygon.minimum_rotated_rectangle
            width, height = min_rect.bounds[2] - min_rect.bounds[0], min_rect.bounds[3] - min_rect.bounds[1]
            radius = np.sqrt((width / 2) ** 2 + (height / 2) ** 2)
            obstacles.append(PolyObstacle(shape_dict={'shape': 'polygon', 'feature': radius}, idx=i,
                                          tid='obstacle' + str(i), vertices=vertices, pos=pos, ))

        obs_vertices = [
            [[-0.5, -0.1], [6.8, -0.1], [6.8, 0.0], [-0.5, 0.0]],
            [[6.7, 0.0], [6.8, 0.0], [6.8, 6.7], [6.7, 6.7]],
            [[-0.5, 6.6], [6.8, 6.6], [6.8, 6.7], [-0.5, 6.7]],
            [[-0.6, -0.1], [-0.5, -0.1], [-0.5, 6.7], [-0.6, 6.7]],
        ]
        for i, vertices in enumerate(obs_vertices):
            i = len(obstacles)
            vertices = np.array(vertices)
            pos, radius = enclosing_circle(vertices)
            obstacles.append(PolyObstacle(shape_dict={'shape': 'polygon', 'feature': radius}, idx=i,
                                          tid='obstacle' + str(i), vertices=vertices, pos=pos, ))
        return obstacles

    def small_indoor_map2(self):
        self.xDim, self.yDim = 6.6, 6.6
        obstacles = []
        x_temp, y_temp = 0.6, 1.6
        ob_pos = [
            [0.9, 2.58, {'shape': 'rect', 'rect': (0.6, 0.5)}],
            # [0.6, 2.5, {'shape': 'rect', 'rect': (0.3, 1.0)}],
            [3.0, 5.6, {'shape': 'rect', 'rect': (0.23, 0.2)}],
            # [3.4, 1.8, {'shape': 'rect', 'rect': (0.23, 0.2)}],
            # [2.4, 0.0, {'shape': 'rect', 'rect': (0.1, 2.99)}],
            [2.4, -0.1, {'shape': 'concave', 'concave': (1.8, 3.0, 0.1)}],
            [1.2, 4.68, {'shape': 'circle', 'circle': 0.3}],
            [4.6, -0.1, {'shape': 'concave', 'concave': (1.0, 2.2, 0.1)}],
            [1.4, 5.9, {'shape': 'rect', 'rect': (0.23, 0.8)}],
            # [3.4, 3.8, {'shape': 'rect', 'rect': (0.1, 2.8)}],
            [5.3, 6.15, {'shape': 'rect', 'rect': (0.2, 0.46)}],
            [1.0, 1.2, {'shape': 'rect', 'rect': (0.23, 0.2)}],
            [2.2, 4.5, {'shape': 'wenhao'}]]

        for i in range(len(ob_pos)):
            vertices = gen_polygon_vertices(ob_pos[i][:2], ob_pos[i][2])
            polygon = Polygon(vertices)
            pos = (polygon.centroid.x, polygon.centroid.y)
            min_rect = polygon.minimum_rotated_rectangle
            width, height = min_rect.bounds[2] - min_rect.bounds[0], min_rect.bounds[3] - min_rect.bounds[1]
            radius = np.sqrt((width / 2) ** 2 + (height / 2) ** 2)
            obstacles.append(PolyObstacle(shape_dict={'shape': 'polygon', 'feature': radius}, idx=i,
                                          tid='obstacle' + str(i), vertices=vertices, pos=pos, ))

        obs_vertices = [
            [[-0.1, -0.1], [6.7, -0.1], [6.7, 0.0], [-0.1, 0.0]],
            [[6.6, 0.0], [6.7, 0.0], [6.7, 6.7], [6.6, 6.7]],
            [[-0.1, 6.6], [6.7, 6.6], [6.7, 6.7], [-0.1, 6.7]],
            [[-0.1, 0.0], [0.0, 0.0], [0.0, 6.6], [-0.1, 6.6]],
        ]
        for i, vertices in enumerate(obs_vertices):
            i = len(obstacles)
            vertices = np.array(vertices)
            pos, radius = enclosing_circle(vertices)
            obstacles.append(PolyObstacle(shape_dict={'shape': 'polygon', 'feature': radius}, idx=i,
                                          tid='obstacle' + str(i), vertices=vertices, pos=pos, ))
        return obstacles

    def small_indoor_map_copy(self):
        self.xDim, self.yDim = 6., 5.
        obstacles = []
        ob_pos = [
            [0.3, 0.98, {'shape': 'rect', 'rect': (0.6, 0.5)}],
            [0.0, 0.9, {'shape': 'rect', 'rect': (0.3, 1.0)}],
            [1.8, 0, {'shape': 'rect', 'rect': (0.1, 3.68)}],
            [2.95, 0.9, {'shape': 'circle', 'circle': 0.35}],
            [4.1, 0.8, {'shape': 'concave', 'concave': (1.0, 2.1, 0.1)}],
            [0.8, 4.63, {'shape': 'rect', 'rect': (0.23, 0.37)}],
            [2.8, 2.2, {'shape': 'rect', 'rect': (0.1, 2.8)}],
            [3.7, 4.0, {'shape': 'rect', 'rect': (0.23, 0.2)}],
            [4.7, 4.3, {'shape': 'rect', 'rect': (0.2, 0.69)}],
            [0.7, 3.0, {'shape': 'rect', 'rect': (0.23, 0.2)}],
        ]
        for i in range(len(ob_pos)):
            vertices = gen_polygon_vertices(ob_pos[i][:2], ob_pos[i][2])
            polygon = Polygon(vertices)
            pos = (polygon.centroid.x, polygon.centroid.y)
            min_rect = polygon.minimum_rotated_rectangle
            width, height = min_rect.bounds[2] - min_rect.bounds[0], min_rect.bounds[3] - min_rect.bounds[1]
            radius = np.sqrt((width / 2) ** 2 + (height / 2) ** 2)
            obstacles.append(PolyObstacle(shape_dict={'shape': 'polygon', 'feature': radius}, idx=i,
                                          tid='obstacle' + str(i), vertices=vertices, pos=pos, ))

        obs_vertices = [
            [[-0.1, -0.1], [6.1, -0.1], [6.1, 0.0], [-0.1, 0.0]],
            [[6.0, 0.0], [6.1, 0.0], [6.1, 5.1], [6.0, 5.1]],
            [[-0.1, 5.0], [6.1, 5.0], [6.1, 5.1], [-0.1, 5.1]],
            [[-0.1, 0.0], [0.0, 0.0], [0.0, 5.0], [-0.1, 5.0]],
        ]
        for i, vertices in enumerate(obs_vertices):
            i = len(obstacles)
            vertices = np.array(vertices)
            pos, radius = enclosing_circle(vertices)
            obstacles.append(PolyObstacle(shape_dict={'shape': 'polygon', 'feature': radius}, idx=i,
                                          tid='obstacle' + str(i), vertices=vertices, pos=pos, ))
        return obstacles

    def poly_obs_map(self):
        """
        大范围的二维场景，多边形绘制，680m*500m
        Initialize obstacles' positions
        :return: map of obstacles
        """
        self.xDim, self.yDim = 680, 500
        x, y = self.xDim, self.yDim
        poly_obs = []
        obs_vertices = [self.rect_vertices(0, 0, x, 1), self.rect_vertices(x - 1, 1, x, y - 1),
                        self.rect_vertices(0, y - 1, x, y), self.rect_vertices(0, 1, 1, y - 1),
                        [[100, 299], [249, 299], [249, 0], [250, 0], [250, 300], [100, 300]],
                        [[374, 200], [375, 200], [375, 500], [374, 500]],
                        [[499, 0], [500, 0], [500, 400], [499, 400]]
                        ]
        for i, vertices in enumerate(obs_vertices):
            vertices = np.array(vertices)
            pos, radius = enclosing_circle(vertices)
            poly_obs.append(PolyObstacle(pos=pos, shape_dict={'shape': 'polygon', 'feature': radius},
                                         idx=i, tid='obstacle' + str(i), vertices=vertices))

        return poly_obs

    def symmetrical_map(self):
        """
        大范围的走廊场景，多边形绘制, 800*200
        """

        self.xDim, self.yDim = 800, 200
        x, y = self.xDim, self.yDim
        poly_obs = []
        obs_vertices = [self.rect_vertices(0, 0, x, 1), self.rect_vertices(x - 1, 1, x, y - 1),
                        self.rect_vertices(0, y - 1, x, y), self.rect_vertices(0, 1, 1, y - 1),
                        [[200, 50], [400, 50], [400, 100], [300, 150], [200, 100]],
                        ]
        for i, vertices in enumerate(obs_vertices):
            vertices = np.array(vertices)
            pos, radius = enclosing_circle(vertices)
            poly_obs.append(PolyObstacle(pos=pos, shape_dict={'shape': 'polygon', 'feature': radius},
                                         idx=i, tid='obstacle' + str(i), vertices=vertices))

        return poly_obs

    def poly_maze_map(self):
        """
        大范围的二维迷宫场景，多边形绘制, 500m*500m
        Initialize obstacles' positions
        return: map of obstacles
        """
        self.xDim, self.yDim = 500., 500.
        x, y = self.xDim, self.yDim
        poly_obs = []
        obs_vertices = [self.rect_vertices(0, 0, x, 1), self.rect_vertices(x - 1, 1, x, y - 1),
                        self.rect_vertices(0, y - 1, x, y), self.rect_vertices(0, 1, 1, y - 1),
                        [[0, 50], [50, 50], [50, 100], [49, 100], [49, 51], [0, 51]],
                        [[0, 150], [100, 150], [100, 151], [0, 151]],
                        [[0, 200], [100, 200], [100, 201], [51, 201], [51, 250], [100, 250], [100, 251],
                         [51, 251], [51, 450], [50, 450], [50, 201], [0, 201]],
                        [[100, 0], [101, 0], [101, 50], [100, 50]],
                        [[300, 0], [301, 0], [301, 50], [350, 50], [350, 51], [251, 51], [251, 100],
                         [300, 100], [300, 101], [251, 101], [251, 150], [301, 150], [301, 200], [300, 200],
                         [300, 151], [251, 151], [251, 250], [300, 250], [300, 251], [251, 251], [251, 300],
                         [300, 300], [300, 301], [201, 301], [201, 400], [200, 400], [200, 351], [151, 351],
                         [151, 400], [150, 400], [150, 351], [101, 351], [101, 400], [100, 400], [100, 350],
                         [150, 350], [150, 301], [100, 301], [100, 300], [150, 300], [150, 200], [151, 200],
                         [151, 350], [200, 350], [200, 300], [250, 300], [250, 251], [200, 251], [200, 250],
                         [250, 250], [250, 201], [200, 201], [200, 200], [250, 200], [250, 51], [201, 51],
                         [201, 151], [150, 151], [150, 101], [100, 101], [100, 100], [150, 100], [150, 50],
                         [151, 50], [151, 150], [200, 150], [200, 50], [300, 50]],
                        [[100, 450], [201, 450], [201, 500], [200, 500], [200, 451], [100, 451]],
                        [[250, 450], [251, 450], [251, 500], [250, 500]],
                        [[400, 0], [401, 0], [401, 50], [400, 50]],
                        [[450, 0], [451, 0], [451, 101], [350, 101], [350, 100], [450, 100]],
                        [[350, 150], [401, 150], [401, 200], [450, 200], [450, 150], [451, 150], [451, 250],
                         [500, 250], [500, 251], [400, 251], [400, 250], [450, 250], [450, 201], [351, 201],
                         [351, 250], [350, 250], [350, 200], [400, 200], [400, 151], [350, 151]],
                        [[350, 300], [500, 300], [500, 301], [350, 301]],
                        [[250, 350], [500, 350], [500, 351], [250, 351]],
                        [[250, 400], [500, 400], [500, 401], [401, 401], [401, 450], [400, 450], [400, 401],
                         [351, 401], [351, 451], [300, 451], [300, 450], [350, 450], [350, 401], [250, 401]],
                        [[450, 450], [451, 450], [451, 500], [450, 500]],
                        ]

        for i, vertices in enumerate(obs_vertices):
            vertices = np.array(vertices)
            pos, radius = enclosing_circle(vertices)
            poly_obs.append(PolyObstacle(pos=pos, shape_dict={'shape': 'polygon', 'feature': radius},
                                         idx=i, tid='obstacle' + str(i), vertices=vertices))

        return poly_obs

    def corridor_map(self):
        """
        大范围的走廊场景，多边形绘制, 800*200
        """

        self.xDim, self.yDim = 800., 200.
        x, y = self.xDim, self.yDim
        poly_obs = []
        obs_vertices = [
            [[0, 0], [800.0, 0], [800.0, 4], [0, 4]],
            [[795.0, 4], [800.0, 4], [800.0, 200.0], [795.0, 200.0]],
            [[0, 195.0], [800.0, 195.0], [800.0, 200.0], [0, 200.0]],
            [[0, 4], [4, 4], [4, 195.0], [0, 195.0]],
            [[0, 44], [37, 44], [37, 67], [0, 67]],
            [[20, 142], [60, 142], [60, 182], [20, 182]],
            [[73, 43], [104, 43], [104, 103], [73, 103]],
            [[138, 15], [166, 15], [166, 43], [138, 43]],
            [[142, 126], [154, 126], [154, 156], [190, 156], [190, 200], [110, 200], [110, 156],
             [142, 156]],
            [[183, 63], [221, 63], [221, 101], [258, 101], [258, 136], [232, 136], [232, 111],
             [183, 111]],
            [[280, 0], [313, 0], [313, 43], [280, 43]],
            [[287, 72], [333, 72], [333, 89], [287, 89]],
            [[345, 184], [357, 184], [357, 200], [345, 200]],
            [[366, 23], [400, 23], [400, 33], [366, 33]],
            [[372, 73], [400, 73], [400, 104], [372, 104]],
            [[444, 32], [460, 32], [460, 66], [444, 66]],
            [[438, 122], [455, 122], [455, 139], [438, 139]],
            [[398, 159], [517, 159], [517, 176], [398, 176]],
            [[544, 31], [581, 31], [581, 47], [544, 47]],
            [[496, 65], [531, 65], [531, 76], [554, 76], [554, 111], [496, 111]],
            [[586, 74], [607, 74], [607, 98], [586, 98]],
            [[629, 38], [646, 38], [646, 59], [629, 59]],
            [[596, 129], [625, 129], [625, 158], [596, 158]],
            [[687, 34], [716, 34], [716, 59], [687, 59]],
            [[684, 89], [710, 89], [710, 138], [684, 138]],
            [[660, 166], [763, 166], [763, 200], [660, 200]],
            [[299, 121], [331, 121], [331, 149], [299, 149]],
        ]
        for i, vertices in enumerate(obs_vertices):
            vertices = np.array(vertices)
            pos, radius = enclosing_circle(vertices)
            poly_obs.append(PolyObstacle(pos=pos, shape_dict={'shape': 'polygon', 'feature': radius},
                                         idx=i, tid='obstacle' + str(i), vertices=vertices))

        return poly_obs

    def small_corridor_map(self):
        """
        大范围的走廊场景，多边形绘制, 800*200
        """

        self.xDim, self.yDim = 150., 100.
        poly_obs = []
        obs_vertices = [
            [[0.0, 0.0], [150.0, 0.0], [150.0, 0.3], [0.0, 0.3]],
            [[150.0, 0.3], [150.3, 0.3], [150.3, 100.3], [150.0, 100.3]],
            [[0.0, 100.0], [150.0, 100.0], [150.0, 100.3], [0.0, 100.3]],
            [[0.0, 0.3], [0.3, 0.3], [0.3, 100.0], [0.3, 100.0]],
            [[0.0, 23.5], [13.5, 23.5], [13.5, 32.5], [0.0, 32.5]],
            [[18.0, 71.0], [32.0, 71.0], [32.0, 81.0], [18.0, 81.0]],
            [[38.0, 23.0], [50.0, 23.0], [50.0, 46.0], [38.0, 46.0]],
            [[76.0, 0.0], [86.0, 0.0], [86.0, 14.0], [76.0, 14.0]],
            [[71., 76.], [77., 76.], [77., 85.], [92., 85.], [92., 100.], [57., 100.], [57., 85.], [71., 85.]],
            [[95., 37.], [110., 37.], [110., 50.], [129., 50.], [129., 65.], [116., 65.], [116., 55.], [95., 55.]],
            [[122.5, 92.0], [128.5, 92.0], [128.5, 100.0], [122.5, 100.0]]
        ]

        for i, vertices in enumerate(obs_vertices):
            vertices = np.array(vertices)
            v_list = [list(arr) for arr in vertices]
            # print(list(v_list))
            pos, radius = enclosing_circle(vertices)
            poly_obs.append(PolyObstacle(pos=pos, shape_dict={'shape': 'polygon', 'feature': radius},
                                         idx=i, tid='obstacle' + str(i), vertices=vertices))

        return poly_obs

    def read_json(self, name='/corridor_env.json'):
        # 从 json 文件中读取数据
        save_dir = os.path.dirname(os.path.realpath(__file__)) + '/map/'
        with open(save_dir + name, 'r') as file:
            data = json.load(file)
        dim = data['size']
        cmap = data['data']
        self.xDim, self.yDim = dim

        blocks = set()
        for p in cmap:
            blocks.add((p[0], p[1]))
        return blocks

    def write_env_cfg(self, blocks, res=1.0, name='/corridor_env.json'):
        corridor = {'size': [], 'resolution': res, 'data': []}
        save_dir = os.path.dirname(os.path.realpath(__file__)) + '/map/'
        os.makedirs(save_dir, exist_ok=True)
        for point in blocks:
            corridor['data'].append(point)
        corridor['size'] = [self.xDim, self.yDim]
        info_str = json.dumps(corridor, indent=4)
        with open(save_dir + name, 'w') as json_file:
            json_file.write(info_str)
        json_file.close()

    def plot_env(self, planner, title="JPS"):
        colors = rcParams['axes.prop_cycle'].by_key()['color']  # 获取默认颜色循环
        fig, ax = plt.subplots(figsize=(15, 12))

        path = planner.path
        # visited = [] if title[:3] == "GOS" or title[:3] == "GPS" else list(planner.CLOSED)
        s_start = planner.s_start_f
        s_goal = planner.s_goal_f

        # plot obstacles
        if self.is_grid_map:
            obs = self.obs_grid
            obs_x = [x[0] for x in obs]
            obs_y = [x[1] for x in obs]
            plt.plot(obs_x, obs_y, "sk", zorder=3)
        else:
            poly_obs = self.poly_obs
            for i in range(len(poly_obs)):
                color = colors[i % len(colors)]
                draw_polygon_2d(ax, poly_obs[i].vertices_pos, alpha=1)
                if title[:3] == "GOS":
                    draw_polygon_2d(ax, planner.inflated_obs[i].vertices_pos, fc_color='none', ec_color=color,
                                    alpha=0.9)

        px = [x[0] for x in path]
        py = [x[1] for x in path]
        ax.scatter(px[:], py[:], marker='s', color='red')
        plt.plot(px, py, linewidth=2, color='blue', zorder=3)
        plt.plot(s_start[0], s_start[1], marker='o', markersize=18, color='b', zorder=4)
        plt.plot(s_goal[0], s_goal[1], marker='*', markersize=20, color='r', zorder=4)

        plt.axis("equal")
        plt.title(title)
        plt.show()

    def plot_paths(self, paths, space=1.0):
        plt.rcParams['pdf.fonttype'] = 42  # 确保PDF使用Type 1字体
        plt.rcParams['ps.fonttype'] = 42  # 确保PostScript使用Type 1字体
        plt.rcParams['font.family'] = 'serif'  # 设置字体家族
        plt.rcParams['font.serif'] = ['Times New Roman']  # 设置字体为Times New Roman

        colors = rcParams['axes.prop_cycle'].by_key()['color']  # 获取默认颜色循环
        fig, ax = plt.subplots(figsize=(15, 12))

        x_range, y_range = (-space, self.xDim + space), (-space, self.yDim + space)
        min_xt, min_yt = x_range[0], y_range[0]
        width_t, height_t = x_range[1] - x_range[0], y_range[1] - y_range[0]
        color = colors[0 % len(colors)]
        rect = patches.Rectangle((min_xt, min_yt), width_t, height_t,
                                 linewidth=5, edgecolor=color, facecolor='none', alpha=1)
        ax.add_patch(rect)

        # 绘制障碍物
        if self.is_grid_map:
            obs = self.obs_grid
            obs_x = [x[0] for x in obs]
            obs_y = [x[1] for x in obs]
            plt.plot(obs_x, obs_y, "sk", zorder=3)
        else:
            color_b = [160/255, 160/255, 160/255]
            if len(self.poly_obs) < 1000:
                for i in range(len(self.poly_obs)):
                    color = colors[(i+1) % len(colors)]
                    # draw_polygon_2d(ax, self.poly_obs[i].vertices_pos, fc_color=color, ec_color=color, alpha=1)
                    draw_polygon_2d(ax, self.poly_obs[i].vertices_pos, fc_color=color_b, ec_color=color_b, alpha=1)

            else:
                from sklearn.cluster import DBSCAN

                coordinates = [[obs.pos_global_frame[0], obs.pos_global_frame[1]] for obs in self.poly_obs]
                coordinates = np.array(coordinates)
                # DBSCAN 聚类
                dbscan = DBSCAN(eps=0.5*16, min_samples=3)
                labels = dbscan.fit_predict(coordinates)
                cluster_indices = []
                for label in set(labels):
                    cluster_index = np.where(labels == label)[0]
                    cluster_indices.append(list(cluster_index))

                for i in range(len(cluster_indices)):
                    color = colors[(i + 1) % len(colors)]
                    for j in cluster_indices[i]:
                        # draw_polygon_2d(ax, self.poly_obs[j].vertices_pos, fc_color=color, ec_color=color, alpha=1)
                        draw_polygon_2d(ax, self.poly_obs[j].vertices_pos, fc_color=color_b, ec_color=color_b, alpha=1)

        # colours = ['red', 'blue', 'DarkOrange', 'DodgerBlue', 'purple']
        colours = np.array([[252, 141, 98], [166, 216, 84], [141, 160, 203], [241, 155, 138], [190, 152, 214]])/255
        labels = ['OVS (Ours)', 'JPS', 'RRT*', 'Neural A*', 'A*']
        line_styles = ['-', '--', ':', '-.', '-']
        markers = ['D', 's', '^', 'o', 'v']
        for i in range(len(paths)):
            if paths[i]:
                x_list, y_list = zip(*paths[i])
                if i == 0:
                    ax.plot(x_list, y_list, marker='o', label=labels[i], color=colours[i], linewidth=5,
                            linestyle='--', markersize=15, zorder=6)
                    # ax.scatter(x_list, y_list, color=colours[i], s=100, marker='o', zorder=5)
                else:
                    ax.plot(x_list, y_list, marker='o', label=labels[i], color=colours[i], linewidth=5,
                            linestyle='--', markersize=15, alpha=0.5)
                    # ax.scatter(x_list, y_list, color=colours[i], s=100, marker='o')

        ax.scatter([paths[0][0][0]], [paths[0][0][1]], color="orange", marker='o', zorder=6, s=400, label='Start')
        ax.scatter([paths[0][-1][0]], [paths[0][-1][1]], color="orange", marker='*', zorder=6, s=600, label='End')

        from matplotlib import font_manager
        plt.rcParams['pdf.fonttype'] = 42  # 确保PDF使用Type 1字体
        plt.rcParams['ps.fonttype'] = 42  # 确保PostScript使用Type 1字体
        plt.rcParams['font.family'] = 'serif'  # 设置字体家族
        plt.rcParams['font.serif'] = ['Times New Roman']  # 设置字体为Times New Roman
        font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'  # 确保路径正确
        prop = font_manager.FontProperties(fname=font_path)
        ax.legend(
            # ncol=10,
            prop={'family': 'Times New Roman', 'size': 20},
            loc='upper right',  # 图例的定位点，例如'upper left', 'upper right', 'lower left', 'lower right'等
            # bbox_to_anchor=(0.85, 1),  # 图例的锚点, (1.05, 1) 意味着图例位于图的右侧
            borderaxespad=0.2,  # 图例与轴之间的填充距离
            labelspacing=1.8
        )

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        # ax.axis('off')
        ax.axis('equal')
        plt.show()


def polygon_elements(idx, point):
    polygons = [
        [[0.0, 0.0], [277.5, 0.0], [277.5, 172.5], [0.0, 172.5]],
        [[0.0, 0.0], [300.0, 0.0], [300.0, 300.0], [0.0, 300.0]],
        [[0.0, 0.0], [232.5, 0.0], [232.5, 450.0], [0.0, 450.0]],
        [[0.0, 0.0], [210.0, 0.0], [210.0, 210.0], [0.0, 210.0]],
        [[0.0, 0.0], [90.0, 0.0], [90.0, 225.0], [360.0, 225.0], [360.0, 555.0], [-240.0, 555.0], [-240.0, 225.0],
         [0.0, 225.0]],
        [[0.0, 0.0], [285.0, 0.0], [285.0, 285.0], [562.5, 285.0], [562.5, 547.5], [367.5, 547.5], [367.5, 360.0],
         [0.0, 360.0]],
        [[0.0, 0.0], [90.0, 0.0], [90.0, 225.0], [360.0, 225.0], [360.0, 555.0], [-240.0, 555.0], [-240.0, 225.0],
         [0.0, 225.0]],
        [[0.0, 0.0], [285.0, 0.0], [285.0, 285.0], [562.5, 285.0], [562.5, 547.5], [367.5, 547.5], [367.5, 360.0],
         [0.0, 360.0]],
        [[0.0, 0.0], [247.5, 0.0], [247.5, 322.5], [0.0, 322.5]],
        [[0.0, 0.0], [345.0, 0.0], [345.0, 127.5], [0.0, 127.5]],
        [[0.0, 0.0], [210.0, 0.0], [210.0, 232.5], [0.0, 232.5]],
        [[0.0, 0.0], [892.5, 0.0], [892.5, 127.5], [0.0, 127.5]],
        [[0.0, 0.0], [262.5, 0.0], [262.5, 82.5], [435.0, 82.5], [435.0, 345.0], [0.0, 345.0]],
        [[0.0, 0.0], [262.5, 0.0], [262.5, 82.5], [435.0, 82.5], [435.0, 345.0], [0.0, 345.0]],
        [[0.0, 0.0], [217.5, 0.0], [217.5, 217.5], [0.0, 217.5]],
        [[0.0, 0.0], [217.5, 0.0], [217.5, 187.5], [0.0, 187.5]],
        [[0.0, 0.0], [195.0, 0.0], [195.0, 367.5], [0.0, 367.5]],
        [[0.0, 0.0], [772.5, 0.0], [772.5, 255.0], [0.0, 255.0]],
        [[0.0, 0.0], [240.0, 0.0], [240.0, 210.0], [0.0, 210.0]],
        # [[0, 0], [600, 0], [600, 100], [100, 100], [100, 1200], [600, 1200], [600, 1300], [0, 1300]],
        # [[0, 0], [600, 0], [600, 100], [100, 100], [100, 1200], [600, 1200], [600, 1300], [0, 1300]]
    ]
    offset_x = point[0]
    offset_y = point[1]
    # 将每个顶点坐标加偏移量
    moved_polygon = [[point[0] + offset_x, point[1] + offset_y] for point in polygons[idx]]
    return moved_polygon


def scale_polygon_vertices(polygon, combinedRadius):
    vertices_pos = []
    for vertice in polygon.vertices_:
        p1 = normalize(vertice.previous_.point_ - vertice.point_)
        p2 = normalize(vertice.next_.point_ - vertice.point_)
        p1p2 = p2 - p1
        nLeft, nRight = unit_normal_vector(p1p2)
        n_scale = combinedRadius / math.sin(math.acos(p1.dot(p2)) / 2)
        vertices_pos.append(vertice.point_ + n_scale * nRight)  # 障碍物的顶点逆时针表示，因此边的右侧即为向障碍物外部扩展
    return vertices_pos


def generate_large_obs_map(combined_radius=50, obs_num=50):
    x_range = 6000.
    y_range = 4000.
    iters = 10000
    all_obs_vertices = []
    obstacles = []
    out_line_poly = [[0, 0], [x_range, 0], [x_range, y_range], [0, y_range]]
    out_poly = Polygon(out_line_poly)
    while iters > 0:
        iters -= 1
        space = 2
        x = round(np.random.uniform(space * combined_radius, x_range - space * combined_radius), 2)
        y = round(np.random.uniform(space * combined_radius, y_range - space * combined_radius), 2)
        pos_start = [x, y]
        obs_vertices = polygon_elements(np.random.randint(0, 19), pos_start)
        is_intersect = False
        if obstacles:
            for obs in obstacles:
                scale_vertices = scale_polygon_vertices(obs, 2 * combined_radius)
                poly1 = Polygon(obs_vertices)
                poly2 = Polygon(scale_vertices)
                if poly1.intersects(poly2) or not poly1.within(out_poly):
                    is_intersect = True
                    break
        # else:
        #     all_obs_vertices.append(obs_vertices)
        #     i = len(obstacles)
        #     obs_vertices = np.array(obs_vertices)
        #     pos, radius = enclosing_circle(obs_vertices)
        #     obstacles.append(Obstacle(pos=pos, shape_dict={'shape': 'polygon', 'feature': radius},
        #                               idx=i, tid='obstacle' + str(i), vertices=obs_vertices, is_poly=True))
        if not is_intersect:
            all_obs_vertices.append(obs_vertices)
            i = len(obstacles)
            obs_vertices = np.array(obs_vertices)
            pos, radius = enclosing_circle(obs_vertices)
            obstacles.append(PolyObstacle(pos=pos, shape_dict={'shape': 'polygon', 'feature': radius},
                                          idx=i, tid='obstacle' + str(i), vertices=obs_vertices))
        print(len(obstacles), '------------------------current iteras-------------------------', iters)
        if len(obstacles) == obs_num:
            break
    # all_obs_vertices.pop(1)
    print(all_obs_vertices)
    plot_env(obstacles, [], boundary=out_line_poly)
    return obstacles


def gen_polygonal_vertices(center, radius, num_vertice, is_random=False):
    regular_polygon = True if num_vertice <= 12 else False  # 多边形边数少于6则默认为正多边形，否则为随机多边形

    if regular_polygon:
        vertices = []
        s_theta = np.random.uniform(0, 360 / num_vertice) if is_random else 360 / num_vertice / 2
        for theta in [np.deg2rad(i * (360 / num_vertice) + s_theta) for i in range(num_vertice)]:
            vertice = center + radius * np.array([np.cos(theta), np.sin(theta)])
            vertices.append(vertice)
    else:
        points_num = int(2 * np.pi * radius / 30.0)
        all_vertices = []
        for angular in [np.deg2rad(i * (360 / points_num)) for i in range(points_num)]:
            vertice = center + radius * np.array([np.cos(angular), np.sin(angular)])
            all_vertices.append([vertice, angular])
        vertices_theta = random.sample(all_vertices, num_vertice)
        vertices_theta = sorted(vertices_theta, key=lambda x: x[1])
        vertices = []
        for vertice in vertices_theta:
            vertices.append(vertice[0])

    return vertices


def gen_polygon_vertices(start_pos, shape_dict):
    vertices = []
    x, y = start_pos[0], start_pos[1]
    shape = shape_dict['shape']
    if shape == 'rect':
        length, width = shape_dict['rect']
        vertices = np.array([[x, y], [x + length, y], [x + length, y + width], [x, y + width]])
    elif shape == 'circle':
        radius = shape_dict['circle']
        vertices = gen_polygonal_vertices(start_pos, radius, 8, is_random=False)
    elif shape == 'concave':
        w, cave_l, con_w = shape_dict['concave']  # 实物实验设为0.6, 1.2, 0.1
        vertices = np.array([[x, y], [x + con_w, y],
                             [x + con_w, y + cave_l], [x + w, y + cave_l], [x + w, y + cave_l + con_w],
                             [x, y + cave_l + con_w]])
    elif shape == 'wenhao':
        vertices = np.array(
            [[0.0, 0.0], [0.67, -0.02], [0.96, -0.14], [1.2, -0.27], [1.51, -0.49], [1.8, -0.6], [2.12, -0.66],
             [2.4, -0.63], [2.66, -0.5], [2.85, -0.31], [2.96, 0.0], [2.98, 0.31], [2.85, 0.6], [2.65, 0.86],
             [2.4, 1.01], [2.18, 1.1], [2.16, 0.72], [2.42, 0.57], [2.59, 0.3], [2.57, 0.0], [2.38, -0.24],
             [2.08, -0.3], [1.8, -0.2], [1.52, 0.05], [1.2, 0.2], [0.9, 0.32], [0.67, 0.37], [0.0, 0.38]]
        ) + np.array([x, y])
    return vertices


def plot_env(obs, targets, is_grid=False, boundary=None):
    fig, ax = plt.subplots()
    colors = rcParams['axes.prop_cycle'].by_key()['color']  # 获取默认颜色循环

    # plot obstacles
    if boundary is not None:
        draw_polygon_2d(ax, boundary, fc_color='none', ec_color='red', alpha=1)
    if not is_grid:
        for i in range(len(obs)):
            color = colors[i % len(colors)]
            draw_polygon_2d(ax, obs[i].vertices_pos, alpha=1)
    else:
        obs_x = [x[0] for x in obs]
        obs_y = [x[1] for x in obs]
        plt.plot(obs_x, obs_y, "sk", zorder=3)

    if targets:
        for i in range(len(targets)):
            # color = colors[i % len(colors)]
            ob_rd = targets[i].radius
            pos = targets[i].pos
            ax.add_patch(
                plt.Circle((pos[0], pos[1]), radius=ob_rd, fc='none', ec='red', linewidth=1, linestyle='dashed'))
            plt.plot(pos[0], pos[1], color='red', marker='*', markersize=6, alpha=1, zorder=3)

    # plt.grid(True)
    # plt.xticks(np.arange(-0, 257, 4))  # x轴每隔0.6个单位显示一个刻度
    # plt.yticks(np.arange(-0, 257, 4))  # y轴每隔0.6个单位显示一个刻度
    plt.axis("equal")
    plt.show()


def path():
    path_data = dict()
    path_data['small'] = [
        [(1, 1), (9.722544155877284, 13.277455844122716), (11.277455844122716, 13.277455844122716),
         (20.722544155877284, 9.722544155877284), (22.277455844122716, 9.722544155877284), (30, 30)],
        [(1, 1), (9.75, 13.0), (10.0, 13.25), (11.0, 13.25), (14.5, 9.75), (22.0, 9.75), (22.25, 10.0), (30, 30)],
        [(1, 1), (3.0357217490046704, 5.938199769792099), (10.952940345165318, 14.652080174255792),
         (20.057979272541814, 9.339960143355514), (23.821666045032458, 9.090764413483912), [30, 30]],
        [(1.0, 1.0), (10.0, 14.0), (22.0, 9.0), (30, 30)],
        [(1, 1), (9.75, 13.0), (10.0, 13.25), (11.0, 13.25), (21.0, 9.75), (22.0, 9.75), (22.25, 10.0), (30, 30)]
    ]
    path_data['medium'] = [
        [(50.0, 6.0), (120.28290657996813, 63.83102157415167), (143.7147403368296, 108.11815842128317),
         (151.829675924646, 116.23631203127825), (244.28262342910068, 183.82532911479873),
         (244.28703696362084, 204.06776023546965), (232.2132855542327, 228.2132855542327),
         (224.06776023546965, 232.28703696362084), (207.72254415587727, 232.27745584412273),
         (196.27657713301275, 220.0899384068654), (216, 200)],
        [(50.0, 6.0), (116.0, 59.5), (120.0, 63.5), (120.5, 64.0), (143.5, 108.0), (164.0, 128.5), (240.0, 183.5),
         (244.5, 188.0), (244.5, 204.0), (232.5, 228.0), (228.0, 232.5), (212.0, 232.5), (207.5, 228.0),
         (207.5, 216.0), (216, 200)],
        [(50.0, 6.0), (90.34196243086956, 35.86670500272896), (120.49496095049207, 64.05518129912386),
         (122.20816469532, 103.18050167928546), (246.2974190222981, 178.89607472052458),
         (247.5890971302031, 200.2656006670695), (228.90023697480999, 236.77419970247414),
         (199.1940032328989, 231.19984164442135), [216, 200]],
        [[50.0, 6.0], (124.0, 64.0), (156.0, 136.0), (244.0, 184.0), (248.0, 204.0), (228.0, 236.0), (208.0, 232.0),
         (204.0, 208.0), (216.0, 200.0)],
        [(50.0, 6.0), (116.0, 59.5), (116.5, 60.0), (120.0, 63.5), (120.5, 64.0), (143.5, 108.0), (144.0, 108.5),
         (147.5, 112.0), (148.0, 112.5), (151.5, 116.0), (152.0, 116.5), (240.0, 183.5), (240.5, 184.0), (244.0, 187.5),
         (244.5, 188.0), (244.5, 204.0), (232.5, 228.0), (232.0, 228.5), (224.0, 232.5), (212.0, 232.5), (211.5, 232.0),
         (208.0, 228.5), (207.5, 228.0), (207.5, 216.0), (216, 200)]
    ]
    path_data['large'] = [
        [(105.0, 160.0), (305.43644660940674, 435.4435533905933), (1101.8764466094067, 976.5135533905932),
         (2969.3664466094065, 2221.1635533905933), (5137.036446609407, 3716.7935533905934), (5955.0, 3865.0)],
        [(105.0, 160.0), (1530, 275), (2940, 1685), (3865, 2370), (4555, 3060), (5405, 3360), (5580, 3440), (5955.0, 3865.0)],
        [(105.0, 160.0), (490.5646622399331, 851.0753113796447), (3560.0335185394392, 1952.668650289027), (5127.357634547346, 3719.4509440143124), [5955.0, 3865.0]],
        [],
        [(105.0, 160.0), (1530, 275), (2790, 1535), (3860, 2370), (4370, 2875), (5575, 3440), (5955.0, 3865.0)]
    ]
    return path_data


def path_neural():
    path_na = [(28.0, 26.0), (27.0, 25.0), (26.0, 24.0), (25.0, 23.0), (24.0, 22.0), (23.0, 21.0), (23.0, 20.0),
               (23.0, 19.0), (23.0, 18.0), (23.0, 17.0), (23.0, 16.0), (22.0, 15.0), (22.0, 14.0), (22.0, 13.0),
               (22.0, 12.0), (22.0, 11.0), (22.0, 10.0), (21.0, 9.0), (20.0, 9.0), (19.0, 9.0), (18.0, 9.0),
               (17.0, 9.0), (16.0, 10.0), (15.0, 9.0), (14.0, 9.0), (13.0, 10.0), (12.0, 11.0), (11.0, 12.0),
               (10.0, 13.0), (9.0, 12.0), (9.0, 11.0), (9.0, 10.0), (8.0, 9.0), (7.0, 8.0), (7.0, 7.0)]
    p1 = [[204.0, 88.0], [204.0, 84.0], [200.0, 80.0], [196.0, 76.0], [192.0, 76.0], [188.0, 72.0], [184.0, 72.0],
         [180.0, 68.0], [176.0, 64.0], [172.0, 60.0], [168.0, 56.0], [164.0, 60.0], [160.0, 64.0], [156.0, 68.0],
         [152.0, 72.0], [148.0, 76.0], [144.0, 80.0], [140.0, 84.0], [136.0, 80.0], [132.0, 76.0], [128.0, 80.0],
         [124.0, 84.0], [120.0, 88.0], [116.0, 92.0], [112.0, 96.0], [108.0, 100.0], [104.0, 104.0], [100.0, 108.0],
         [96.0, 112.0], [96.0, 116.0], [96.0, 120.0], [96.0, 124.0], [92.0, 128.0], [92.0, 132.0], [88.0, 136.0],
         [88.0, 140.0], [84.0, 144.0], [84.0, 148.0], [80.0, 152.0], [76.0, 156.0], [72.0, 160.0], [68.0, 164.0],
         [64.0, 168.0], [60.0, 172.0], [60.0, 176.0], [60.0, 180.0], [60.0, 184.0], [60.0, 188.0], [60.0, 192.0],
         [60.0, 196.0], [56.0, 200.0], [52.0, 204.0], [52.0, 208.0], [48.0, 212.0], [44.0, 216.0], [40.0, 220.0],
         [36.0, 224.0], [32.0, 224.0], [28.0, 220.0], [24.0, 224.0], [20.0, 228.0], [16.0, 224.0], [12.0, 220.0],
         [8.0, 216.0], [4.0, 212.0], [0.0, 208.0]]
    new_path = [path_na[0]]
    res = 0.25
    for i in range(len(path_na)-1):
        p1 = path_na[i]
        p2 = path_na[i+1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        for j in range(int(1/res)):
            p_new = (p1[0] + dx * j * res, p1[1] + dy * j * res)
            new_path.append(p_new)
    new_path.append(path_na[-1])
    print(new_path)


if __name__ == "__main__":
    envs_type = {
        0: "small", 1: "large_maze", 2: "large", 3: "large_corridor",
        4: "symmetric", 5: "large_obs", 6: "small_indoor", 7: "small_blocks", 8: "128", 9: "256"
    }
    grid_map = False
    environ = Env(envs_type[6], is_grid_map=grid_map)
    environ8 = Env(envs_type[8], is_grid_map=grid_map)
    environ9 = Env(envs_type[9], is_grid_map=grid_map)
    environ5 = Env(envs_type[5], is_grid_map=grid_map)
    path_dicts = path()
    environ8.plot_paths(path_dicts['small'], space=1.0)
    environ9.plot_paths(path_dicts['medium'], space=10.0)
    environ5.plot_paths(path_dicts['large'], space=50.0)

    # polygon_obs = environ.obs_grid
    # polygon_obs = environ.poly_obs
    # plot_env(polygon_obs, [], grid_map)

    # generate_large_obs_map(combined_radius=80, obs_num=50)
