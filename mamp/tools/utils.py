import numpy as np
from scipy import spatial
from itertools import combinations
from math import sqrt, cos, sin, acos, degrees, atan2, pi, hypot, tan, radians
from scipy.spatial import ConvexHull
from pyproj import Transformer
from shapely.geometry import Point, Polygon, LineString
from mamp.tools.vector import Vector2
from mamp.tools import rvo_math
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

trans2utm = Transformer.from_crs("EPSG:4326", "EPSG:3857")
trans2lonlat = Transformer.from_crs("EPSG:3857", "EPSG:4326")
eps = 1e5


def line_inter_circle(p1, p2, center, r, rs):
    """
    判断线段(x1, y1)-(x2, y2)是否与以(cx, cy)为圆心，半径为r的圆相交
    """
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    cx, cy = center.x, center.y

    dx1, dx2 = x1 - cx, x2 - cx
    dy1, dy2 = y1 - cy, y2 - cy
    distance1 = hypot(dx1, dy1)
    distance2 = hypot(dx2, dy2)
    if distance1 < rs and distance2 < rs:
        return False
    elif rs < distance1 < r or rs < distance2 < r:
        return True

    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - cx, y1 - cy

    a = dx * dx + dy * dy
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r

    discriminant = b * b - 4 * a * c

    if discriminant < 0:
        return False
    else:
        discriminant = sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)
        if 0 <= t1 <= 1:
            return True
        if 0 <= t2 <= 1:
            return True
        return False


def line_intersect_segment(point, direction, p1, p2):
    """判断直线与线段是否相交"""
    x0, y0 = point.x, point.y
    dx, dy = direction.x, direction.y
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    dx1 = x2 - x1
    dy1 = y2 - y1

    # 计算参数 t 和 u
    denominator = dx * dy1 - dy * dx1
    if abs(denominator) <= 1e-5:
        return False  # 平行，不相交

    u = ((x1 - x0) * dy - (y1 - y0) * dx) / denominator

    return 0 <= u <= 1


def line_intersects_circle(p1, p2, center, r):
    """
    判断线段(x1, y1)-(x2, y2)是否与以(cx, cy)为圆心，半径为r的圆相交
    """
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    cx, cy = center.x, center.y
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - cx, y1 - cy

    a = dx * dx + dy * dy
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r

    discriminant = b * b - 4 * a * c

    if discriminant < 0:
        return []
    else:
        discriminant = sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)
        points = []
        if 0 <= t1 <= 1:
            points.append(p1 + t1 * (p2 - p1))
        if 0 <= t2 <= 1:
            points.append(p2 + t2 * (p2 - p1))
        return points


def deg360(angle):
    return (angle + 360) % 360


def point_in_sector(point, center, r, rs, start_angle, end_angle):
    """
    判断点(px, py)是否在以(cx, cy)为圆心，半径为r，起始角度为start_angle，结束角度为end_angle的扇形内。
    """
    px, py = point.x, point.y
    cx, cy = center.x, center.y
    dx = px - cx
    dy = py - cy
    distance = hypot(dx, dy)
    if distance > r or distance < rs:
        return False

    angle = degrees(atan2(dy, dx))
    angle = (angle + 360) % 360

    if start_angle <= end_angle:
        return start_angle <= angle <= end_angle
    else:  # handle the case where the sector crosses the 0-degree line
        return angle >= start_angle or angle <= end_angle


def sector_line_intersection(center, r, rs, start_angle, end_angle, p1, p2):
    """
    判断线段p1p2是否与扇形相交
    """
    # 检查线段的两个端点是否在扇形内
    if point_in_sector(p1, center, r, rs, start_angle, end_angle) or \
            point_in_sector(p2, center, r, rs, start_angle, end_angle):
        return True

    # 检查线段是否与扇形的弧相交
    intersection_points = line_intersects_circle(p1, p2, center, r)
    if intersection_points:
        is_intersection = False
        start_angle = (start_angle + 360) % 360
        end_angle = (end_angle + 360) % 360
        for point in intersection_points:
            angle = degrees(atan2(point.y - center.y, point.x - center.x))
            angle = (angle + 360) % 360
            if start_angle <= end_angle:
                is_intersection = start_angle <= angle <= end_angle
            else:
                is_intersection = angle >= start_angle or angle <= end_angle
            if is_intersection:
                break
        return is_intersection
    else:
        return False


def robot_pose_to_outline(position, yaw, agent_rd):
    x, y = position.x, position.y
    # Vehicle parameters
    if agent_rd > 0.5:
        scale = agent_rd / 2.5
        LENGTH = 4.43 * scale  # [m]
        WIDTH = 2.0 * scale  # [m]
        BACK_TO_WHEEL = 0.9 * scale  # [m]
    else:
        LENGTH = 0.30  # [m]
        WIDTH = 0.195  # [m]
        BACK_TO_WHEEL = 0.05  # [m]
    l_norm = (LENGTH / 2) - BACK_TO_WHEEL
    x = x - l_norm * cos(yaw)
    y = y - l_norm * sin(yaw)
    outline = np.array(
        [[-BACK_TO_WHEEL, (LENGTH - BACK_TO_WHEEL), (LENGTH - BACK_TO_WHEEL),
          -BACK_TO_WHEEL],
         [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2]])

    Rot1 = np.array([[cos(yaw), sin(yaw)], [-sin(yaw), cos(yaw)]])
    outline = (outline.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    outline_coords = [list(coord) for coord in zip(*outline)]
    return outline_coords


def dist_point_line_segment(vector1, vector2, point):
    dist_sq = dist_sq_point_line_segment(vector1, vector2, point)
    return sqrt(dist_sq)


def transform_position(position, width, height, bounds):
    # Transform the position to fit within the window bounds.
    x_min, x_max, y_min, y_max = bounds

    x = (position.x - x_min) / (x_max - x_min) * width
    y = (position.y - y_min) / (y_max - y_min) * height

    return x, height - y  # Flip y-axis for correct display


def path_length(path):
    total_length = 0.0
    for i in range(len(path) - 1):
        total_length += l2norm(path[i], path[i + 1])
    return total_length


def enclosing_circle(points):
    # 计算凸包
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    # 找到凸包的中心点
    center = np.mean(hull_points, axis=0)

    # 计算凸包顶点到中心点的最大距离，作为外接圆的半径
    radius = np.max(np.linalg.norm(hull_points - center, axis=1))

    return center, radius


def subtract_angles(angle1, angle2):
    """
    角度为逆时针，计算两个角度的差值，结果在0到360度之间。
    :param angle1: 第一个角度（单位：度）
    :param angle2: 第二个角度（单位：度）
    :return: 两个角度相减的结果，范围在0到360度之间
    """
    difference = (angle1 - angle2) % 360
    if difference < 0:  # 确保结果为正
        difference += 360
    return difference


def turning_distance(p0, p1, p2, turning_radius, path=None):
    radius = turning_radius / 3
    p0, p1, p2 = Vector2(p0[0], p0[1]), Vector2(p1[0], p1[1]), Vector2(p2[0], p2[1])
    p0p1 = p1 - p0
    p1p2 = p2 - p1
    if p0p1 @ p1p2 < 0:
        norm_vel = rvo_math.normalize(p0p1)
        position_goal = p2
        combinedRadius = turning_radius
        combinedRadiusSq = combinedRadius * combinedRadius

        normal_left = Vector2(-norm_vel.y, norm_vel.x)
        center_left = p1 + turning_radius * normal_left
        relativePosLeft = center_left - position_goal
        distSqLeft = rvo_math.abs_sq(relativePosLeft)
        dist_all_left = 0.0
        dist_all_right = 0.0

        if distSqLeft >= combinedRadiusSq:
            start_left = p1 - center_left
            s_angle_left = deg360(degrees(atan2(start_left.y, start_left.x)))

            leg = sqrt(distSqLeft - turning_radius ** 2)
            left_direction = Vector2(relativePosLeft.x * leg - relativePosLeft.y * turning_radius,
                                     relativePosLeft.x * turning_radius + relativePosLeft.y * leg) / distSqLeft
            end_left = position_goal + leg * left_direction
            end_dir_left = end_left - center_left
            e_angle_left = deg360(degrees(atan2(end_dir_left.y, end_dir_left.x)))
            difference_left = subtract_angles(e_angle_left, s_angle_left)
            dist_left_l = (difference_left / 360) * 2 * pi * turning_radius
            dist_all_left = dist_left_l + rvo_math.l2norm(end_left, position_goal)

        normal_right = Vector2(norm_vel.y, -norm_vel.x)
        center_right = p1 + turning_radius * normal_right
        relativePosRight = center_right - position_goal
        distSqRight = rvo_math.abs_sq(relativePosRight)
        if distSqRight > combinedRadiusSq:
            start_right = p1 - center_right
            s_angle_right = deg360(degrees(atan2(start_right.y, start_right.x)))
            leg = sqrt(distSqRight - turning_radius ** 2)
            right_direction = Vector2(relativePosRight.x * leg + relativePosRight.y * turning_radius,
                                      -relativePosRight.x * turning_radius + relativePosRight.y * leg) / distSqRight
            end_right = position_goal + leg * right_direction
            end_dir_right = end_right - center_right
            e_angle_right = deg360(degrees(atan2(end_dir_right.y, end_dir_right.x)))
            difference_right = subtract_angles(s_angle_right, e_angle_right)
            dist_right_l = (difference_right / 360) * 2 * pi * turning_radius
            dist_all_right = dist_right_l + rvo_math.l2norm(end_right, position_goal)
        if path is not None:
            fig = plt.figure(0)
            fig_size = (10 * 1, 8 * 1)
            fig.set_size_inches(fig_size[0], fig_size[1])
            ax = fig.add_subplot(1, 1, 1)
            ax.set(xlabel='X', ylabel='Y', )
            ax.axis('equal')
            # Plot the left path
            if dist_all_left > 0:
                wedge = Wedge((center_left.x, center_left.y), turning_radius, s_angle_left, e_angle_left,
                              fill=False,
                              edgecolor='orange', linewidth=1.5, zorder=5)
                ax.add_patch(wedge)
                plt.plot(end_left.x, end_left.y, color='orange', marker='o', markersize=5)
                ax.plot([end_left.x, position_goal.x], [end_left.y, position_goal.y], color='orange', linewidth=0.5,
                        linestyle='dashed', alpha=0.9)
            # Plot the right path
            if dist_all_right > 0:
                wedge = Wedge((center_right.x, center_right.y), turning_radius, e_angle_right, s_angle_right,
                              fill=False,
                              edgecolor='blue', linewidth=1.5, zorder=5)
                ax.add_patch(wedge)
                plt.plot(end_right.x, end_right.y, color='blue', marker='o', markersize=5)
                ax.plot([end_right.x, position_goal.x], [end_right.y, position_goal.y], color='blue', linewidth=0.5,
                        linestyle='dashed', alpha=0.9)
            ax.add_patch(plt.Circle((p0.x, p0.y), radius=radius, fc='none', ec='purple', linewidth=1, zorder=5))
            ax.add_patch(plt.Circle((p1.x, p1.y), radius=radius, fc='none', ec='purple', linewidth=1, zorder=5))
            ax.add_patch(plt.Circle((p2.x, p2.y), radius=radius, fc='none', ec='purple', linewidth=1, zorder=5))
            plt.plot(center_left.x, center_left.y, color='red', marker='o', markersize=5)
            plt.plot(center_right.x, center_right.y, color='blue', marker='o', markersize=5)
            px = [p[0] for p in path]
            py = [p[1] for p in path]
            ax.plot(px, py, color='purple', linewidth=1.5, marker='o', linestyle='dashed', alpha=1)
            print('straight_distance:', rvo_math.l2norm(p1, p2))
            print('turning_distance:', min(dist_all_left, dist_all_right))
            # plt.show()
        if dist_all_left > 0 and dist_all_right > 0:
            return min(dist_all_left, dist_all_right)
        elif dist_all_left == 0 and dist_all_right == 0:
            return rvo_math.l2norm(p1, p2)
        else:
            return max(dist_all_left, dist_all_right)
    else:
        return rvo_math.l2norm(p1, p2)


def smooth_with_turning(path, obstacles, turning_radius=0.4):
    length = 0.0
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
                if len(path_smooth) <= 2:
                    length += l2norm(path_smooth[-1], path_smooth[-2])
                else:
                    length += turning_distance(path_smooth[-3], path_smooth[-2], path_smooth[-1], turning_radius, path_smooth)
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
                if c1 and not is_intersect_polys(pos, pos_next, obstacles):
                    path_smooth.append((route[i][0], route[i][1]))
                    if len(path_smooth) <= 2:
                        length += l2norm(path_smooth[-1], path_smooth[-2])
                    else:
                        length += turning_distance(path_smooth[-3], path_smooth[-2], path_smooth[-1], turning_radius, path_smooth)
                    for j in range(i):
                        route.pop(0)
                    break
    return path_smooth, length


def smooth(path, obstacles):
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
                if not is_intersect_polys(pos, pos_next, obstacles):
                    path_smooth.append((route[i][0], route[i][1]))
                    for j in range(i):
                        route.pop(0)
                    break
    return path_smooth, path_length(path_smooth)


def intersect_ploy_edges(pos1, pos2, vertices):
    num_vertices = len(vertices)
    is_intersect = False

    for i in range(num_vertices):
        point = vertices[i].point_
        point_next = vertices[(i + 1) % num_vertices].point_
        is_intersect = seg_is_intersec(pos1, pos2, point, point_next)
        if is_intersect:
            return is_intersect

    return is_intersect


def is_intersect_polys(pos, pos_next, obstacles):
    for obstacle in obstacles:
        if intersect_ploy_edges(pos, pos_next, obstacle.vertices_):
            return True
    return False


def pos_in_polygons(pos, polygons):
    for ob in polygons:
        polygon = Polygon(ob.vertices_pos)
        point = Point((pos[0], pos[1]))
        if point.within(polygon) or point.touches(polygon):
            return True
    return False


def circle_polygon_intersect(center, radius, poly_coords):
    """
    Determines whether a circle and a polygon intersect or touch each other.

    Parameters:
    circle_center: coordinates of the center of the circle in the format (x, y)
    circle_radius: radius of the circle
    polygon_coords: coordinates of the vertices of the polygon in the format [(x1, y1), (x2, y2), ...]

    Returns:
    If the circle and the polygon intersect or touch each other, returns True; otherwise, returns False.
    """
    circle = Point(center).buffer(radius)
    polygon = Polygon(poly_coords)
    return circle.intersects(polygon)


def min_enclosing_circle(points):
    """多边形的最小包围圆"""
    # 从点集中选取一个、两个或三个点
    for k in range(1, 4):
        for pts in combinations(points, k):
            pts = np.array(pts)
            if k == 1:
                center, radius = pts[0], 0.0
            elif k == 2:
                center = (pts[0] + pts[1]) / 2
                radius = np.linalg.norm(pts[0] - center)
            else:
                A, B, C = pts
                a = np.linalg.norm(B - C)
                b = np.linalg.norm(A - C)
                c = np.linalg.norm(A - B)
                s = (a + b + c) / 2
                area_sq = s * (s - a) * (s - b) * (s - c)
                if area_sq > 0:
                    area = sqrt(s * (s - a) * (s - b) * (s - c))
                    radius = a * b * c / (4 * area)
                else:
                    return points[0], 0.
                # Circumcenter from barycentric coordinates
                deter = (A[0] - C[0]) * (B[1] - C[1]) - (B[0] - C[0]) * (A[1] - C[1])
                if deter == 0:
                    continue  # Collinear points; circumcenter doesn't exist
                center_x = ((A[0] ** 2 + A[1] ** 2) * (B[1] - C[1]) + (B[0] ** 2 + B[1] ** 2) * (C[1] - A[1]) + (
                        C[0] ** 2 + C[1] ** 2) * (A[1] - B[1])) / (2 * deter)
                center_y = ((A[0] ** 2 + A[1] ** 2) * (C[0] - B[0]) + (B[0] ** 2 + B[1] ** 2) * (A[0] - C[0]) + (
                        C[0] ** 2 + C[1] ** 2) * (B[0] - A[0])) / (2 * deter)
                center = np.array([center_x, center_y])

            if all(sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) <= radius for x, y in points):  # 检测一个圆是否包含所有点
                return center, radius

    # 如果以上都没有返回，则取所有点的中心和最远点距离作为最后的尝试
    center = np.mean(points, axis=0)
    radius = np.max(np.linalg.norm(points - center, axis=1))
    return center, radius


def dist_sq_point_line_segment(vector1, vector2, vector3):
    """
    Computes the squared distance from a line segment with the specified endpoints to a specified point.

    Args:
        vector1 (Vector2): The first endpoint of the line segment.
        vector2 (Vector2): The second endpoint of the line segment.
        vector3 (Vector2): The point to which the squared distance is to be calculated.

    Returns:
        float: The squared distance from the line segment to the point.
    """
    r = np.dot((vector3 - vector1), (vector2 - vector1)) / absSq(vector2 - vector1)

    if r < 0.0:
        return absSq(vector3 - vector1)

    if r > 1.0:
        return absSq(vector3 - vector2)

    return absSq(vector3 - (vector1 + r * (vector2 - vector1)))


def to_left(p, q, s):
    """
    判断点s是否在有向线段pq的左边
    :param p: (x1, y1)
    :param q: (x2, y2)
    :param s: (x, y)
    :return: True or False
    """
    return area2(p, q, s) > 0


def area2(p, q, s):
    """
    正向（逆时针方向）三角形的有向面积
    :param p: (x1, y1)
    :param q: (x2, y2)
    :param s: (x, y)
    :return: 有向面积
    """
    return p[0] * q[1] - p[1] * q[0] + q[0] * s[1] - q[1] * s[0] + s[0] * p[1] - s[1] * p[0]


# 经纬度转utm坐标系，单位为m，x轴指向东，y轴指向北
def lonlat2grid(lonlat_pos):
    x, y = trans2utm.transform(lonlat_pos[1], lonlat_pos[0])
    return [round(x, 3), round(y, 3)]


def grid2lonlat(utm_pos):
    lat, lon = trans2lonlat.transform(utm_pos[0], utm_pos[1])
    return [round(lon, 8), round(lat, 8)]


# 矩形区域的四条边界线
def get_boundaries(area):
    min_x = 999999999999.
    max_x = -999999999999.
    min_y = 999999999999.
    max_y = -999999999999.
    for k in range(len(area) - 1):
        min_x = min(min_x, area[k][0])
        min_y = min(min_y, area[k][1])
        max_x = max(max_x, area[k][0])
        max_y = max(max_y, area[k][1])
    return min_x, max_x, min_y, max_y


def takeSecond(elem):
    return elem[1]


def pedal(p1, p2, p3):
    """
    过p3作p1和p2相连直线的垂线, 计算垂足的坐标
    直线1: 垂足坐标和p3连线
    直线2: p1和p2连线
    两条直线垂直, 且交点为垂足
    :param p1: (x1, y1)
    :param p2: (x2, y2)
    :param p3: (x3, y3)
    :return: 垂足坐标 (x, y)
    """
    if p2[0] != p1[0]:
        # ########## 根据点x1和x2计算线性方程的k, b
        k, b = np.linalg.solve([[p1[0], 1], [p2[0], 1]], [p1[1], p2[1]])  # 得到k和b
        # #######原理: 垂直向量数量积为0
        x = np.divide(((p2[0] - p1[0]) * p3[0] + (p2[1] - p1[1]) * p3[1] - b * (p2[1] - p1[1])),
                      (p2[0] - p1[0] + k * (p2[1] - p1[1])))
        y = k * x + b

    else:  # 点p1和p2的连线垂直于x轴时
        x = p1[0]
        y = p3[1]

    return np.array([x, y])


# keep angle between [-pi, pi]
def wrap(angle):
    while angle >= np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def left_of(a, b, c):
    """
    line point 1: a
    line point 2: b
    point: c
    """
    return det(a - c, b - a)  #


def l2norm(p1, p2):
    """ Compute Euclidean distance in 2D domains"""
    p1, p2 = np.array(p1[:2]), np.array(p2[:2])
    return round(np.linalg.norm(p2 - p1), 5)


def l2normsq(x, y):
    return (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2


def sqr(a):
    return a ** 2


def unit_normal_vector(p1p2):
    """ Compute the unit normal vector of  vector p1p2"""
    nRight = normalize(np.array([p1p2[1], -p1p2[0]]))
    nLeft = -nRight
    return nLeft, nRight


def linear_equation(p1, p2):
    a = p2[1] - p1[1]
    b = p1[0] - p2[0]
    c = p1[0] * (p1[1] - p2[1]) + p1[1] * (p2[0] - p1[0])
    return a, b, c


def norm(vec):
    return round(np.linalg.norm(vec), 5)


def normalize(vec):
    if np.linalg.norm(vec) > 0:
        return vec / np.linalg.norm(vec)
    else:
        return np.array([0.0, 0.0])


def absSq(vec):
    return np.dot(vec, vec)


def det(p, q):
    return p[0] * q[1] - p[1] * q[0]


def pi_2_pi(angle):  # to -pi-pi
    return (angle + np.pi) % (2 * np.pi) - np.pi


def mod2pi(theta):  # to 0-2*pi
    return theta % (2 * np.pi)


def is_parallel(vec1, vec2):
    """ 判断二个向量是否平行 """
    assert vec1.shape == vec2.shape, r'输入的参数 shape 必须相同'
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    vec1_normalized = vec1 / norm_vec1
    vec2_normalized = vec2 / norm_vec2
    if norm_vec1 <= 1e-3 or norm_vec2 <= 1e-3:
        return True
    elif 1.0 - abs(np.dot(vec1_normalized, vec2_normalized)) < 1e-3:
        return True
    else:
        return False


def angle_2_vectors(v1, v2):
    v1v2_norm = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if v1v2_norm == 0.0:
        v1v2_norm = 1e-5
    cosdv = np.dot(v1, v2) / v1v2_norm
    if cosdv > 1.0:
        cosdv = 1.0
    elif cosdv < -1.0:
        cosdv = -1.0
    else:
        cosdv = cosdv
    angle = acos(cosdv)
    return angle


# 判断点是否在圆内
def point_in_circle(pos, obj_pos, combinedRadius):
    if l2norm(pos, obj_pos) < combinedRadius:
        return True
    return False


# 判断线段和圆是否相交
def seg_cross_circle(p_1, p_2, obj_pos, combinedRadius):
    p1_in_circle = point_in_circle(p_1, obj_pos, combinedRadius)
    p2_in_circle = point_in_circle(p_2, obj_pos, combinedRadius)
    if p1_in_circle and p2_in_circle:
        return False
    elif (p1_in_circle and not p2_in_circle) or (not p1_in_circle and p2_in_circle):
        return True
    if p_1[0] == p_2[0]:  # 当x相等
        a, b, c = 1, 0, -p_1[0]
    elif p_1[1] == p_2[1]:  # 当y相等
        a, b, c = 0, 1, -p_1[1]
    else:
        a = p_1[1] - p_2[1]
        b = p_2[0] - p_1[0]
        c = p_1[0] * p_2[1] - p_1[1] * p_2[0]
    dist_1 = (a * obj_pos[0] + b * obj_pos[1] + c) ** 2
    dist_2 = (a * a + b * b) * combinedRadius * combinedRadius
    if dist_1 > dist_2:  # 点到直线距离大于半径r  不相交
        return False
    angle_1 = (obj_pos[0] - p_1[0]) * (p_2[0] - p_1[0]) + (obj_pos[1] - p_1[1]) * (p_2[1] - p_1[1])
    angle_2 = (obj_pos[0] - p_2[0]) * (p_1[0] - p_2[0]) + (obj_pos[1] - p_2[1]) * (p_1[1] - p_2[1])
    if angle_1 > 0 and angle_2 > 0:
        return True
    return False


def cross(p1, p2, p3):  # 跨立实验
    x1 = p2[0] - p1[0]
    y1 = p2[1] - p1[1]
    x2 = p3[0] - p1[0]
    y2 = p3[1] - p1[1]
    return float(x1 * y2 - x2 * y1)


def seg_is_intersec(p1, p2, p3, p4):  # 判断线段p1p2与线段p3p4是否相交
    # 快速排斥，以l1、l2为对角线的矩形必相交，否则两线段不相交
    if (max(p1[0], p2[0]) >= min(p3[0], p4[0])  # 矩形1最右端大于矩形2最左端
            and max(p3[0], p4[0]) >= min(p1[0], p2[0])  # 矩形2最右端大于矩形最左端
            and max(p1[1], p2[1]) >= min(p3[1], p4[1])  # 矩形1最高端大于矩形最低端
            and max(p3[1], p4[1]) >= min(p1[1], p2[1])):  # 矩形2最高端大于矩形最低端

        # 若通过快速排斥则进行跨立实验
        c1 = cross(p1, p2, p3) * cross(p1, p2, p4)
        c2 = cross(p3, p4, p1) * cross(p3, p4, p2)
        if c1 <= 0. and c2 <= 0.:
            D = True
        else:
            D = False
    else:
        D = False
    return D


def line_intersect(p1, p2, p3, p4):
    line1 = LineString([p1, p2])
    line2 = LineString([p3, p4])
    return line1.intersects(line2)


def point_line_dist(line_p1, line_p2, p):
    return np.linalg.norm(np.cross(line_p2 - line_p1, line_p1 - p)) / np.linalg.norm(line_p2 - line_p1)


def signed_distance_to_line(p1, p2, p):
    """
    Calculate the signed distance from point p to the directed line segment (p1, p2).

    Argus:
    p : Coordinates of point p
    p1, p2 : Coordinates of the endpoints of the directed line segment

    Returns:
    float
        If the result is greater than 0, the point p is on the left side of the directed line segment,
        otherwise, it's on the right side.
    """
    v1 = np.array(p) - np.array(p1)
    v2 = np.array(p2) - np.array(p1)
    cross_product = np.cross(v1, v2)
    return cross_product


def dist_sq_point_line_seg(vector1, vector2, point):
    """
    Computes the squared distance from a line segment with the specified endpoints to a specified point.

    Args:
        vector1 (Vector2): The first endpoint of the line segment.
        vector2 (Vector2): The second endpoint of the line segment.
        point (Vector2): The point to which the squared distance is to be calculated.

    Returns:
        float: The squared distance from the line segment to the point.
    """
    r = np.dot((point - vector1), (vector2 - vector1)) / absSq(vector2 - vector1)

    if r < 0.0:
        return absSq(point - vector1)

    if r > 1.0:
        return absSq(point - vector2)

    return absSq(point - (vector1 + r * (vector2 - vector1)))


def determin_between_line(vector1, vector2, point):
    """
        Determine between line segments.
        Args:
            vector1 (Vector2): The first endpoint of the line segment.
            vector2 (Vector2): The second endpoint of the line segment.
            point (Vector2): The point to which the squared distance is to be calculated.
        Returns:
            bool
        """
    vec = vector2 - vector1
    r = np.dot((point - vector1), vec) / np.dot(vec, vec)

    return True if 0.0 <= r <= 1.0 else False


# 无人机到各目标点距离
def dis_UT(pos, tasks_pos):
    agent_pos = np.array([pos[:2]])
    tasks_pos_2d = [p[:2] for p in tasks_pos]
    tasks_pos_2d = np.array(tasks_pos_2d)
    UtoT_matrix = spatial.distance.cdist(agent_pos, tasks_pos_2d, metric='euclidean')
    return UtoT_matrix  # 函数返回距离矩阵


# 各目标点之间的距离
def dis_TT(tasks_pos):
    tasks_pos_2d = [pos[:2] for pos in tasks_pos]
    tasks_pos_2d = np.array(tasks_pos_2d)
    TtoT_matrix = spatial.distance.cdist(tasks_pos_2d, tasks_pos_2d, metric='euclidean')
    return TtoT_matrix


# 各目标点到终止区域的距离
def dis_TE(pos, tasks_pos):
    agent_pos = np.array([pos[:2]])
    tasks_pos_2d = [p[:2] for p in tasks_pos]
    tasks_pos_2d = np.array(tasks_pos_2d)
    TtoE_matrix = spatial.distance.cdist(agent_pos, tasks_pos_2d, metric='euclidean')
    return TtoE_matrix  # 函数返回距离矩阵


def dist_sector(radius):
    L = 2 * pi * radius
    dist_dict = {}
    for i in range(1, 360):
        dist_dict[i] = (i / 360) * L
    return dist_dict


if __name__ == '__main__':
    pass
