import numpy as np
from shapely.geometry import Polygon
from mamp.tools.utils import normalize, left_of, min_enclosing_circle


class Vertex(object):
    """
    Defines vertice of polygonal static obstacles in the simulation.
    """

    def __init__(self, point=None):
        self.next_ = None  # 下一顶点
        self.previous_ = None  # 上一顶点
        self.direction_ = None  # 顶点的连接方向
        self.point_ = point  # 当前顶点
        self.id_ = -1
        self.convex_ = False
        self.ob_id = -1  # 顶点组成的障碍物的id
        self.is_disrupted = False
        self.is_checked = False


class Obstacle(object):
    def __init__(self, shape_dict, idx, tid, vertices=None, pos=None):
        self.is_poly = True
        self.shape = shape_dict['shape']
        if self.shape == 'rect':
            self.width, self.height = vertices[2] - vertices[0]
        elif self.shape == 'polygon':
            self.width, self.height = -1, -1
        self.id = idx
        self.tid = {self.id: tid}
        self.pos, self.radius = min_enclosing_circle(vertices)
        self.pos_global_frame = self.pos if pos is None else np.array(pos[:2], dtype='float64')
        self.goal_global_frame = self.pos if pos is None else np.array(pos[:2], dtype='float64')
        self.vel_global_frame = np.array([0.0, 0.0])
        # self.pos = None if pos is None else pos[:2]
        self.obstacle_dict = dict()

        self.is_agent = False
        self.is_at_goal = True
        self.is_obstacle = True
        self.is_danger_and_avoided = False  # 如果是True，则后续相交检测的时候不再考虑该障碍物
        self.is_collision = False
        self.was_in_collision_already = False
        self.unreachable_vertices = []

        self.t = 0.0
        self.step_num = 0

        self.vertices_ = []
        self.vertices_pos = []
        self.connect_vertices(vertices)
        self.outline_ = Polygon(self.vertices_pos)

    def connect_vertices(self, vertices):
        """
        build a new polygonal obstacle to the simulation.

        Args:
            vertices (list): List of the vertices of the polygonal obstacle in counterclockwise order.

        Returns:
            int: The number of the first vertex of the obstacle, or -1 when the number of vertices is less than two.

        Remarks:
            To add a "negative" obstacle, e.g. a bounding polygon around the environment, the vertices should be
            listed in clockwise order.
        """
        if len(vertices) < 2:
            raise Exception('Must have at least 2 vertices.')

        for i in range(len(vertices)):
            vertex = Vertex()
            vertex.point_ = np.array(vertices[i])

            if i != 0:
                vertex.previous_ = self.vertices_[len(self.vertices_) - 1]
                vertex.previous_.next_ = vertex

            if i == len(vertices) - 1:
                vertex.next_ = self.vertices_[0]
                vertex.next_.previous_ = vertex

            vertex.direction_ = normalize(vertices[0 if i == len(vertices) - 1 else i + 1] - vertices[i])

            if len(vertices) == 2:
                vertex.convex_ = True
            else:
                vertex.convex_ = left_of(vertices[len(vertices) - 1 if i == 0 else i - 1], vertices[i],
                                         vertices[0 if i == len(vertices) - 1 else i + 1]) >= 0.0

            vertex.id_ = i
            vertex.ob_id = self.id
            self.vertices_.append(vertex)
            self.vertices_pos.append(list(vertex.point_))
            self.obstacle_dict[tuple(vertex.point_)] = (self.id, i)
