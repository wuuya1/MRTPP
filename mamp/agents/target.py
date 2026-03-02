import numpy as np
from mamp.tools.vector import Vector2


class Target(object):
    def __init__(self, pos, shape_dict, idx=0, tid='tar'):
        self.is_agent_ = False
        self.is_obstacle_ = False
        self.is_target_ = True

        self.shape = shape = shape_dict['shape']
        self.feature = shape_dict['feature']
        if shape == 'rect':
            self.width, self.heigh, self.rect_pos = shape_dict['feature']
            self.radius = np.sqrt(self.width ** 2 + self.heigh ** 2) / 2
        elif shape == 'circle':
            self.radius = shape_dict['feature']
        else:
            raise NotImplementedError
        self.ta_pos = np.array(pos)
        self.pos_global_frame = np.array(pos[:2], dtype='float64')
        self.goal_global_frame = np.array(pos[:2], dtype='float64')
        self.vel_global_frame = np.array([0.0, 0.0])
        self.pos = pos[:2]
        self.position_ = Vector2(pos[0], pos[1])
        self.id = idx
        self.tid = {self.id: tid}
        self.t = 0.0
        self.step_num = 0
        self.is_close = False
        self.is_at_goal = True
        self.was_in_collision_already = False
        self.is_collision = False
        self.is_poly = False
        self.is_expansion = False

        self.is_visited = False
        self.x = pos[0]
        self.y = pos[1]
        self.r = shape_dict['feature']
