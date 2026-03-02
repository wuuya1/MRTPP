"""
Contains functions and constants used in multiple classes.
"""
import math

"""
A sufficiently small positive number.
"""
EPSILON = 0.00001


def dist_point_line_segment(vector1, vector2, point):
    dist_sq = dist_sq_point_line_segment(vector1, vector2, point)
    return math.sqrt(dist_sq)


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
    r = ((vector3 - vector1) @ (vector2 - vector1)) / abs_sq(vector2 - vector1)

    if r < 0.0:
        return abs_sq(vector3 - vector1)

    if r > 1.0:
        return abs_sq(vector3 - vector2)

    return abs_sq(vector3 - (vector1 + r * (vector2 - vector1)))


def cross(p1, p2, p3):  # 跨立实验
    x1 = p2.x - p1.x
    y1 = p2.y - p1.y
    x2 = p3.x - p1.x
    y2 = p3.y - p1.y
    return float(x1 * y2 - x2 * y1)


def seg_is_intersect(p1, p2, p3, p4):  # 判断线段p1p2与线段p3p4是否相交
    # 快速排斥，以l1、l2为对角线的矩形必相交，否则两线段不相交
    if (max(p1.x, p2.x) >= min(p3.x, p4.x)  # 矩形1最右端大于矩形2最左端
            and max(p3.x, p4.x) >= min(p1.x, p2.x)  # 矩形2最右端大于矩形最左端
            and max(p1.y, p2.y) >= min(p3.y, p4.y)  # 矩形1最高端大于矩形最低端
            and max(p3.y, p4.y) >= min(p1.y, p2.y)):  # 矩形2最高端大于矩形最低端

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


def l2norm(p1, p2):
    return math.sqrt(abs_sq(p1-p2))


def norm(vector):
    return math.sqrt(abs_sq(vector))


def abs_sq(vector):
    """
    Computes the squared length of a specified two-dimensional vector.

    Args:
        vector (Vector2): The two-dimensional vector whose squared length is to be computed.

    Returns:
        float: The squared length of the two-dimensional vector.
    """
    return vector @ vector


def normalize(vector):
    """
    Computes the normalization of the specified two-dimensional vector.

    Args:
        vector (Vector2): The two-dimensional vector whose normalization is to be computed.

    Returns:
        Vector2: The normalization of the two-dimensional vector.
    """
    return vector / abs(vector)


def det(vector1, vector2):
    """
    Computes the determinant of a two-dimensional square matrix with rows consisting of the specified two-dimensional vectors.

    Args:
        vector1 (Vector2): The top row of the two-dimensional square matrix.
        vector2 (Vector2): The bottom row of the two-dimensional square matrix.

    Returns:
        float: The determinant of the two-dimensional square matrix.
    """
    return vector1.x_ * vector2.y_ - vector1.y_ * vector2.x_


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
    r = ((vector3 - vector1) @ (vector2 - vector1)) / abs_sq(vector2 - vector1)

    if r < 0.0:
        return abs_sq(vector3 - vector1)

    if r > 1.0:
        return abs_sq(vector3 - vector2)

    return abs_sq(vector3 - (vector1 + r * (vector2 - vector1)))


def left_of(a, b, c):
    """
    Computes the signed distance from a line connecting the specified points to a specified point.

    Args:
        a (Vector2): The first point on the line.
        b (Vector2): The second point on the line.
        c (Vector2): The point to which the signed distance is to be calculated.

    Returns:
        float: Positive when the point c lies to the left of the line ab.
    """
    return det(a - c, b - a)


def square(scalar):
    """
    Computes the square of a float.

    Args:
        scalar (float): The float to be squared.

    Returns:
        float: The square of the float.
    """
    return scalar * scalar
