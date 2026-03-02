import math
import mamp.tools.rvo_math as rvo_math


class Vector2:
    """
    Defines a two-dimensional vector.
    """

    def __init__(self, x=0.0, y=0.0):
        """
        Constructs and initializes a two-dimensional vector from the specified xy-coordinates.

        Args:
            x (float): The x-coordinate of the two-dimensional vector.
            y (float):The y-coordinate of the two-dimensional vector.
        """
        self.x_ = x
        self.y_ = y

    def __str__(self):
        return "Vector2(x={}, y={})".format(self.x_, self.y_)

    @property
    def x(self):
        return self.x_

    @property
    def y(self):
        return self.y_

    def __matmul__(self, other):
        assert isinstance(other, Vector2), '__matmul__ argument should be a Vector2'
        return self.x_ * other.x_ + self.y_ * other.y_

    def __mul__(self, other):
        assert not isinstance(other, Vector2), '__mul__ argument should be a float'
        return Vector2(self.x_ * other, self.y_ * other)

    def __rmul__(self, other):
        assert not isinstance(other, Vector2), '__rmul__ argument should be a float'
        return Vector2(other * self.x_, other * self.y_)

    def __truediv__(self, scalar):
        return Vector2(self.x_ / scalar, self.y_ / scalar)

    def __add__(self, other):
        return Vector2(self.x_ + other.x_, self.y_ + other.y_)

    def __radd__(self, other):
        return Vector2(other.x_ + self.x_, other.y_ + self.y_)

    def __sub__(self, other):
        return Vector2(self.x_ - other.x_, self.y_ - other.y_)

    def __rsub__(self, other):
        return Vector2(other.x_ - self.x_, other.y_ - self.y_)

    def __neg__(self):
        return Vector2(-self.x_, -self.y_)

    def __abs__(self):
        """
        Computes the length of a specified two-dimensional vector.

        Args:
            vector (Vector2): The two-dimensional vector whose length is to be computed.

        Returns:
            float: The length of the two-dimensional vector.
        """
        return math.sqrt(rvo_math.abs_sq(self))
