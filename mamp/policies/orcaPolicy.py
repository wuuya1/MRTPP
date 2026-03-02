"""
@ ORCA (Optimal Reciprocal Collision Avoidance) policy in 2D environments
@ Author: Gang Xu
@ Date: 2023.7.10
@ Function:
"""
import math
import numpy as np
import mamp.tools.rvo_math as rvo_math

from mamp.tools.vector import Vector2
from draw.plt2d import plot_half_planes
from mamp.tools.utils import l2norm, seg_cross_circle, dist_point_line_segment, determin_between_line, seg_is_intersec


class Line(object):
    def __init__(self):
        self.direction = Vector2()    # The direction of the directed line.
        self.point = Vector2()        # A point on the directed line.


class ORCAPolicy(object):
    def __init__(self):
        self.str = "ORCAPolicy"
        self.type = "internal"
        self.epsilon = 1e-5
        self.orca_lines_ = []
        self.new_velocity_ = Vector2()

    def find_next_action(self, agent):
        self.orca_lines_ = []

        invTimeHorizonObst = 1.0 / agent.time_horizon_obst_
        agent_rad = agent.radius_ + 2 * agent.planner_.inflation

        # Create obstacle ORCA lines.
        for i in range(len(agent.obstacle_neighbors_)):
            obstacle1 = agent.obstacle_neighbors_[i][1]
            obstacle2 = obstacle1.next_

            relativePosition1 = obstacle1.point_ - agent.position_
            relativePosition2 = obstacle2.point_ - agent.position_

            # Check if velocity obstacle of obstacle is already taken care of
            # by previously constructed obstacle ORCA lines.
            alreadyCovered = False

            for j in range(len(self.orca_lines_)):
                det1 = rvo_math.det(invTimeHorizonObst * relativePosition1 - self.orca_lines_[j].point,
                                    self.orca_lines_[j].direction)
                det2 = rvo_math.det(invTimeHorizonObst * relativePosition2 - self.orca_lines_[j].point,
                                    self.orca_lines_[j].direction)
                if (det1 - invTimeHorizonObst * agent_rad >= -rvo_math.EPSILON) and (
                        det2 - invTimeHorizonObst * agent_rad >= -rvo_math.EPSILON):
                    alreadyCovered = True
                    break

            if alreadyCovered:
                continue

            # Not yet covered. Check for collisions.
            distSq1 = rvo_math.abs_sq(relativePosition1)
            distSq2 = rvo_math.abs_sq(relativePosition2)

            radiusSq = rvo_math.square(agent_rad)

            obstacleVector = obstacle2.point_ - obstacle1.point_
            s = (-relativePosition1 @ obstacleVector) / rvo_math.abs_sq(obstacleVector)
            distSqLine = rvo_math.abs_sq(-relativePosition1 - s * obstacleVector)

            line = Line()

            if s < 0.0 and distSq1 <= radiusSq:
                # Collision with left vertex. Ignore if non-convex.
                if obstacle1.convex_:
                    line.point = Vector2(0.0, 0.0)
                    line.direction = rvo_math.normalize(Vector2(-relativePosition1.y, relativePosition1.x))
                    self.orca_lines_.append(line)
                continue
            elif s > 1.0 and distSq2 <= radiusSq:
                # Collision with right vertex. Ignore if non-convex or if it will be taken care of
                # by neighboring obstacle.
                if obstacle2.convex_ and rvo_math.det(relativePosition2, obstacle2.direction_) >= 0.0:
                    line.point = Vector2(0.0, 0.0)
                    line.direction = rvo_math.normalize(Vector2(-relativePosition2.y, relativePosition2.x))
                    self.orca_lines_.append(line)
                continue
            elif 0.0 <= s < 1.0 and distSqLine <= radiusSq:
                # Collision with obstacle segment.
                line.point = Vector2(0.0, 0.0)
                line.direction = -obstacle1.direction_
                self.orca_lines_.append(line)
                continue

            # No collision. Compute legs. When obliquely viewed, both legs can come from a single vertex.
            # Legs extend cut-off line when non-convex vertex.
            leftLegDirection = None
            rightLegDirection = None

            if s < 0.0 and distSqLine <= radiusSq:
                # Obstacle viewed obliquely so that left vertex defines velocity obstacle.
                if not obstacle1.convex_:
                    # Ignore obstacle.
                    continue

                obstacle2 = obstacle1

                leg1 = math.sqrt(distSq1 - radiusSq)
                leftLegDirection = Vector2(relativePosition1.x * leg1 - relativePosition1.y * agent_rad,
                                           relativePosition1.x * agent_rad + relativePosition1.y * leg1) / distSq1
                rightLegDirection = Vector2(relativePosition1.x * leg1 + relativePosition1.y * agent_rad,
                                            -relativePosition1.x * agent_rad + relativePosition1.y * leg1) / distSq1
            elif s > 1.0 and distSqLine <= radiusSq:
                # Obstacle viewed obliquely so that right vertex defines velocity obstacle.
                if not obstacle2.convex_:
                    # Ignore obstacle.
                    continue

                obstacle1 = obstacle2

                leg2 = math.sqrt(distSq2 - radiusSq)
                leftLegDirection = Vector2(relativePosition2.x * leg2 - relativePosition2.y * agent_rad,
                                           relativePosition2.x * agent_rad + relativePosition2.y * leg2) / distSq2
                rightLegDirection = Vector2(relativePosition2.x * leg2 + relativePosition2.y * agent_rad,
                                            -relativePosition2.x * agent_rad + relativePosition2.y * leg2) / distSq2
            else:
                # Usual situation.
                if obstacle1.convex_:
                    leg1 = math.sqrt(distSq1 - radiusSq)
                    leftLegDirection = Vector2(relativePosition1.x * leg1 - relativePosition1.y * agent_rad,
                                               relativePosition1.x * agent_rad + relativePosition1.y * leg1) / distSq1
                else:
                    # Left vertex non-convex left leg extends cut-off line.
                    leftLegDirection = -obstacle1.direction_

                if obstacle2.convex_:
                    leg2 = math.sqrt(distSq2 - radiusSq)
                    rightLegDirection = Vector2(relativePosition2.x * leg2 + relativePosition2.y * agent_rad,
                                                -relativePosition2.x * agent_rad + relativePosition2.y * leg2) / distSq2
                else:
                    # Right vertex non-convex right leg extends cut-off line.
                    rightLegDirection = obstacle1.direction_

            # Legs can never point into neighboring edge when convex vertex, take cutoff-line of
            # neighboring edge instead. If velocity projected on "foreign" leg, no constraint is added.

            leftNeighbor = obstacle1.previous_

            isLeftLegForeign = False
            isRightLegForeign = False

            if obstacle1.convex_ and rvo_math.det(leftLegDirection, -leftNeighbor.direction_) >= 0.0:
                # Left leg points into obstacle.
                leftLegDirection = -leftNeighbor.direction_
                isLeftLegForeign = True

            if obstacle2.convex_ and rvo_math.det(rightLegDirection, obstacle2.direction_) <= 0.0:
                # Right leg points into obstacle.
                rightLegDirection = obstacle2.direction_
                isRightLegForeign = True

            # Compute cut-off centers.
            leftCutOff = invTimeHorizonObst * (obstacle1.point_ - agent.position_)
            rightCutOff = invTimeHorizonObst * (obstacle2.point_ - agent.position_)
            cutOffVector = rightCutOff - leftCutOff

            # Project current velocity on velocity obstacle.

            # Check if current velocity is projected on cutoff circles.
            t = 0.5 if obstacle1 == obstacle2 else ((agent.velocity_ - leftCutOff) @ cutOffVector) / rvo_math.abs_sq(
                cutOffVector)
            tLeft = (agent.velocity_ - leftCutOff) @ leftLegDirection
            tRight = (agent.velocity_ - rightCutOff) @ rightLegDirection

            if (t < 0.0 and tLeft < 0.0) or (obstacle1 == obstacle2 and tLeft < 0.0 and tRight < 0.0):
                # Project on left cut-off circle.
                unitW = rvo_math.normalize(agent.velocity_ - leftCutOff)
                line.direction = Vector2(unitW.y, -unitW.x)
                line.point = leftCutOff + agent_rad * invTimeHorizonObst * unitW
                self.orca_lines_.append(line)
                continue

            elif t > 1.0 and tRight < 0.0:
                # Project on right cut-off circle.
                unitW = rvo_math.normalize(agent.velocity_ - rightCutOff)
                line.direction = Vector2(unitW.y, -unitW.x)
                line.point = rightCutOff + agent_rad * invTimeHorizonObst * unitW
                self.orca_lines_.append(line)
                continue

            # Project on left leg, right leg, or cut-off line, whichever is closest to velocity.
            distSqCutoff = math.inf if t < 0.0 or t > 1.0 or obstacle1 == obstacle2 else rvo_math.abs_sq(
                agent.velocity_ - (leftCutOff + t * cutOffVector))
            distSqLeft = math.inf if tLeft < 0.0 else rvo_math.abs_sq(
                agent.velocity_ - (leftCutOff + tLeft * leftLegDirection))
            distSqRight = math.inf if tRight < 0.0 else rvo_math.abs_sq(
                agent.velocity_ - (rightCutOff + tRight * rightLegDirection))

            if distSqCutoff <= distSqLeft and distSqCutoff <= distSqRight:
                # Project on cut-off line.
                line.direction = -obstacle1.direction_
                line.point = leftCutOff + agent_rad * invTimeHorizonObst * Vector2(-line.direction.y, line.direction.x)
                self.orca_lines_.append(line)
                continue

            if distSqLeft <= distSqRight:
                # Project on left leg.
                if isLeftLegForeign:
                    continue

                line.direction = leftLegDirection
                line.point = leftCutOff + agent_rad * invTimeHorizonObst * Vector2(-line.direction.y, line.direction.x)
                self.orca_lines_.append(line)

                continue

            # Project on right leg.
            if isRightLegForeign:
                continue

            line.direction = -rightLegDirection
            line.point = rightCutOff + agent_rad * invTimeHorizonObst * Vector2(-line.direction.y, line.direction.x)
            self.orca_lines_.append(line)

        numObstLines = len(self.orca_lines_)

        invTimeHorizon = 1.0 / agent.time_horizon_

        # Create agent ORCA lines.
        agent_rad = agent.radius_ + agent.planner_.inflation
        for i in range(len(agent.agent_neighbors_)):
            other = agent.agent_neighbors_[i][1]
            other_rad = other.radius_ + agent.planner_.inflation
            combinedRadius = agent_rad + other_rad
            agent_pos = np.array([agent.position_.x, agent.position_.y])
            other_pos = np.array([other.position_.x, other.position_.y])
            agent_goal = np.array(agent.now_goal_[:2])
            other_goal = np.array(other.now_goal_[:2])
            # c_dist1 = dist_point_line_segment(agent_pos, agent_goal, other_goal) > combinedRadius
            # c_dist2 = dist_point_line_segment(other_pos, other_goal, agent_goal) > combinedRadius
            # c_intersect = not seg_is_intersec(agent_pos, agent_goal, other_pos, other_goal)
            # if c_dist1 and c_dist2 and c_intersect:
            #     continue
            relativePosition = other.position_ - agent.position_
            relativeVelocity = agent.velocity_ - other.velocity_

            distSq = rvo_math.abs_sq(relativePosition)
            combinedRadiusSq = rvo_math.square(combinedRadius)

            line = Line()
            u = Vector2()

            if distSq > combinedRadiusSq:
                # No collision.
                w = relativeVelocity - invTimeHorizon * relativePosition

                # Vector from cutoff center to relative velocity.
                wLengthSq = rvo_math.abs_sq(w)
                dotProduct1 = w @ relativePosition

                if dotProduct1 < 0.0 and rvo_math.square(dotProduct1) > combinedRadiusSq * wLengthSq:
                    # Project on cut-off circle.
                    wLength = math.sqrt(wLengthSq)
                    unitW = w / wLength

                    line.direction = Vector2(unitW.y, -unitW.x)
                    u = (combinedRadius * invTimeHorizon - wLength) * unitW
                else:
                    # Project on legs.
                    leg = math.sqrt(distSq - combinedRadiusSq)

                    if rvo_math.det(relativePosition, w) > 0.0:
                        # Project on left leg.
                        line.direction = Vector2(relativePosition.x * leg - relativePosition.y * combinedRadius,
                                                 relativePosition.x * combinedRadius + relativePosition.y * leg) / distSq
                    else:
                        # Project on right leg.
                        line.direction = -Vector2(relativePosition.x * leg + relativePosition.y * combinedRadius,
                                                  -relativePosition.x * combinedRadius + relativePosition.y * leg) / distSq

                    dotProduct2 = relativeVelocity @ line.direction
                    u = dotProduct2 * line.direction - relativeVelocity
            else:
                # Collision. Project on cut-off circle of time timeStep.
                invTimeStep = 1.0 / agent.time_step_

                # Vector from cutoff center to relative velocity.
                w = relativeVelocity - invTimeStep * relativePosition

                wLength = abs(w)
                unitW = w / wLength

                line.direction = Vector2(unitW.y, -unitW.x)
                u = (combinedRadius * invTimeStep - wLength) * unitW

            line.point = agent.velocity_ + 0.5 * u

            self.orca_lines_.append(line)

        lineFail, self.new_velocity_ = self.linear_program2(self.orca_lines_, agent.max_speed_, agent.pref_velocity_,
                                                            False, self.new_velocity_)

        if lineFail < len(self.orca_lines_):        # empty feasible velocity set
            self.new_velocity_ = self.linear_program3(self.orca_lines_, numObstLines, lineFail, agent.max_speed_,
                                                      self.new_velocity_)
        if agent.id_ == 1 and agent.step_num >= 773:
            print('pref_velocity: ', agent.pref_velocity_.x, agent.pref_velocity_.y)
            print('new_velocity: ', self.new_velocity_.x, self.new_velocity_.y)
        return self.new_velocity_

    def linear_program1(self, lines, lineNo, radius, optVelocity, directionOpt):
        """
        Solves a one-dimensional linear program on a specified line subject to linear constraints defined
        by lines and a circular constraint.

        Args:
            lines (list): Lines defining the linear constraints.
            lineNo (int): The specified line constraint.
            radius (float): The radius of the circular constraint.
            optVelocity (Vector2): The optimization velocity.
            directionOpt (bool): True if the direction should be optimized.

        Returns:
            bool: True if successful.
            Vector2: A reference to the result of the linear program.
        """
        dotProduct = lines[lineNo].point @ lines[lineNo].direction
        discriminant = rvo_math.square(dotProduct) + rvo_math.square(radius) - rvo_math.abs_sq(lines[lineNo].point)

        if discriminant < 0.0:
            # Max speed circle fully invalidates line lineNo.
            return False, None

        sqrtDiscriminant = math.sqrt(discriminant)
        tLeft = -dotProduct - sqrtDiscriminant
        tRight = -dotProduct + sqrtDiscriminant

        for i in range(lineNo):
            denominator = rvo_math.det(lines[lineNo].direction, lines[i].direction)
            numerator = rvo_math.det(lines[i].direction, lines[lineNo].point - lines[i].point)

            if abs(denominator) <= rvo_math.EPSILON:
                # Lines lineNo and i are (almost) parallel.
                if numerator < 0.0:
                    return False, None
                continue

            t = numerator / denominator

            if denominator >= 0.0:
                # Line i bounds line lineNo on the right.
                tRight = min(tRight, t)
            else:
                # Line i bounds line lineNo on the left.
                tLeft = max(tLeft, t)

            if tLeft > tRight:
                return False, None

        if directionOpt:
            # Optimize direction.
            if optVelocity @ lines[lineNo].direction > 0.0:
                # Take right extreme.
                result = lines[lineNo].point + tRight * lines[lineNo].direction
            else:
                # Take left extreme.
                result = lines[lineNo].point + tLeft * lines[lineNo].direction
        else:
            # Optimize closest point.
            t = lines[lineNo].direction @ (optVelocity - lines[lineNo].point)

            if t < tLeft:
                result = lines[lineNo].point + tLeft * lines[lineNo].direction
            elif t > tRight:
                result = lines[lineNo].point + tRight * lines[lineNo].direction
            else:
                result = lines[lineNo].point + t * lines[lineNo].direction

        return True, result

    def linear_program2(self, lines, radius, optVelocity, directionOpt, result):
        """
        Solves a two-dimensional linear program subject to linear constraints defined by lines and a circular constraint.

        Args:
            lines (list): Lines defining the linear constraints.
            radius (float): The radius of the circular constraint.
            optVelocity (Vector2): The optimization velocity.
            directionOpt (bool): True if the direction should be optimized.
            result (Vector2): A reference to the result of the linear program.

        Returns:
            int: The number of the line it fails on, and the number of lines if successful.
            Vector2: A reference to the result of the linear program.
        """
        if directionOpt:
            # Optimize direction. Note that the optimization velocity is of unit length in this case.
            result = optVelocity * radius
        elif rvo_math.abs_sq(optVelocity) > rvo_math.square(radius):
            # Optimize closest point and outside circle.
            result = rvo_math.normalize(optVelocity) * radius
        else:
            # Optimize closest point and inside circle.
            result = optVelocity

        for i in range(len(lines)):
            if rvo_math.det(lines[i].direction, lines[i].point - result) > 0.0:
                # Result does not satisfy constraint i. Compute new optimal result.
                tempResult = result
                success, result = self.linear_program1(lines, i, radius, optVelocity, directionOpt)
                if not success:
                    result = tempResult
                    return i, result

        return len(lines), result

    def linear_program3(self, lines, numObstLines, beginLine, radius, result):
        """
        Solves a two-dimensional linear program subject to linear constraints defined by lines and a circular
        constraint.

        Args:
            lines (list): Lines defining the linear constraints.
            numObstLines (int): Count of obstacle lines.
            beginLine (int): The line on which the 2-d linear program failed.
            radius (float): The radius of the circular constraint.
            result (Vector2): A reference to the result of the linear program.

        Returns:
            Vector2: A reference to the result of the linear program.
        """
        distance = 0.0

        for i in range(beginLine, len(lines)):
            if rvo_math.det(lines[i].direction, lines[i].point - result) > distance:
                # Result does not satisfy constraint of line i.
                projLines = []

                for ii in range(numObstLines):
                    projLines.append(lines[ii])

                for j in range(numObstLines, i):
                    line = Line()
                    determinant = rvo_math.det(lines[i].direction, lines[j].direction)

                    if abs(determinant) <= rvo_math.EPSILON:
                        # Line i and line j are parallel.
                        if lines[i].direction @ lines[j].direction > 0.0:
                            # Line i and line j point in the same direction.
                            continue
                        else:
                            # Line i and line j point in opposite direction.
                            line.point = 0.5 * (lines[i].point + lines[j].point)
                    else:
                        t = rvo_math.det(lines[j].direction, lines[i].point - lines[j].point) / determinant
                        line.point = lines[i].point + t * lines[i].direction

                    line.direction = rvo_math.normalize(lines[j].direction - lines[i].direction)
                    projLines.append(line)

                tempResult = result
                optVelocity = Vector2(-lines[i].direction.y, lines[i].direction.x)
                lineFail, result = self.linear_program2(projLines, radius, optVelocity, True, result)
                if lineFail < len(projLines):
                    """
                    This should in principle not happen. The result is by definition already in the feasible region of 
                    this linear program. If it fails, it is due to small floating point error, and the current result 
                    is kept.
                    """
                    result = tempResult

                distance = rvo_math.det(lines[i].direction, lines[i].point - result)
        return result


if __name__ == '__main__':
    pass
