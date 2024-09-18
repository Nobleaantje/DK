from typing import Tuple

from pygame import Vector2

from ...bot import Bot
from ...linear_math import Transform

import math


class DK(Bot):
    @property
    def name(self):
        return "DK"

    @property
    def contributor(self):
        return "Nobleo"

    def compute_commands(self, next_waypoint: int, position: Transform, velocity: Vector2) -> Tuple:
        target = self.track.lines[next_waypoint]
        # calculate the target in the frame of the robot
        target = position.inverse() * target

        # calculate the angle to the target
        angle = target.as_polar()[1]

        # calculate the throttle
        target_waypoint_velocity = 150
        acc = -100

        distToTarget = math.sqrt(target[0]**2 + target[1]**2)
        absoluteVelocity = math.sqrt(velocity[0]**2 + velocity[1]**2)

        bTerm = 2*acc*distToTarget + absoluteVelocity**2
        print(absoluteVelocity)

        if bTerm < 0:
            throttle = 1
        else:
            t = (math.sqrt(2*acc*distToTarget + absoluteVelocity**2) - absoluteVelocity)/acc
            vTarget = acc * t + absoluteVelocity
            if vTarget < target_waypoint_velocity:
                throttle = 1
            else:
                throttle = -1

        # calculate the steering
        if angle > 0:
            return throttle, 1
        else:
            return throttle, -1
