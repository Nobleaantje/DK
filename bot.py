from typing import Tuple

from pygame import Vector2

from ...bot import Bot
from ...linear_math import Transform

import math
import numpy

class DK(Bot):
    @property
    def name(self):
        return "DK"

    @property
    def contributor(self):
        return "Jerrel"

    def compute_commands(self, next_waypoint: int, position: Transform, velocity: Vector2) -> Tuple:

        # Amount of targets
        nTargets = len(self.track.lines)
        # Find target after target
        next_next_waypoint = (next_waypoint + 2) % nTargets - 1

        targetTrue = self.track.lines[next_waypoint]
        nextTarget = self.track.lines[next_next_waypoint]
        
        # nextTarget = self.track.lines[next_waypoint+1]
        # calculate the target in the frame of the robot
        target = position.inverse() * targetTrue

        # calculate the angle to the target
        angle = target.as_polar()[1]

        a = [targetTrue[0] - nextTarget[0], targetTrue[1] - nextTarget[1]]
        b = [targetTrue[0] - position.p[0], targetTrue[1] - position.p[1]]

        lengthA = math.sqrt(a[0]**2 + a[1]**2)
        lengthB = math.sqrt(b[0]**2 + b[1]**2)

        angle2 = math.acos( numpy.dot(a,b) / ( lengthA * lengthB ) )

        # print(angle2)

        # calculate the throttle
        target_waypoint_velocity = ( angle2 / 3.1415 )**2 * 300
        acc = -100

        distToTarget = math.sqrt(target[0]**2 + target[1]**2) - self.track.track_width
        absoluteVelocity = math.sqrt(velocity[0]**2 + velocity[1]**2)

        bTerm = 2*acc*distToTarget + absoluteVelocity**2
        # print(absoluteVelocity)

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
