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
    
    def __init__(self, track):
        Bot.__init__(self, track)
        self.velList = self.compute_section_velocities()
    
    def compute_circle_radius(self, p1, p2, p3):
        
        x1 = p1[0]
        y1 = p1[1]

        x2 = p2[0]
        y2 = p2[1]

        x3 = p3[0]
        y3 = p3[1]

        a = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        b = math.sqrt((x3 - x2)**2 + (y3 - y2)**2)
        c = math.sqrt((x1 - x3)**2 + (y1 - y3)**2)

        area = abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)) / 2
    
        # If the area is zero, the points are collinear and no circle exists
        if area == 0:
            raise ValueError("The three points are collinear, so no circle can be formed.")
        
        # Calculate the circumradius (R)
        radius = (a * b * c) / (4 * area)
        
        return radius
    
    def compute_section_velocities(self):

        nTargets = len(self.track.lines)

        angleList = [0] * nTargets
        velList = [0] * nTargets
        radius = [0] * nTargets

        # Compute all angles
        for section in range(nTargets):

            id1 = section
            id2 = (section - 1) % (nTargets - 1)
            id3 = (section + 2) % (nTargets - 1)

            point1 = self.track.lines[id1]
            point2 = self.track.lines[id2]
            point3 = self.track.lines[id3]

            a = [point1[0] - point2[0], point1[1] - point2[1]]
            b = [point1[0] - point3[0], point1[1] - point3[1]]

            lengthA = math.sqrt(a[0]**2 + a[1]**2)
            lengthB = math.sqrt(b[0]**2 + b[1]**2)

            angleList[section] = math.acos( numpy.dot(a,b) / ( lengthA * lengthB ) )

            velList[section] = ( angleList[section] / 3.1415 )**2 * 400

            # radius[section] = self.compute_circle_radius(point1, point2, point3)
            # velList[section] = radius[section]**1 *3.5

        return velList
    
    def compute_commands(self, next_waypoint: int, position: Transform, velocity: Vector2) -> Tuple:

        # Amount of targets
        nTargets = len(self.track.lines)
        # Find target after target

        targetTrue = self.track.lines[next_waypoint]
 
        # calculate the target in the frame of the robot
        target = position.inverse() * targetTrue

        # calculate the angle to the target
        angle = target.as_polar()[1]

        # Deceleration of car
        acc = -100

        # Current absolute velocity
        absoluteVelocity = math.sqrt(velocity[0]**2 + velocity[1]**2)

        distToTarget = 0

        # Look for (all) sections ahead and check if we need to brake to meet the desired velocity
        for section in range(nTargets):
            if section == 0:
                distToTarget = distToTarget + math.sqrt(target[0]**2 + target[1]**2) - self.track.track_width
            else:
                section1ID = (next_waypoint + section - 1 ) % (nTargets - 1)
                section2ID = (next_waypoint + section ) % (nTargets - 1)

                target1 = self.track.lines[section1ID]
                target2 = self.track.lines[section2ID]

                distToTarget = distToTarget + math.sqrt((target2[0] - target1[0])**2 + (target2[1] - target1[1])**2)

            # Square root term in ABC formula. If this is negative it means we are abele to stand
            bTerm = 2*acc*distToTarget + absoluteVelocity**2

            if bTerm < 0:
                throttle = 1
                break
            else:
                t = (math.sqrt(2*acc*distToTarget + absoluteVelocity**2) - absoluteVelocity)/acc
                vTarget = acc * t + absoluteVelocity
                sectionID = (next_waypoint + section) % (nTargets - 1)
                if vTarget < self.velList[( next_waypoint + section ) % (nTargets - 1)]:
                    throttle = 1
                else:
                    throttle = -1
                    break

        if angle > 0:
            return throttle, 1
        else:
            return throttle, -1