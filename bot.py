from typing import Tuple

from pygame import Vector2
import pygame

from ...bot import Bot
from ...linear_math import Transform

from copy import deepcopy

import math
import numpy

import numpy as np

DEBUG = True

class DK(Bot):
    @property
    def name(self):
        return "DK"

    @property
    def contributor(self):
        return "Jerrel"
    
    def __init__(self, track):
        Bot.__init__(self, track)
        self.velList, self.angleList = self.compute_section_velocities()
        self._font = pygame.font.SysFont(None, 24)
        self.optimSetpoints = self.compute_optimal_raceline(self.track.lines, 15)
        self.tempAngleList = [0]
        self.tempVelList = [0]
    
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
    
    def convert_angle_to_speed(self,angle):

        angleInterp = [0, 0.5, 1  , 1.5, 2  , 2.5, 3  , 3.1415]
        velInterp =   [0,  50, 110, 130, 180, 260, 410, 999   ]

        return numpy.interp(angle,angleInterp,velInterp)
    
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

            # velList[section] = ( angleList[section] / 3.1415 )**2 * 400

            velList[section] = self.convert_angle_to_speed(angleList[section])

            # radius[section] = self.compute_circle_radius(point1, point2, point3)
            # velList[section] = radius[section]**1 *3.5

        return velList, angleList
    
    def update_next_vel_target(self, next_waypoint):

        nTargets = len(self.track.lines)
        id1 = next_waypoint
        # id2 = (next_waypoint - 1) % (nTargets - 1)
        id3 = (next_waypoint + 2) % (nTargets - 1)

        point1 = self.track.lines[id1]
        point2 = self.curPos
        point3 = self.track.lines[id3]

        a = [point1[0] - point2[0], point1[1] - point2[1]]
        b = [point1[0] - point3[0], point1[1] - point3[1]]

        lengthA = math.sqrt(a[0]**2 + a[1]**2)
        lengthB = math.sqrt(b[0]**2 + b[1]**2)

        self.tempAngleList = deepcopy(self.angleList)
        self.tempVelList   = deepcopy(self.velList)

        self.tempAngleList[next_waypoint] = math.acos( numpy.dot(a,b) / ( lengthA * lengthB ) )
        self.tempVelList[next_waypoint]   = self.convert_angle_to_speed(self.tempAngleList[next_waypoint])

        # velList[section] = ( angleList[section] / 3.1415 )**2 * 400

        # angleInterp = [0, 1.62, 2.32, 2.5 , 3.1415]
        # velInterp =   [0, 150,  250,  300 , 999]

    def compute_optimal_raceline(self, checkpoints, r):
        """
        Computes the optimal raceline through a set of checkpoints.
        
        Parameters:
        - checkpoints: List of tuples (x, y) representing the coordinates of the middle of the track.
        - r: The width of the track (distance from the center to the edge).
        
        Returns:
        - optimal_path: List of (x, y) coordinates representing the optimal raceline.
        """
        n = len(checkpoints)
        
        if n == 0:
            return []
        
        # Positions for edges of the track
        edge_positions = []
        for (x, y) in checkpoints:
            edge_positions.append([(x, y + r), (x, y - r)])  # Upper and Lower edges

        # Initialize the DP table
        dp = np.inf * np.ones((n, 2))  # Two edges per checkpoint
        dp[0, 0] = 0  # Starting from the upper edge of the first checkpoint (outer edge)
        dp[0, 1] = np.inf  # We will not consider starting from the lower edge

        # To reconstruct the path later
        path = np.zeros((n, 2), dtype=int)

        # Dynamic programming to find minimum distances
        for i in range(1, n):
            for j in range(2):  # Previous edge (0: upper, 1: lower)
                for k in range(2):  # Current edge
                    distance = np.linalg.norm(np.array(edge_positions[i][k]) - np.array(edge_positions[i - 1][j]))
                    if dp[i - 1, j] + distance < dp[i, k]:
                        dp[i, k] = dp[i - 1, j] + distance
                        path[i, k] = j  # Store the previous edge index

        # Always end at the upper edge of the last checkpoint
        last_edge = 0  # Upper edge of the last checkpoint

        # Backtrack to find the optimal path
        optimal_path = []
        for i in range(n - 1, -1, -1):
            optimal_path.append(edge_positions[i][last_edge])
            last_edge = path[i, last_edge]
        
        optimal_path.reverse()  # Reverse to get the correct order

        return optimal_path

    
    def draw(self, map_scaled, zoom):
        if DEBUG:
            for angle, vel, coords in zip(self.tempAngleList, self.tempVelList, self.track.lines):
                color = pygame.Color(0, 0, 0, 50)
                textAngle  = self._font.render(f'{angle:.2f}', True, color)
                textVel    = self._font.render(f'{vel:.2f}', True, color)
                color = pygame.Color(255, 0, 0, 50)
                textCurVel = self._font.render(f'{self.AbsVel:.2f}', True, color)
                map_scaled.blit(textAngle, (coords * zoom) - pygame.Vector2(25, 25))
                map_scaled.blit(textVel, (coords * zoom) - pygame.Vector2(25, 10))
                map_scaled.blit(textCurVel, (self.curPos * zoom) - pygame.Vector2(25, 25))
    
    def compute_commands(self, next_waypoint: int, position: Transform, velocity: Vector2) -> Tuple:

        self.track.lines = [Vector2(b) for b in self.optimSetpoints]

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
        self.AbsVel = absoluteVelocity
        self.curPos = position.p

        distToTarget = 0

        self.update_next_vel_target(next_waypoint)

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
                if vTarget < self.tempVelList[( next_waypoint + section ) % (nTargets - 1)]:
                    throttle = 1
                else:
                    throttle = -1
                    break

        if angle > 0:
            return throttle, 1
        else:
            return throttle, -1