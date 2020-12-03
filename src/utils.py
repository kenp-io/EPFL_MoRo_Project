# ******** IMPORTS ********
import numpy as np
import time
import cv2

import vision
import globalNavigation

FORWARDCONSTANT = 37.95
FULLROTATIONTIME = 8900
A_STAR_Y_AXIS_SIZE = 50

# ******** CLASSES ********

class virtualThymio(object):
    def __init__(self, cap, th):

        self.cap = cap
        self.th = th

        robotCenter = None
        while robotCenter is None:
            frame = self.cap.read()
            robotCenter, _ = vision.find_thymio_center(frame)
        self.pos_x = robotCenter[0]
        self.pos_y = robotCenter[1]

        robotFront = None
        while robotFront is None:
            robotFront, _ = vision.find_thymio_front(frame)
        self.front_x = robotFront[0]
        self.front_y = robotFront[1]

        self.angle = globalNavigation.angleTwoPoints(robotFront,robotCenter)
        self.vel_x = 0.
        self.vel_y = 0.

    def update(self):
        frame = self.cap.read()
        robotCenter, _ = vision.find_thymio_center(frame)
        robotFront, _ = vision.find_thymio_front(frame)

        if robotCenter is not None and robotFront is not None:
            self.pos_x = robotCenter[0]
            self.pos_y = robotCenter[1]
            self.front_x = robotFront[0]
            self.front_y = robotFront[1]
            self.angle = globalNavigation.angleTwoPoints(robotFront,robotCenter)
            # we know that everytime the read function is called the Thymio is
            # moving at around speed 100
            self.vel_x = FORWARDCONSTANT*np.cos(self.angle)
            self.vel_y = FORWARDCONSTANT*np.sin(self.angle)

    def readKalman(self):
        self.update()
        return [self.pos_x,
                self.pos_y,
                self.vel_x,
                self.vel_y]

    def getFront(self):
        return [self.front_x,
                self.front_y]

    def getCenter(self):
        return [self.pos_x,
                self.pos_y]

# ******** FUNCTIONS ********

def analyze(ourThymio):

    #Take a picture
    raw_frame = ourThymio.cap.read()
    frame = raw_frame.copy()

    # Find robot position
    robotCenter = ourThymio.getCenter()

    # Find destination(goal) position
    destinationCenter = None
    while destinationCenter is None:
        destinationCenter, _ = vision.find_destination_center(frame)

    # Detect obstacles positions
    output_objects = vision.find_objects(frame)
    output_objects = cv2.cvtColor(output_objects, cv2.COLOR_BGR2GRAY)

    occupancy_grid, cmap = globalNavigation.display_occupancy_grid(output_objects)

    # Define the start and end goal
    start = (int(robotCenter[0]*0.0694), int(robotCenter[1]*0.0694))
    goal = (int(destinationCenter[0]*0.0694), int(destinationCenter[1]*0.0694))

    #Compute path
    #First value of path is current position
    #Last value of path is destination(goal) position
    path = globalNavigation.runAstar(start, goal, A_STAR_Y_AXIS_SIZE, occupancy_grid, cmap)

    return path
