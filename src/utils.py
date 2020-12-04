# ******** IMPORTS ********
import numpy as np
import time
import cv2
from threading import Event


import vision
import globalNavigation

FORWARDCONSTANT = 37.95
MOTORSPEED = 100
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
        self.vel_left = 0.
        self.vel_right = 0.

        self.inLocal = False
        self.runningKalman = False
        self.stopKalmanFlag = Event()

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
        return np.array([[self.pos_x],
                         [self.pos_y],
                         [self.vel_x],
                         [self.vel_y]])

    def getVel(self):
        return np.array([[self.vel_x],
                         [self.vel_y]])

    def getFront(self):
        return [self.front_x,
                self.front_y]

    def getCenter(self):
        return [self.pos_x,
                self.pos_y]

    def forward(self):
        self.vel_left = MOTORSPEED
        self.vel_right = MOTORSPEED
        self.th.set_var("motor.left.target", MOTORSPEED)
        self.th.set_var("motor.right.target", MOTORSPEED)

    def antiClockwise(self):
        self.vel_left = -MOTORSPEED
        self.vel_right = MOTORSPEED
        self.th.set_var("motor.left.target", 2**16-MOTORSPEED)
        self.th.set_var("motor.right.target", MOTORSPEED)

    def clockwise(self):
        self.vel_left = MOTORSPEED
        self.vel_right = -MOTORSPEED
        self.th.set_var("motor.left.target", MOTORSPEED)
        self.th.set_var("motor.right.target", 2**16-MOTORSPEED)

    def stop(self):
        self.vel_left = 0.
        self.vel_right = 0.
        self.th.set_var("motor.left.target", 0)
        self.th.set_var("motor.right.target", 0)

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
