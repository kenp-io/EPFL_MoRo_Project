# ******** IMPORTS ********
import numpy as np
import time
import cv2

import vision
import globalNavigation

FORWARDCONSTANT = 37.95
FULLROTATIONTIME = 8700
A_STAR_Y_AXIS_SIZE = 50

# ******** CLASSES ********

class virtualThymio(object):
    def __init__(self, cap, th):

        self.cap = cap
        self.th = th
        
        robot_center_absolute = None
        while robot_center_absolute is None:
            frame = self.cap.read()
            robot_center_absolute, _ = vision.find_thymio_center(frame)
        self.pos_x = robot_center_absolute[0]
        self.pos_y = robot_center_absolute[1]
        self.vel_x = 0.
        self.vel_y = 0.
    
        robot_front_absolute = None
        while robot_front_absolute is None:
            robot_front_absolute, _ = vision.find_thymio_center(frame)
        self.front_x = robot_front_absolute[0]
        self.front_y = robot_front_absolute[1]

    def update(self):
        frame = self.cap.read()
        robot_center_absolute, _ = vision.find_thymio_center(frame)
        robot_front_absolute, _ = vision.find_thymio_front(frame)

        if robot_center_absolute is not None and robot_front_absolute is not None:
            self.pos_x = robot_center_absolute[0]
            self.pos_y = robot_center_absolute[1]
            self.front_x = robot_front_absolute[0]
            self.front_y = robot_front_absolute[1]
            angle = globalNavigation.angleThymioCalculator(robot_front_absolute, robot_center_absolute)
            # we know that everytime the read function is called the Thymio is
            # moving at around speed 100
            self.vel_x = FORWARDCONSTANT*np.cos(angle)
            self.vel_y = FORWARDCONSTANT*np.sin(angle)

    def getPosVel(self):
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

    def cap(self):
        return self.cap
    
    def th(self):
        return self.th

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