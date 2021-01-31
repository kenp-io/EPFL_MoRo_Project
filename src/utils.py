"""
## utils.py ##

Definition of constants, classes and functions tools used to compute the
global path, define and control our robot.

Imported by: main.ipynb, localNavigation.py, globalNavigation.py.

*** Classes ***
virtualThymio(cap,th)

*** Function ***
analyze(ourThymio, destinationCenter=None)

"""

# ******** IMPORTS ********

import numpy as np
import time
import cv2
from threading import Event

import vision
import globalNavigation

# ******** CONSTANTS ********

FORWARDCONSTANT = 37.95
MOTORSPEED = 100
FULLROTATIONTIME = 8900
A_STAR_Y_AXIS_SIZE = 50
MAXCORRECTION  = 1.05

# ******** CLASSES ********

class virtualThymio(object):
    """
    Class representing our Thymio robot, gathering state information and class
    methods to modifiy its state, command the motors, get Kalman filter values
    to adapt its motor speed

    **Initialisation**
    virtualThymio(cap,th)

    **Functions**
    update()
    readKalman()
    clearKalman()
    getVel()
    getFront()
    getCenter()
    forward()
    antiClockwise()
    clockwise()
    stop()
    correctToRight(ratio)
    correctToLeft(ratio)
    """

    def __init__(self, cap, th):
        """
        Initialisation of the object

        :param cap: capture read with function read() of class vision.VideoCapture
        :param th: instance linking to the Thymio robot connected via USB

        :return: creates the object of class virtualThymio
        """

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
        self.ratioXKalman = 0.
        self.ratioYKalman = 0.

        self.inLocal = False
        self.runningKalman = False
        self.stopKalmanFlag = Event()
        self.reached = False
            #print(f'ini {self.vel_left} , {self.vel_right}')

    def update(self):
        """
        Updates the state of the robot: position of front and center circles, ¨
        Thymio angle and axis velocities
        """
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
        """
        Reads Thymio's position coordinates and axis velocities

        :return: np array of position coordinates and velocities
        """
        self.update()
        return np.array([[self.pos_x],
                         [self.pos_y],
                         [self.vel_x],
                         [self.vel_y]])

    def clearKalman(self):
        """
        Creates a new Stop Event to assign it to a newly created Kalman filter
        """
        self.stopKalmanFlag = Event()
        return

    def getVel(self):
        """
        Reads Thymio's axis velocities

        :return: np array of axis velocities
        """
        return np.array([[self.vel_x],
                         [self.vel_y]])

    def getRatios(self):
            """
            Reads Thymio's Kalman ratios

            :return: np array of ratios applied to the motors to correct
                     the position
            """
            return np.array([[self.ratioXKalman],
                             [self.ratioYKalman]])

    def getFront(self):
        """
        Reads coordinates of Thymio's blue front circle's center

        :return: array of front circle's center coordinates
        """
        return [self.front_x,
                self.front_y]

    def getAngle(self):
            """
            Reads the absolute angle of Thymio

            :return: angle in rad
            """
            return self.angle

    def getCenter(self):
        """
        Reads coordinates of Thymio's green center circle's center

        :return: array of center circle's center coordinates
        """
        return [self.pos_x,
                self.pos_y]

    def forward(self):
        """
        Set the speed of both motors to MOTORSPEED to make the Thymio move forward
        """
        self.vel_left = MOTORSPEED
        self.vel_right = MOTORSPEED
        self.th.set_var("motor.left.target", MOTORSPEED)
        self.th.set_var("motor.right.target", MOTORSPEED)
            #print(f'ini {self.vel_left} , {self.vel_right}')

    def antiClockwise(self):
        """
        Set the speed of the motors to ±MOTORSPEED to make the Thymio turn on himself in anticlockwise direction
        """
        self.vel_left = -MOTORSPEED
        self.vel_right = MOTORSPEED
        self.th.set_var("motor.left.target", 2**16-MOTORSPEED)
        self.th.set_var("motor.right.target", MOTORSPEED)
            #print(f'ini {self.vel_left} , {self.vel_right}')

    def clockwise(self):
        """
        Set the speed of the motors to ±MOTORSPEED to make the Thymio turn on himself in clockwise direction
        """
        self.vel_left = MOTORSPEED
        self.vel_right = -MOTORSPEED
        self.th.set_var("motor.left.target", MOTORSPEED)
        self.th.set_var("motor.right.target", 2**16-MOTORSPEED)
            #print(f'ini {self.vel_left} , {self.vel_right}')

    def stop(self):
        """
        Set the speed of the motors to 0 to make the Thymio stop
        """
        self.vel_left = 0
        self.vel_right = 0
        self.th.set_var("motor.left.target", 0)
        self.th.set_var("motor.right.target", 0)
            #print(f'ini {self.vel_left} , {self.vel_right}')

    def correctToRight(self, ratio):
        """
        Set the speed of the motors to correct Thymio's trajectory to the right
        (with the help of the Kalman filter)

        :param ratio: correcting speed factor
        """
        ratio = ratio + 1
        if ratio > MAXCORRECTION:
            ratio = MAXCORRECTION
        if not self.reached:
            self.vel_left = int(MOTORSPEED*ratio)
            self.vel_right = int(MOTORSPEED/ratio)
                #print(f'moving to right {self.vel_left} , {self.vel_right}')
            self.th.set_var("motor.left.target", self.vel_left)
            self.th.set_var("motor.right.target", self.vel_right)

    def correctToLeft(self, ratio):

        """
        Set the speed of the motors to correct Thymio's trajectory to the left
        (with the help of the Kalman filter)

        :param ratio: correcting speed factor
        """
        ratio = ratio + 1
        if ratio > MAXCORRECTION:
            ratio = MAXCORRECTION
        if not self.reached:
            self.vel_left = int(MOTORSPEED/ratio)
            self.vel_right = int(MOTORSPEED*ratio)
                #print(f'moving to left {self.vel_left} , {self.vel_right}')
            self.th.set_var("motor.left.target", self.vel_left)
            self.th.set_var("motor.right.target", self.vel_right)

# ******** FUNCTIONS ********

def analyze(ourThymio, destinationCenter=None):
    """
    Computes the global navigation path to follow to reach destination:
    -Reads frame
    -Finds Thymio, destination center (red circle) and the objects to avoid
    -Compute the occupancy grid map and the path by calling the Astar algorithm

    :param ourThymio: object of class virtualThymio representing our robot,
                      gathering state information and class methods
    :param destinationCenter: center of the red circle destination if already
                              computed (if function runned after going in local
                              obstacle avoidance mode)

    :return: Computed global path and destination center
    """

    #Take a picture
    raw_frame = ourThymio.cap.read()
    frame = raw_frame.copy()

    # Find robot position
    robotCenter = ourThymio.getCenter()

    # Find destination(goal) position
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

    return path, destinationCenter
