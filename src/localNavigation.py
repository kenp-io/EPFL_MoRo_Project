"""
## localNavigation.py ##

Definition of functions used in the local navigation algorithm.

Imported by: globalNavigation.py.

*** Functions ***
navigate(ourThymio, proximity)
get_prox_values(ourThymio)
localCheck(ourThymio)

"""

# ******** IMPORTS ********

import time
import numpy as np
import utils
import globalNavigation
from Thymio import Thymio

# ******** CONSTANTS ********

ROTATE_SLOWBW = 2**16-100 # bw for backward
ROTATE_FASTFW = 200 #fw for frontward
FAST          = 150
SLOW          = 100

DIST_FRONT = 1200  #put corresponding distance in cm
DIST_DIAG  = 1500
DIST_SIDE  = 1500
DIST_THRESH = 1000 #value au bol

# side times
FRONT_TIME = 2.4
DIAG_TIME  = 1.20
SIDE_TIME  = 0.15
ROT_TIME   = 1.58
FW_TIME    = 6.75 #forward time

#####

PROX_LEFT  = 0
PROX_FL   = 1
PROX_FRONT = 2
PROX_FR   = 3
PROX_RIGHT = 4

#measured DISTANCES indexes
DIST_LEFT_SIDE   = 0
DIST_LEFT_DIAG   = 1
DIST_FLEFT       = 2
DIST_FRONT       = 3
DIST_FRIGHT      = 4
DIST_RIGHT_DIAG  = 5
DIST_RIGHT_SIDE  = 6

DIST_THRES       = 1000

CLOSE_OFFSET     = 200

LEFT_OVERTAKE    = (True, False)
RIGHT_OVERTAKE   = (False, True)
CLEAR_WAY        = (True, True)
LEFT             = 0
RIGHT            = 1

PRESENT          = 1
ABSENT           = 0

DISTANCES = [2000, 2500 , 2600, 2200, 2600, 2500, 2000]

# ******** FUNCTIONS ********

def navigate(ourThymio):
    """
    Local Navigation moving algorith

    :param ourThymio: object of class virtualThymio representing our robot,
              gathering state information and class methods
    """

    side_space = (True,True) # Boolean True if there is enough space on left and/or right sides
    side = (False, False)
    long_overtake = ABSENT #ABSENT 0, PRESENT 1 for long overtake

    ourThymio.th.set_var_array("leds.top", [5, 100, 0])

    rot_angle = -90
    proximity = get_prox_values(ourThymio)

    if (sum(proximity[i] for i in range(3)) >= sum(proximity[i] for i in [3,4]) and sum(proximity) > DIST_THRES):
        #object on the left or front
        if side_space[RIGHT]:
            side = RIGHT_OVERTAKE
            long_overtake = ABSENT
            ourThymio.th.set_var_array("leds.top", [0, 50, 100]) #light blue
        elif side_space[LEFT]:
            rot_angle = -rot_angle #=90
            side = LEFT_OVERTAKE
            long_overtake = PRESENT
            ourThymio.th.set_var_array("leds.top", [200, 10, 0]) #dark orange

    elif (sum(proximity[i] for i in [3,4]) > DIST_THRES):
        # object on the right
        if side_space[LEFT]:
            rot_angle = -rot_angle #=90
            side = LEFT_OVERTAKE
            long_overtake = ABSENT
            ourThymio.th.set_var_array("leds.top", [100, 50, 0]) #jaune light
        elif side_space[RIGHT]:
            side = RIGHT_OVERTAKE
            long_overtake = PRESENT
            ourThymio.th.set_var_array("leds.top", [0, 0, 200]) #dark blue

    print("side before if",side)
    if (sum(proximity) > DIST_THRES): #if object detected

        print("\nside",side)
        if (side == LEFT_OVERTAKE): #overtake on left side, inversion of pivoting speeds
            if long_overtake:
                time.sleep(1)
            elif (side_space == CLEAR_WAY):
                while (proximity[PROX_RIGHT] < DISTANCES[PROX_RIGHT]+CLOSE_OFFSET and
                       proximity[PROX_FR] < DISTANCES[PROX_FR]+CLOSE_OFFSET):
                    proximity = get_prox_values(ourThymio)
                    print("proxRovertakeleft",proximity)
                    time.sleep(0.05)

        elif (side == RIGHT_OVERTAKE):
            if long_overtake:
                time.sleep(1)
            elif (side_space == CLEAR_WAY):
                while (proximity[PROX_LEFT] < DISTANCES[PROX_LEFT]+CLOSE_OFFSET and
                       proximity[PROX_FL] < DISTANCES[PROX_FL]+CLOSE_OFFSET and
                       proximity[PROX_FRONT] < DISTANCES[PROX_FRONT]+CLOSE_OFFSET):
                    proximity = get_prox_values(ourThymio)
                    print("proxLovertakeright",proximity)
                    time.sleep(0.05)

        # first turn when the object is in sight
        print(proximity)
        globalNavigation.turnAngle(np.deg2rad(rot_angle),ourThymio)

        # step 1 contournement

        cube = PRESENT
        while(cube == PRESENT):
            #move on the side a bit
            ourThymio.forward()
            print("forward1")
            if long_overtake:
                time.sleep(1.6)
            else:
                time.sleep(1)
            #turn to see if cube overpassed (cube ABSENT)
            globalNavigation.turnAngle(np.deg2rad(-rot_angle),ourThymio)
            proximity[PROX_FRONT] = ourThymio.th["prox.horizontal"][PROX_FRONT]
            proximity[(PROX_LEFT if side[RIGHT] else PROX_RIGHT)] = ourThymio.th["prox.horizontal"][(PROX_LEFT if side[RIGHT] else PROX_RIGHT)]
            proximity[(PROX_FL if side[RIGHT] else PROX_FR)] = ourThymio.th["prox.horizontal"][(PROX_FL if side[RIGHT] else PROX_FR)]
            print("check1")
            #set_motor(left_rotanalyse,right_rotanalyse)
            if (proximity[(PROX_LEFT if side[RIGHT] else PROX_RIGHT)] < DIST_THRES and
                proximity[(PROX_FL if side[RIGHT] else PROX_FR)] < DIST_THRES and
                proximity[PROX_FRONT] < DIST_THRES):
                cube = ABSENT
                print("absent")

            else:
                globalNavigation.turnAngle(np.deg2rad(rot_angle),ourThymio)
                print("returncheck1")

        cube = PRESENT
        while(cube == PRESENT):
            #move on the side a bit
            ourThymio.forward()
            time.sleep(1.3)
            #turn to see if cube overpassed (cube ABSENT)
            globalNavigation.turnAngle(np.deg2rad(-0.65*rot_angle),ourThymio)
            #set_motor(left_rotanalyse,right_rotanalyse)
            proximity[PROX_FRONT] = ourThymio.th["prox.horizontal"][PROX_FRONT]
            proximity[(PROX_LEFT if side[RIGHT] else PROX_RIGHT)] = ourThymio.th["prox.horizontal"][(PROX_LEFT if side[RIGHT] else PROX_RIGHT)]
            proximity[(PROX_FL if side[RIGHT] else PROX_FR)] = ourThymio.th["prox.horizontal"][(PROX_FL if side[RIGHT] else PROX_FR)]
            print("check1")
            if (proximity[(PROX_LEFT if side[RIGHT] else PROX_RIGHT)] < DIST_THRES and
                proximity[(PROX_FL if side[RIGHT] else PROX_FR)] < DIST_THRES and
                proximity[PROX_FRONT] < DIST_THRES):
                cube = ABSENT
                print("absent")

            else:
                globalNavigation.turnAngle(np.deg2rad(0.65*rot_angle),ourThymio)
            ourThymio.th.set_var_array("leds.top", [0, 0, 0])
            side = (False, False)
            rot_angle = -90

def get_prox_values(ourThymio):
    """
    Returns the proximity sensors' values

    :param ourThymio: object of class virtualThymio representing our robot,
                      gathering state information and class methods

    :return: np array of size 5 with proximity sensors values catched
             at the function call
    """
    proximity = np.zeros(5)
    proximity[PROX_LEFT]  = ourThymio.th["prox.horizontal"][PROX_LEFT]
    proximity[PROX_FL]    = ourThymio.th["prox.horizontal"][PROX_FL]
    proximity[PROX_FRONT] = ourThymio.th["prox.horizontal"][PROX_FRONT]
    proximity[PROX_FR]    = ourThymio.th["prox.horizontal"][PROX_FR]
    proximity[PROX_RIGHT] = ourThymio.th["prox.horizontal"][PROX_RIGHT]

    return proximity

def localCheck(ourThymio): #check if object detected to enter loccal nav
    """
    Indicates if the Thymio locally detects an obstacle with its proximity sensors' values

    :param ourThymio: object of class virtualThymio representing our robot,
                      gathering state information and class methods

    :return: boolean indicating whether the Thymio detects an local obstacle
    """

    prox_values = get_prox_values(ourThymio)
    if sum(prox_values) > DIST_THRESH:

        print('PROX VALUES')
        print(prox_values)
        ourThymio.inLocal = True
        ourThymio.stopKalmanFlag.set()
        navigate(ourThymio)

        return True

    else:
        return False
