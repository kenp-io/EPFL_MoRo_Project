import time
import numpy as np
import utils
import globalNavigation
from Thymio import Thymio

ROTATE_SLOWBW = 2**16-100 # bw for backward
ROTATE_FASTFW = 200 #fw for frontward
FAST          = 150
SLOW          = 100

PROX_LEFT  = 0
PROX_FL   = 1
PROX_FRONT = 2
PROX_FR   = 3
PROX_RIGHT = 4

DIST_FRONT = 1200  ##put corresponding distance in cm
DIST_DIAG  = 2400
DIST_SIDE  = 1500
DIST_THRESH = 1000 #value au bol

# side times
FRONT_TIME = 2.4
DIAG_TIME  = 1.20
SIDE_TIME  = 0.15
ROT_TIME   = 1.58
FW_TIME    = 4.5 #forward time



def set_motor(left,right, ourThymio):
    ourThymio.th.set_var("motor.left.target", left)
    ourThymio.th.set_var("motor.right.target", right)


def navigate(ourThymio, proximity):
    #print(proximity)

    right_rotspeed = ROTATE_SLOWBW
    left_rotspeed = ROTATE_FASTFW
    #rot_time = ROT_TIME

    delay_side    = 0
    delay_forward = FW_TIME
    ourThymio.th.set_var_array("leds.top", [0, 50, 10])

    time.sleep(0.5)
    set_motor(SLOW,SLOW, ourThymio)

    while(delay_side == 0):
        proximity = get_prox_values(ourThymio)
        # side proximity sensors
        if (proximity[PROX_LEFT] + proximity[PROX_RIGHT] > DIST_SIDE):
            # object seen by diagonal sensors
            if ((proximity[PROX_FR]  + proximity[PROX_FL] ) > DIST_DIAG): # Computation of an adapted overtaking time
                delay_side = (DIAG_TIME + SIDE_TIME) / 2
                delay_forward = 0.90 * delay_forward
                print("diagside")
                
            # object seen only by side sensors
            else:
                delay_side = SIDE_TIME
                delay_forward = 0.82 * delay_forward
                #rot_time = 0.7 * rot_time # angle of 45°

            # object on the right
            if (proximity[PROX_RIGHT] > proximity[PROX_LEFT]):
                # inversion of speed to turn on the right side
                right_rotspeed, left_rotspeed = left_rotspeed, right_rotspeed
                print("right",proximity[PROX_RIGHT])

            else:
                print("left")

        # diagonal prox sensors
        elif (proximity[PROX_FL]  + proximity[PROX_FR]  > DIST_DIAG):
            # object also seen by front sensor
            if (proximity[PROX_FRONT] > DIST_FRONT):
                delay_side = (FRONT_TIME + DIAG_TIME) / 2
                print("diag_front")
            else:
                delay_side = DIAG_TIME
                delay_forward = delay_forward #0.90 * delay_forward

            # object on the right
            if (proximity[PROX_FR]  > proximity[PROX_FL] ):
                # inversion of speed to turn on the right side
                right_rotspeed, left_rotspeed = left_rotspeed, right_rotspeed
                print("rightdiag")
            print("left_diag")

        # front sensor alone
        elif (proximity[PROX_FRONT] > DIST_FRONT):
            time.sleep(0.05)
            # wiat if the object is on the side
            if (ourThymio.th["prox.horizontal"][PROX_FL] + ourThymio.th["prox.horizontal"][PROX_FR] < DIAG_TIME):
                delay_side = FRONT_TIME
                # turn by right side by default
                print("front")

    # no object visible by any of the proximity sensors
        else:
            print("no object")

    if (delay_side):
        # rotation 90° on left/right side
        set_motor(left_rotspeed,right_rotspeed, ourThymio)
        time.sleep(ROT_TIME)

        # step 1 contournement
        set_motor(SLOW,SLOW, ourThymio)
        time.sleep(delay_side)

        # rotation 90° on right/left side
        set_motor(right_rotspeed,left_rotspeed, ourThymio)
        time.sleep(ROT_TIME)

        # step 2 contournement
        #if (delay != SIDE_TIME):
        set_motor(FAST,FAST, ourThymio)
        time.sleep(delay_forward)

        # rotation 90° on right/left side
        #if (delay != SIDE_TIME):
        '''set_motor(right_rotspeed,left_rotspeed, ourThymio)
        time.sleep(ROT_TIME)

        # step 3 contournement
        set_motor(SLOW,SLOW, ourThymio)
        time.sleep(delay_side)

        # rotation 90° on left/right side
        set_motor(left_rotspeed,right_rotspeed, ourThymio)
        time.sleep(ROT_TIME)'''

        ourThymio.th.set_var_array("leds.top", [0, 50, 10])
        # exit of the robot
        ourThymio.th.set_var_array("leds.top", [0, 0, 0])
        # out of while loop
        i = 0
        delay_side    = 0
        delay_forward = 4.2
        rot_time = ROT_TIME
    # set speed on turning right
    '''right_speed = ROTATE_SLOWBW
    left_speed = ROTATE_FASTFW'''

    #stop the robot
    set_motor(0,0, ourThymio)

def get_prox_values(ourThymio):
    proximity = np.zeros(5)#[0, 0, 0, 0, 0]
    proximity[PROX_LEFT]  = ourThymio.th["prox.horizontal"][PROX_LEFT]
    proximity[PROX_FL]    = ourThymio.th["prox.horizontal"][PROX_FL]
    proximity[PROX_FRONT] = ourThymio.th["prox.horizontal"][PROX_FRONT]
    proximity[PROX_FR]    = ourThymio.th["prox.horizontal"][PROX_FR]
    proximity[PROX_RIGHT] = ourThymio.th["prox.horizontal"][PROX_RIGHT]
    return proximity

def localCheck(ourThymio):
    prox_values = get_prox_values(ourThymio)
    if sum(prox_values) > DIST_THRESH:
        print(prox_values)
        ourThymio.inLocal = True
        ourThymio.stopKalmanFlag.set()
        navigate(ourThymio, prox_values)
        return True
    else:
        return False

