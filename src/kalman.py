"""
## kalman.py ##

All code related to the Kalman filter algorithm.

Imported by: main.ipynb, globalNavigation.py.

*** Classes ***
kalmanThread(ourThymio)

*** Functions ***
createTracker(iniVector)
runTracker()
runCorrection()

"""

# ******** IMPORTS ********

from filterpy.kalman import KalmanFilter
import numpy as np
from scipy.linalg import block_diag
from threading import Timer,Thread,Event

# ******** CONSTANTS ********

DT = 0.3

# ******** CLASSES ********

class kalmanThread(Thread):
    """
    ## Thread class managing the Kalman filter

    **Initialisation**
    kalmanThread(ourThymio)

    **Functions**
    run()
    """

    def __init__(self, ourThymio):
        """
        ## Initialisation of the thread

        :param ourThymio: object of class virtualThymio representing our robot,
                          gathering state information and class methods
        """
        Thread.__init__(self)
        self.stopped = ourThymio.stopKalmanFlag
        self.ourThymio = ourThymio
        self.tracker = createTracker(self.ourThymio.readKalman())

    def run(self):
        """
        ## Periodically calls the function runTracker() (if thread running),
           updating Kalman state and predictions.
        """
        while not self.stopped.wait(DT):
            if not self.ourThymio.reached:
                runTracker(self)

# ******** FUNCTIONS********

def createTracker(iniVector):
    """
    Creates the Kalman filter matrices to track and predict the Thymio's trajectory

    :param iniVector: initial position and velocity state of the robot

    :return: object tracker of class KalmanFilter with all Kalman matrices
    """

    tracker = KalmanFilter(dim_x=4, dim_z=4)
    dt = 0.1   # time step

    tracker.F = np.array([[1, 0, dt, 0],
                          [0, 1, 0, dt],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
    tracker.B = np.array([[dt, 0],
                          [0, dt],
                          [1,  0],
                          [0,  1]])
    tracker.u = np.array([[1.],[1.]])
    tracker.H = np.diag([1.,1.,1.,1.])
    R = np.diag([5.,5.,3.,3.])
    tracker.Q = np.diag([1.3,1.3,3.5,3.5])
    tracker.x = iniVector
    tracker.P = np.diag([1,1,1,1])

    return tracker


def runTracker(self):
    """
    Update Kalman state and prediction, and call runCorrection() that will
    adapt the speed of the motors acccordingly
    """
    uT = self.ourThymio.getRatios()
    self.tracker.predict(u=uT)
    z = self.ourThymio.readKalman()
    self.tracker.update(z)
        #print('self.x')
    runCorrection(self)
    uT = self.ourThymio.getRatios()


def runCorrection(self):
    """
    Modifiy Thymio's motors' speed to adjust its trajectory
    """

    estXVel = self.tracker.x[2]
    estYVel = self.tracker.x[3]
    [camXVel,camYVel] = self.ourThymio.getVel()
    #avoid dividing by zero
    if camXVel == 0 or camYVel == 0:
        return
    #relative error
    ratioX = (estXVel-camXVel)/max(abs(camXVel),abs(estXVel))
    ratioY = (estYVel-camYVel)/max(abs(camYVel),abs(estYVel))
    if ratioX == 0 or ratioY == 0:
        return
        #print(f'RATIO {ratioX} {ratioY}')
    #choose most important axis
    if abs(ratioX)>=abs(ratioY):
        if ratioX > 0:
            self.ourThymio.correctToRight(abs(ratioX))
        else:
            self.ourThymio.correctToLeft(abs(ratioX))
    else:
        if ratioY > 0:
            self.ourThymio.correctToRight(abs(ratioY))
        else:
            self.ourThymio.correctToLeft(abs(ratioY))
