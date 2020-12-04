from filterpy.kalman import KalmanFilter
import numpy as np
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
from threading import Timer,Thread,Event

class (Thread):
    def __init__(self, event, ourThymio):
        Thread.__init__(self)
        self.stopped = event
        self.ourThymio = ourThymio

    def run(self):
        while not self.stopped.wait(0.5):
            print("my thread")
            # call a function

def createTracker(iniVector):
    
    tracker = KalmanFilter(dim_x=4, dim_z=4)
    dt = 0.1   # time step

    tracker.F = np.array([[1, 0, dt, 0],
                          [0, 1, 0, dt],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]])
    tracker.B = np.array([[dt, 0],
                          [0, dt],
                          [1,  0],
                          [0,  1]])
    tracker.u = np.array([[1.],[1.]])
    tracker.H = np.diag([1.,1.,1.,1.])
    R = np.diag([5.,5.,3.,3.])
    q = Q_discrete_white_noise(dim=2, dt=dt, var=0.04**2)
    tracker.Q = block_diag(q, q)
    tracker.x = iniVector
    tracker.P = np.diag([3,3,3,3])

    return tracker

def runTracker(iniVector):
    

