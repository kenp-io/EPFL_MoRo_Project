from filterpy.kalman import KalmanFilter
import numpy as np
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
from threading import Timer,Thread,Event

DT = 0.5

class kalmanThread(Thread):
    def __init__(self, ourThymio):
        Thread.__init__(self)
        self.stopped = ourThymio.stopKalmanFlag
        self.ourThymio = ourThymio
        self.tracker = createTracker(self.ourThymio.readKalman())

    def run(self):
        while not self.stopped.wait(DT):
            if not self.ourThymio.reached:
                runTracker(self)

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

def runTracker(self):

    self.tracker.predict(u=self.ourThymio.getVel())
        #print('predict')
        #print(self.tracker.x)
    z = self.ourThymio.readKalman()
    self.tracker.update(z)
        #print('update')
        #print(self.tracker.x)
        #print('self.x')
        #print(self.ourThymio.getVel())
    runCorrection(self)

def runCorrection(self):

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

            #print(f'RATIO {ratioX} {ratioY}')
