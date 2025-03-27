from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

# Definition of rotation matrix and derivative of rotation matrix:
RM = lambda theta: np.ndarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
RMdot = lambda theta: np.ndarray([[-np.sin(theta),-np.cos(theta)],[np.cos(theta),-np.sin(theta)]])

# Enums for state space:
class xn(Enum):
    THETA = 0
    W = 1
    P = slice(2,4)
    V = slice(4,6)
    A = slice(6,8)
    BW = 8
    BA = slice(9,11)

# Enums for measurement space:
class zn(Enum):
    PHI = 0
    R = 1
    W = 2
    A = slice(3,5)

# ENUMS for input space:
class un(Enum):
    ETAW = 0
    ETAA = slice(1,3)
    ETABW = 3
    ETABA = slice(4,6)

STATE_LEN = 11
MEAS_LEN = 5
INPUT_LEN = 5

class MeasModel:
    """ Base measurement model class
    """
    def __init__(self, t = np.array([0,0]), R = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])):
        """"
        Inits the measurement model
        Args:
            t (np.ndarray): Antenna offset, relative to center of drone
            R (np.ndarray): Measurement noise
        """
        
        self._t = t
        self._R = R 

    @property
    def R(self) -> np.ndarray:
        """ Get measurement covariance matrix
        """
        return self._R
    
    @property
    def t(self) -> np.ndarray:
        """ Get offset 
        """
        return self._t

    def h(self, xi: np.ndarray) -> np.ndarray:
        """
        Computes the simple mesurement update, where we only have access to propertiary sensors

        Args:
            xi (np.ndarray): The information of the state of the ego robot at time 'k'
            
        Returns:
            (np.ndarray): The measurement of state xi, xj 
        """
        z = np.zeros(MEAS_LEN)
        z[zn.W] = xi[xn.W] - xi[xn.BW]
        z[zn.A] = RM(xn.THETA) @ (xi[xn.A] - xi[xn.BA])
        return z

    def h(self, xi: np.ndarray, xj: np.ndarray, tj: np.ndarray) -> np.ndarray:
        """ 
        Computes the masurement update when we have access to both propertiary sensors and measurement with other robot

        Args:
            xi (np.ndarray): The information of the state of the ego robot i at time 'k'
            xj (np.ndarray): The information of the state at the measured robot j at time 'k'
            
        Returns:
            (np.ndarray): The measurement of state xi, xj 
        """
        z = np.zeros(MEAS_LEN)
        # Precompute q
        q = (xj[xn.P] + RM(xj[xn.THETA]) @ tj - xi[xn.P] - RM(xi[xn.THETA]) @ self._t)
        z[zn.PHI] = np.arctan2(q[1], q[0]) - xi[xn.THETA]
        z[zn.R] = np.sqrt(np.transpose(q) @ q)
        z[zn.W] = xi[xn.W] - xi[xn.BW]
        z[zn.A] = RM(xn.THETA) @ (xi[xn.A] - xi[xn.BA])
        return z

    def get_jacobian(self, x0: np.ndarray) -> np.ndarray:
        """
        Computes the jacobian at time k for the measurement

        Args:
            x0 (np.ndarray): The point to evaluate the jacobian at
        
        Returns:
            (np.ndarray): The Jacobian matrix evaluated at x0
        """
        H = np.zeros((MEAS_LEN, STATE_LEN))
        H[zn.W, xn.W] = 1; H[zn.W, xn.BW] = -1
        H[zn.A, xn.THETA] = RMdot(x0[xn.THETA]) @ (x0[xn.A] - x0[xn.BA]); H[zn.A, xn.A] = RM(x0[xn.THETA]); H[zn.A, xn.BA] = -RM(x0[xn.THETA]) 

        return H

    def get_jacobian(self, xi0: np.ndarray, xj0: np.ndarray, tj: np.ndarray) -> np.ndarray:
        """
        Computes the jacobian at time k for the measurement

        Args:
            xi0 (np.ndarray): The stationary point for the ego bot
            xj0 (np.ndarray): The stationary point for the measurement bot
            tj: (np.ndarray): The offset of the measurement bot

        Returns:
            (np.ndarray): The Jacobian matrix evaluated at xi0, xj0
        """
        H = np.zeros((MEAS_LEN, STATE_LEN))
        ti = self._t
        # Precompute q
        q = (xj0[xn.P] + RM(xj0[xn.THETA]) @ tj - xi0[xn.P] - RM(xi0[xn.THETA]) @ ti)
        H[zn.PHI, xn.THETA] = (RMdot(xj0[xn.THETA])[0,:] @ ti * q[1] - RMdot(xj0[xn.THETA])[1,:] @ ti * q[0])/(np.transpose(q) @ q) + 1
        H[zn.PHI, xn.P] = np.array([q[1], -q[0]]) / (np.transpose(q) @ q)
        H[zn.R, xn.THETA] = (np.transpose(q) @ (-RMdot(xi0[xn.THETA])) @ ti + np.transpose(-RMdot(xi0[xn.THETA]) @ ti) @ q) / (2*np.sqrt(np.transpose(q) @ q))
        H[zn.R, xn.P] = - np.transpose(q) / np.sqrt(np.transpose(q) @ q)
        # and the IMU measurements:
        H[zn.W, xn.W] = 1; H[zn.W, xn.BW] = -1
        H[zn.A, xn.THETA] = RMdot(xi0[xn.THETA]) @ (xi0[xn.A] - xi0[xn.BA]); H[zn.A, xn.A] = RM(xi0[xn.THETA]); H[zn.A, xn.BA] = -RM(xi0[xn.THETA]) 

        return H

class MotionModel:
    """ Base measurement model class
    """
    def __init__(self, dt: float = 1.0, 
                 Q: np.ndarray = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]), 
                 x0: np.ndarray = np.zeros(STATE_LEN)):
        """"
        Inits the measurement model
        Args:
            dt float: time difference between k
            R (np.ndarray): Measurement noise
        """

        self._Q = Q   
        A = np.eye(STATE_LEN)
        B = np.zeros((STATE_LEN, INPUT_LEN))

        A[xn.THETA, xn.W] = dt
        A[xn.P, xn.V] = dt*np.eye(2); A[xn.P, xn.A] = 0.5*dt**2*np.eye(2)
        A[xn.V, xn.A] = dt*np.eye(2) 
        self._A = A

        B[xn.W, un.ETAW] = 1; B[xn.A, un.ETAA] = np.eye(2); B[xn.BW, un.ETABW] = 1; B[xn.BA, un.ETABA] = np.eye(2)
        self._B = B

        self._x = x0

    def predict(self, x: np.ndarray) -> np.ndarray:
        xnew = self._A @ self._x + self._B @ self._Q
        self._x = xnew
        return xnew
    
    def propagate(self) -> np.ndarray:
        """
        Propagate process noise
        """
        pass
        
 

