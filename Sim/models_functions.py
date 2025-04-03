from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

# Definition of rotation matrix and derivative of rotation matrix:
RM = lambda theta: np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
RMdot = lambda theta: np.array([[-np.sin(theta),-np.cos(theta)],[np.cos(theta),-np.sin(theta)]])

# Enums for state space:
X_THETA = 0
X_W = 1
X_P = slice(2,4)
X_V = slice(4,6)
X_A = slice(6,8)
X_BW = 8
X_BA = slice(9,11)

# Enums for measurement space:
Z_PHI = 0
Z_R = 1
Z_W = 2
Z_A = slice(3,5)

# ENUMS for input space:
U_ETAW = 0
U_ETAA = slice(1,3)
U_ETABW = 3
U_ETABA = slice(4,6)

STATE_LEN = 11
MEAS_LEN = 5
INPUT_LEN = 6

def wrappingPi(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ Takes in two arrays of radians and computes the difference when the scale is -pi to pi

    Args:
        x (np.ndarray): First vector
        y (np.ndarray): Second vector

    Returns:
        np.ndarray: Difference of vector
    """
    n, nx, _ = x.shape
    if x.size:
        return ((x - y + np.pi) % (2*np.pi)) - np.pi
    else:
        return np.zeros(shape=(n, nx, 1), dtype=x.dtype)

def subtractState(s0x: np.ndarray, s1x: np.ndarray, rad_sel: np.ndarray) -> np.ndarray:
    # Handle states with radians as units
    n, nx, _ = s0x.shape
    x_diff = np.zeros((n, nx, 1))
    # Radian subtraction
    x_diff[:,rad_sel] = wrappingPi(s0x[:,rad_sel], s1x[:,rad_sel])
    # Other states; normal subtraction
    x_diff[:,~rad_sel] = s0x[:,~rad_sel] - s1x[:,~rad_sel]
    return x_diff


#### Models ####


class MeasModel:
    """ Base measurement model class
    """
    def __init__(self, t = np.array([[0],[0]]), R = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])):
        """"
        Inits the measurement model
        Args:
            t (np.ndarray): Antenna offset, relative to center of drone
            R (np.ndarray): Measurement noise
        """
        
        self._t = t
        self._R = R 
        self._radian_sel = np.array([Z_PHI])
        self._H = np.zeros((MEAS_LEN, STATE_LEN))
        self._z = np.zeros((MEAS_LEN, 1))

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

    @property
    def radian_sel(self) -> np.ndarray:
        """ Get indicies of states with radian unit
        """
        return self._radian_sel

    def h_IMU(self, xi: np.ndarray) -> np.ndarray:
        """
        Computes the simple mesurement update, where we only have access to propertiary sensors

        Args:
            xi (np.ndarray): The information of the state of the ego robot at time 'k'
            
        Returns:
            (np.ndarray): The measurement of state xi, xj 
        """
        self._z[Z_W] = xi[X_W] - xi[X_BW]
        self._z[Z_A] = RM(xi[X_THETA][0]) @ (xi[X_A] - xi[X_BA])
        
        # ONly return the part that corresponds to IMU measurement
        return self._z[Z_W:]

    def h_full(self, xi: np.ndarray, xj: np.ndarray, tj: np.ndarray) -> np.ndarray:
        """ 
        Computes the masurement update when we have access to both propertiary sensors and measurement with other robot

        Args:
            xi (np.ndarray): The information of the state of the ego robot i at time 'k'
            xj (np.ndarray): The information of the state at the measured robot j at time 'k'
            
        Returns:
            (np.ndarray): The measurement of state xi, xj 
        """
        # Precompute q
        thetaj = xj[X_THETA][0]
        thetai = xi[X_THETA][0]
        q = (xj[X_P] + RM(thetaj) @ tj - xi[X_P] - RM(thetai) @ self._t)
        # Range and bearing:
        self._z[Z_PHI] = np.arctan2(q[1], q[0]) - xi[X_THETA]
        self._z[Z_R] = np.sqrt(np.transpose(q) @ q)
        # The IMU measurement:
        self.h_IMU(xi)

        return self._z

    def get_jacobian_IMU(self, x0: np.ndarray) -> np.ndarray:
        """
        Computes the jacobian at time k for the IMU measurement

        Args:
            x0 (np.ndarray): The point to evaluate the jacobian at
        
        Returns:
            (np.ndarray): The Jacobian matrix evaluated at x0
        """
        self._H[Z_W, X_W] = 1; self._H[Z_W, X_BW] = -1
        self._H[Z_A, X_THETA] = RMdot(x0[X_THETA]) @ (x0[X_A] - x0[X_BA]); self._H[Z_A, X_A] = RM(x0[X_THETA]); self._H[Z_A, X_BA] = -RM(x0[X_THETA]) 

        # Return only partial H, corresponding to IMU measurements:
        return self._H[Z_W,:]

    def get_jacobian_full(self, xi0: np.ndarray, xj0: np.ndarray, tj: np.ndarray) -> np.ndarray:
        """
        Computes the jacobian at time k for the measurement

        Args:
            xi0 (np.ndarray): The stationary point for the ego bot
            xj0 (np.ndarray): The stationary point for the measurement bot
            tj: (np.ndarray): The offset of the measurement bot

        Returns:
            (np.ndarray): The Jacobian matrix evaluated at xi0, xj0
        """
        ti = self._t
        # Precompute q
        q = (xj0[X_P] + RM(xj0[X_THETA]) @ tj - xi0[X_P] - RM(xi0[X_THETA]) @ ti)
        # Compute the Jacobian for IMU measurement:
        self.get_jacobian_IMU(xi0)
        # Compute the Jacobian for range/angle measurement:
        self._H[Z_PHI, X_THETA] = (RMdot(xj0[X_THETA])[0,:] @ ti * q[1] - RMdot(xj0[X_THETA])[1,:] @ ti * q[0])/(np.transpose(q) @ q) + 1
        self._H[Z_PHI, X_P] = np.array([q[1], -q[0]]) / (np.transpose(q) @ q)
        self._H[Z_R, X_THETA] = (np.transpose(q) @ (-RMdot(xi0[X_THETA])) @ ti + np.transpose(-RMdot(xi0[X_THETA]) @ ti) @ q) / (2*np.sqrt(np.transpose(q) @ q))
        self._H[Z_R, X_P] = - np.transpose(q) / np.sqrt(np.transpose(q) @ q)

        return self._H

class MotionModel:
    """ Base measurement model class
    """
    def __init__(self, dt: float = 1.0,  
                x0: np.ndarray = np.zeros((STATE_LEN, 1)),
                P: np.ndarray = np.zeros((STATE_LEN, STATE_LEN)),
                Q = np.diag([0.5]*INPUT_LEN)     ):
        """"
        Inits the measurement model
        Args:
            dt float: time difference between k
            P (np.ndarray): Process noise
        """

        self._P = P
        self._Q = Q
        A = np.eye(STATE_LEN)
        B = np.zeros((STATE_LEN, INPUT_LEN))

        A[X_THETA, X_W] = dt
        A[X_P, X_V] = dt*np.eye(2); A[X_P, X_A] = 0.5*dt**2*np.eye(2)
        A[X_V, X_A] = dt*np.eye(2) 
        self._A = A

        B[X_W, U_ETAW] = 1
        B[X_A, U_ETAA] = np.eye(2)
        B[X_BW, U_ETABW] = 1
        B[X_BA, U_ETABA] = np.eye(2)
        self._B = B

        self._x = x0
        self._radian_sel = np.array([X_THETA])

    @property
    def x(self) -> np.ndarray:
        """
        Get current state
        """
        return self._x

    @x.setter
    def x(self, state: np.ndarray):
        self._x = state

    @property
    def P(self) -> np.ndarray:
        """ Get state covariance matrix
        """
        return self._P

    @P.setter
    def P(self, var: np.ndarray):
        self._P = var

    @property
    def Q(self) -> np.ndarray:
        return self._Q
    
    @Q.setter
    def Q(self, var: np.ndarray):
        self._Q = var

    @property
    def radian_sel(self):
        return self._radian_sel

    @property
    def A(self) -> np.ndarray:
        return self._A

    @property
    def B(self) -> np.ndarray:
        return self._B

    def predict(self) -> np.ndarray:
        """
        Predict next state
        """
        xnew = self._A @ self._x # This motion model assumes zero mean on the inputs
        self._x = xnew
        return xnew
    
    def propagate(self) -> np.ndarray:
        """
        Propagate process noise
        Args:
            Q (np.ndarray): Input noise
        Returns:
            Updated state covariance
        """
        Pnew = self._A @ self._P @ np.transpose(self._A) + self._B @ self._Q @ np.transpose(self._B)
        self._P = Pnew
        return Pnew


#### Kalman Filters ####


def SimpleKalmanFilter(mot: MotionModel, meas: MeasModel, y: np.ndarray):
    """
    Computes the Kalman gain and updates the states and covariance.
    This filter is for single robot uses or when implementing a 
    naive collaborative localiZ_Ation system where cross correlations are neglected.
    Args: 
        mot (MotionModel): The motion model to use
        meas (MeasModel): The measurement model to use
        y (np.ndarray): the measurement recived by the sensors 
    """
    P = mot.P
    x = mot.x
    C = meas.get_jacobian(mot.x)
    R = meas.R
    Q = C @ P @ np.transpose(C) + R
    ypred = C @ x
    r_sel = meas.radian_sel
    inno = subtractState(y, ypred, r_sel)
    # Compute gain:
    K = P @ np.transpose(C) @ np.invert(Q)
    # Update state estimate
    xnew = x + K @ inno 
    # Update covariance
    Pnew = (np.eye(STATE_LEN) - K @ C) @ P @ np.transpose(np.eye(STATE_LEN) - K @ C) + K @ R @ np.transpose(K)
    # Apply updates to robots:
    mot.x = xnew
    mot.P = Pnew
    return xnew, Pnew


def centralizedKF(moti: MotionModel, motj: MotionModel, meas: MeasModel, y: np.ndarray):
    """
    Computes the Kalman gain and updates the states and covariance.
    This filter is for implementing a collaborative centralized KF 
    that tracks all the eX_Act correlations between robots
    Args: 
        moti (MotionModel): The motion model to use for the robot i recieving the measurement from j
        motj (MotionModel): The motion model to use for the robot j sending the measurement to i
        meas (MeasModel): The measurement model to use
        y (np.ndarray): the measurement recived by the sensors 
    """
    Pii = 0 
    Pij = 0
    Pji = 0
    Pjj = 0
    xinew = 0
    xjnew = 0

    return xinew, xjnew, Pii, Pij, Pji, Pjj

def luftKF():
    """
    Computes the Kalman gain and updates the states and covariance.
    This filter implements the collaborative localiZ_Ation scheme used by Lukas Luft et al. (2018), 
    where the inter-robot correlations are approximated.
    Args: 
        mot (MotionModel): The motion model to use
        meas (MeasModel): The measurement model to use
        y (np.ndarray): the measurement recived by the sensors 
    """
    pass









