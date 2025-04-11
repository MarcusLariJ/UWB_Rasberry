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
IMU_LEN = 3
INPUT_LEN = 6


#### Helper functions ####


def wrappingPi(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ Takes in two arrays of radians and computes the difference when the scale is -pi to pi

    Args:
        x (np.ndarray): First vector
        y (np.ndarray): Second vector

    Returns:
        np.ndarray: Difference of vector
    """
    n, nx = x.shape
    if x.size:
        return ((x - y + np.pi) % (2*np.pi)) - np.pi
    else:
        return np.zeros(shape=(n, nx), dtype=x.dtype)

def subtractState(s0x: np.ndarray, s1x: np.ndarray, rad_sel: np.ndarray) -> np.ndarray:
    # Handle states with radians as units
    n, nx = s0x.shape
    x_diff = np.zeros((n, nx))
    # Radian subtraction
    x_diff[rad_sel] = wrappingPi(s0x[rad_sel], s1x[rad_sel])
    # Other states; normal subtraction
    x_diff[~rad_sel] = s0x[~rad_sel] - s1x[~rad_sel]
    return x_diff


#### Models ####


class MeasModel:
    """ Base measurement model class
    """
    def __init__(self, t = np.array([[0],[0]]), R = np.diag([0.1]*MEAS_LEN)):
        """"
        Inits the measurement model
        Args:
            t (np.ndarray): Antenna offset, relative to center of drone
            R (np.ndarray): Measurement noise
        """
        
        self._t = t
        self._R = R 
        sel = np.array([False]*MEAS_LEN)
        sel[Z_PHI] = True 
        self._radian_sel = sel
        self._H = np.zeros((MEAS_LEN, STATE_LEN))
        self._z = np.zeros((MEAS_LEN, 1))

    @property
    def R(self) -> np.ndarray:
        """ Get measurement covariance matrix
        """
        return self._R
    
    @R.setter
    def R(self, var: np.ndarray):
        """ Set measurement matrix
        """
        self._R = var
    
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
        self._z[Z_A] = RM(-xi[X_THETA][0]) @ (xi[X_A] - xi[X_BA])
        
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
        thetai = x0[X_THETA][0]
        self._H[Z_W, X_W] = 1; self._H[Z_W, X_BW] = -1
        self._H[Z_A, X_THETA:X_THETA+1] = RMdot(-thetai) @ (x0[X_A] - x0[X_BA]); self._H[Z_A, X_A] = RM(-thetai); self._H[Z_A, X_BA] = -RM(-thetai) 

        # Return only partial H, corresponding to IMU measurements:
        return self._H[Z_W:,:]

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
        thetai = xi0[X_THETA][0]
        thetaj = xj0[X_THETA][0]
        # Precompute q
        q = (xj0[X_P] + RM(thetaj) @ tj - xi0[X_P] - RM(thetai) @ ti)
        # Compute the Jacobian for IMU measurement:
        self.get_jacobian_IMU(xi0)
        # Compute the Jacobian for range/angle measurement:
        self._H[Z_PHI, X_THETA] = (RMdot(thetai)[0,:] @ ti * q[1] - RMdot(thetai)[1,:] @ ti * q[0])/(np.transpose(q) @ q) - 1
        self._H[Z_PHI, X_P] = np.array([q[1], -q[0]]).reshape(1,-1) / (np.transpose(q) @ q)
        self._H[Z_R, X_THETA] = (np.transpose(q) @ (-RMdot(thetai)) @ ti + np.transpose(-RMdot(thetai) @ ti) @ q) / (2*np.sqrt(np.transpose(q) @ q))
        self._H[Z_R, X_P] = -np.transpose(q) / np.sqrt(np.transpose(q) @ q)

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
        self._dt = dt
        A = np.eye(STATE_LEN)
        B = np.zeros((STATE_LEN, INPUT_LEN))

        A[X_THETA, X_W] = dt
        A[X_P, X_V] = dt*np.eye(2); A[X_P, X_A] = 0.5*np.eye(2)*(dt**2)
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
        """
        Set the state
        NOTICE: do NOT set individual states using indicies!
        This will overwrite all states of all motion models.
        Instead, pass a new numpy array
        """
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
        # TODO: notice dt here
        Pnew = self._A @ self._P @ np.transpose(self._A) + self._B @ self._Q @ np.transpose(self._B)*self._dt
        self._P = Pnew
        return Pnew


#### Kalman Filters ####


def _KF(x: np.ndarray, 
       P: np.ndarray, 
       H: np.ndarray, 
       R: np.ndarray, 
       y: np.ndarray,
       ypred: np.ndarray,
       radian_sel: np.ndarray):
    """
    Computes the Kalman gain and updates the states and covariance.
    Args: 
        x (np.ndarray): States to update
        P (np.ndarray): Process noise to update
        H (np.ndarray): Jacobian of output matrix
        y (np.ndarray): Measured output
        radian_sel (np.ndarray): Array indicating which states are given in radians
    """
    Q = H @ P @ np.transpose(H) + R
    r_sel = radian_sel
    inno = subtractState(y, ypred, r_sel)
    # Compute gain:
    K = P @ np.transpose(H) @ np.linalg.inv(Q)
    # Update state estimate
    xnew = x + K @ inno 
    # Update covariance
    Pnew = (np.eye(STATE_LEN) - K @ H) @ P
    # Return new values:
    return xnew, Pnew, inno, K

def KF_IMU(mot: MotionModel, meas: MeasModel, y: np.ndarray) -> np.ndarray:
    """
    Simple KF for when IMU measurements come in
    Args:
        mot (MotionModel): Motion model of the robot 
        meas (MeasModel): Measurement model of the robot
        y (np.ndarray): Incoming measurement
    Returns:
        inno (np.ndarray): Innovation
    """
    x = mot.x
    P = mot.P
    H = meas.get_jacobian_IMU(x)
    R = meas.R[Z_W:,Z_W:] # Only use noise related to IMU
    ypred = meas.h_IMU(x)
    radian_sel = meas.radian_sel[Z_W:]
    xnew, Pnew, inno, _ = _KF(x, P, H, R, y, ypred, radian_sel)
    mot.x = xnew
    mot.P = Pnew

    return inno

def KF_full(moti: MotionModel, 
            motj: MotionModel, 
            measi: MeasModel,
            measj: MeasModel, 
            y: np.ndarray) -> np.ndarray:
    """
    KF for both IMU for robot i and range/bearing measurement between robot i and j
    Args:
        moti (MotionModel): Motion model of the ego robot i 
        motj (MotionModel): Motion model of the other robot j
        measi (MeasModel): Measurement model of the robot i
        measj (MeasModel): Meadurement model of the robot j
        y (np.ndarray): Incoming measurement
    Returns:
        inno (np.ndarray): Innovation
    """
    xi = moti.x
    xj = motj.x
    tj = measj.t
    P = moti.P
    H = measi.get_jacobian_full(xi, xj, tj)
    R = measi.R
    ypred = measi.h_full(xi, xj, tj)
    radian_sel = measi.radian_sel
    xnew, Pnew, inno, _ = _KF(xi, P, H, R, y, ypred, radian_sel)
    moti.x = xnew
    moti.P = Pnew

    return inno

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


#### Fake measurements ####


def gen_IMU_meas(pos: np.ndarray,
                dt,
                sigma: np.ndarray = np.diag([0]*IMU_LEN), 
                bias: np.ndarray = np.zeros((IMU_LEN,1))) -> np.ndarray:
    """
    From a sequence of poses, this function generates a sequence of IMU measurements
    TODO: Maybe pad output meas a bit out with zeros
    TODO: Delete
    Args:
        pos (np.ndarray): Array of positions. Dimension = 3 X MEAS_NUM
        sigma (np.ndarray): Noise power. Dimension = IMU_LEN X IMU_LEN
        bias (np.ndarray): Constant measurement bias to apply. Dimension = IMU_LEN X 1
        dt: Time difference
    returns:
        meas (np.ndarray): Generated IMU measurements
        x0 (np.ndarray): Starting conditions of posistion, velocity and acceleration 
    """
    x0 = np.zeros((STATE_LEN, 1))
    x0[X_THETA] = pos[0,0]
    x0[X_P] = pos[1:,0:1]

    # First, find first derivative:
    v_diff = np.diff(pos, axis=1)*(1/dt)
    x0[X_W] = v_diff[0,0]
    x0[X_V] = v_diff[1:,0:1]

    # Then next derivative:
    a_diff = np.diff(v_diff, axis=1)*(1/dt)
    x0[X_A] = a_diff[1:,0:1]
    a_diff[0,:] = v_diff[0,:-1] # The angle is only of 1st order
    meas_len, meas_num = a_diff.shape
    
    # Convert to body acceleration:
    for i in range(meas_num):
        a_diff[1:,i] = RM(-pos[0,i]) @ a_diff[1:,i]

    # Random noise: TODO: is this applied correctly?
    noise = (sigma*dt) @ np.random.rand(IMU_LEN, meas_num)
    # Bias:
    bias_m = np.repeat(bias, meas_num, axis=1)
    # Combine noise and bias:
    meas = a_diff + noise + bias_m

    return meas, x0
    








