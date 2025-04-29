from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

# Definition of rotation matrix and derivative of rotation matrix:
RM = lambda theta: np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
RMdot = lambda theta: np.array([[-np.sin(theta),-np.cos(theta)],[np.cos(theta),-np.sin(theta)]])

STATE_LEN = 11
MEAS_LEN = 5
IMU_LEN = 3
RB_LEN = 2
INPUT_LEN = 6

# Enums for state space:
X_THETA = 0
X_W = 1
X_P = slice(2,4)
X_V = slice(4,6)
X_A = slice(6,8)
X_BW = 8
X_BA = slice(9,11)
X_THETA_EXT = X_THETA+STATE_LEN
X_P_EXT = slice(2+STATE_LEN, 4+STATE_LEN)

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

    def h_bearing(self, xi: np.ndarray, xj: np.ndarray, tj: np.ndarray) -> np.ndarray:
        """
        Computes the bearing prediction only

        Args:
            xi (np.ndarray): The information of the state of the ego robot at time 'k'
            xj (np.ndarray): The information of the state at the measured robot j at time 'k'
            
        Returns:
            float: the bearing of state xi, xj 
        """
        # Precompute q
        thetaj = xj[X_THETA][0]
        thetai = xi[X_THETA][0]
        q = (xj[X_P] + RM(thetaj) @ tj - xi[X_P] - RM(thetai) @ self._t)
        
        phi = np.arctan2(q[1], q[0]) - xi[X_THETA]
        
        # ONly return the part that corresponds to bearing
        return phi
    
    def h_rb(self, xi: np.ndarray, xj: np.ndarray, tj: np.ndarray) -> np.ndarray:
        """ 
        Computes the masurement update when we have access to measurement with other robot

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

        return self._z[:Z_W]

    def h_IMU(self, xi: np.ndarray) -> np.ndarray:
        """
        Computes the simple mesurement update, where we only have access to propertiary sensors

        Args:
            xi (np.ndarray): The information of the state of the ego robot at time 'k'
            
        Returns:
            (np.ndarray): The measurement of state xi, xj 
        """
        self._z[Z_W] = xi[X_W] + xi[X_BW]
        self._z[Z_A] = RM(-xi[X_THETA][0]) @ xi[X_A] + xi[X_BA]
        
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
        #thetaj = xj[X_THETA][0]
        #thetai = xi[X_THETA][0]
        #q = (xj[X_P] + RM(thetaj) @ tj - xi[X_P] - RM(thetai) @ self._t)
        # Range and bearing:
        #self._z[Z_PHI] = np.arctan2(q[1], q[0]) - xi[X_THETA]
        #self._z[Z_R] = np.sqrt(np.transpose(q) @ q)
        
        # RB measurement:
        self.h_rb(xi, xj, tj)
        # The IMU measurement:
        self.h_IMU(xi)

        return self._z

    def get_jacobian_bearing(self, xi0: np.ndarray, xj0: np.ndarray, tj: np.ndarray) -> np.ndarray:
        
        H = np.zeros((1, STATE_LEN))

        ti = self._t
        thetai = xi0[X_THETA][0]
        thetaj = xj0[X_THETA][0]
        # Precompute q
        q = (xj0[X_P] + RM(thetaj) @ tj - xi0[X_P] - RM(thetai) @ ti)

        H[0, X_THETA] = (RMdot(thetai)[0,:] @ ti * q[1] - RMdot(thetai)[1,:] @ ti * q[0])/(np.transpose(q) @ q) - 1
        H[0, X_P] = np.array([q[1], -q[0]]).reshape(1,-1) / (np.transpose(q) @ q)
        
        return H


    def get_jacobian_rb(self, xi0: np.ndarray, xj0: np.ndarray, tj: np.ndarray) -> np.ndarray:
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
        # Compute the Jacobian for range/angle measurement:
        self._H[Z_PHI, X_THETA] = (RMdot(thetai)[0,:] @ ti * q[1] - RMdot(thetai)[1,:] @ ti * q[0])/(np.transpose(q) @ q) - 1
        self._H[Z_PHI, X_P] = np.array([q[1], -q[0]]).reshape(1,-1) / (np.transpose(q) @ q)
        self._H[Z_R, X_THETA] = (np.transpose(q) @ (-RMdot(thetai)) @ ti + np.transpose(-RMdot(thetai) @ ti) @ q) / (2*np.sqrt(np.transpose(q) @ q))
        self._H[Z_R, X_P] = -np.transpose(q) / np.sqrt(np.transpose(q) @ q)

        return self._H[:Z_W,:]

    def get_jacobian_rb_ext(self, xi0: np.ndarray, xj0: np.ndarray, tj: np.ndarray) -> np.ndarray:
        """
        Computes the jacobian at time k for the measurement of both robots

        Args:
            xi0 (np.ndarray): The stationary point for the ego bot
            xj0 (np.ndarray): The stationary point for the measurement bot
            tj: (np.ndarray): The offset of the measurement bot

        Returns:
            (np.ndarray): The Jacobian matrix evaluated at xi0, xj0
        """
        Hij = np.zeros((RB_LEN, STATE_LEN*2))
        ti = self._t
        thetai = xi0[X_THETA][0]
        thetaj = xj0[X_THETA][0]
        # Precompute q
        q = (xj0[X_P] + RM(thetaj) @ tj - xi0[X_P] - RM(thetai) @ ti)
        # Compute the normal jacobian first
        Hij[:,:STATE_LEN] = self.get_jacobian_rb(xi0=xi0, xj0=xj0, tj=tj)
        # Then the extended part:
        Hij[Z_PHI, X_THETA_EXT] = (-RMdot(thetaj)[0,:] @ tj * q[1] + RMdot(thetaj)[1,:] @ tj * q[0])/(np.transpose(q) @ q)
        Hij[Z_PHI, X_P_EXT] = np.array([-q[1], q[0]]).reshape(1,-1) / (np.transpose(q) @ q)
        Hij[Z_R, X_THETA_EXT] = (np.transpose(q) @ (RMdot(thetaj)) @ tj + np.transpose(RMdot(thetaj) @ tj) @ q) / (2*np.sqrt(np.transpose(q) @ q))
        Hij[Z_R, X_P_EXT] = np.transpose(q) / np.sqrt(np.transpose(q) @ q)

        return Hij

    def get_jacobian_IMU(self, x0: np.ndarray) -> np.ndarray:
        """
        Computes the jacobian at time k for the IMU measurement

        Args:
            x0 (np.ndarray): The point to evaluate the jacobian at
        
        Returns:
            (np.ndarray): The Jacobian matrix evaluated at x0
        """
        thetai = x0[X_THETA][0]
        self._H[Z_W, X_W] = 1; self._H[Z_W, X_BW] = 1
        self._H[Z_A, X_THETA:X_THETA+1] = -RMdot(-thetai) @ x0[X_A]; self._H[Z_A, X_A] = RM(-thetai); self._H[Z_A, X_BA] = np.eye(2) 

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
        self.get_jacobian_rb(xi0, xj0, tj)

        #self._H[Z_PHI, X_THETA] = (RMdot(thetai)[0,:] @ ti * q[1] - RMdot(thetai)[1,:] @ ti * q[0])/(np.transpose(q) @ q) - 1
        #self._H[Z_PHI, X_P] = np.array([q[1], -q[0]]).reshape(1,-1) / (np.transpose(q) @ q)
        #self._H[Z_R, X_THETA] = (np.transpose(q) @ (-RMdot(thetai)) @ ti + np.transpose(-RMdot(thetai) @ ti) @ q) / (2*np.sqrt(np.transpose(q) @ q))
        #self._H[Z_R, X_P] = -np.transpose(q) / np.sqrt(np.transpose(q) @ q)

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
        Returns:
            Updated state covariance
        """
        # TODO: notice dt here
        Pnew = self._A @ self._P @ np.transpose(self._A) + self._B @ self._Q @ np.transpose(self._B)*self._dt
        self._P = Pnew
        return Pnew

    def propagate_rom(self, cor: np.ndarray, cor_N: int):
        """
        Runs the normal propagate step and updates every correlation using Rom's method
        Args:
            cor (np.ndarray): list of inter-robot correlations
            cor_N (np.ndarray): number of inter-robot correlations
        """
        Pnew = self.propagate()
        for i in range(cor_N):
            cor[:,:,i] = self._A @ cor[:,:,i]
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
    Pnew = (np.eye(P.shape[0]) - K @ H) @ P
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
        K (np.ndarray): Kalman gain vector
    """
    x = mot.x
    P = mot.P
    H = meas.get_jacobian_IMU(x)
    R = meas.R[Z_W:,Z_W:] # Only use noise related to IMU
    ypred = meas.h_IMU(x)
    radian_sel = meas.radian_sel[Z_W:]
    xnew, Pnew, inno, K = _KF(x, P, H, R, y, ypred, radian_sel)
    mot.x = xnew
    mot.P = Pnew

    return inno, K

def KF_rb(moti: MotionModel, 
            motj: MotionModel, 
            measi: MeasModel,
            measj: MeasModel, 
            y: np.ndarray) -> np.ndarray:
    """
    KF for and range/bearing measurement between robot i and anchor j
    Assumes no uncertainty about state vector of j!
    Args:
        moti (MotionModel): Motion model of the ego robot i 
        motj (MotionModel): Motion model of the other robot j
        measi (MeasModel): Measurement model of the robot i
        measj (MeasModel): Meadurement model of the robot j
        y (np.ndarray): Incoming measurement
    Returns:
        inno (np.ndarray): Innovation
        K (np.ndarray): Kalman gain vector
    """
    xi = moti.x
    xj = motj.x
    tj = measj.t
    P = moti.P
    H = measi.get_jacobian_rb(xi, xj, tj)
    R = measi.R[:Z_W, :Z_W] # Use only noise related to range/bearing:
    ypred = measi.h_rb(xi, xj, tj)
    radian_sel = measi.radian_sel[:Z_W]
    xnew, Pnew, inno, K = _KF(xi, P, H, R, y, ypred, radian_sel)
    moti.x = xnew
    moti.P = Pnew

    return inno, K

def KF_rb_ext(moti: MotionModel, 
            motj: MotionModel, 
            measi: MeasModel,
            measj: MeasModel, 
            y: np.ndarray) -> np.ndarray:
    """
    KF for and range/bearing measurement between robot i and j,
    that takes both noise sources into account.
    Used for the naive implementation of collaborative localization
    Args:
        moti (MotionModel): Motion model of the ego robot i 
        motj (MotionModel): Motion model of the other robot j
        measi (MeasModel): Measurement model of the robot i
        measj (MeasModel): Meadurement model of the robot j
        y (np.ndarray): Incoming measurement
    Returns:
        inno (np.ndarray): Innovation
        K (np.ndarray): Kalman gain vector
    """
    xi = moti.x
    xj = motj.x
    tj = measj.t
    P = np.zeros((2*STATE_LEN, 2*STATE_LEN))
    P[:STATE_LEN, :STATE_LEN] = moti.P #Pii
    P[STATE_LEN:, STATE_LEN:] = motj.P #Pjj
    H = measi.get_jacobian_rb_ext(xi, xj, tj)
    R = measi.R[:Z_W, :Z_W] # Use only noise related to range/bearing:
    ypred = measi.h_rb(xi, xj, tj)
    radian_sel = measi.radian_sel[:Z_W]
    # Pad the x state with zeros, so dimensions fit:
    x = np.pad(xi, ((0, STATE_LEN), (0, 0)), mode='constant') 
    xnew, Pnew, inno, K = _KF(x, P, H, R, y, ypred, radian_sel)
    moti.x = xnew[:STATE_LEN,:] #Only update state xi
    moti.P = Pnew[:STATE_LEN, :STATE_LEN] #Only update Pii

    return inno, K

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
        K (np.ndarray): Kalman gain vector
    """
    xi = moti.x
    xj = motj.x
    tj = measj.t
    P = moti.P
    H = measi.get_jacobian_full(xi, xj, tj)
    R = measi.R
    ypred = measi.h_full(xi, xj, tj)
    radian_sel = measi.radian_sel
    xnew, Pnew, inno, K = _KF(xi, P, H, R, y, ypred, radian_sel)
    moti.x = xnew
    moti.P = Pnew

    return inno, K

def rom_private(K: np.ndarray, H: np.ndarray, cor: np.ndarray, cor_len: int):
    """
    Function for updating all inter-robot correlations when a private measurement is made, 
    according to Rom paper
    Args:
        K (np.ndarray): Kalman gains
        H (np.ndarray): 
        cor (np.ndarray): array of inter-robot correlations
        cor_len (int): number of inter-robot correlations
    """
    for i in range(cor_len):
        cor[:,:,i] = (np.eye(K.shape[0]) - K @ H) @ cor[:,:,i]

def luft_relative(Pii_new: np.ndarray, 
                  Pii_old: np.ndarray, 
                  other_id: int,
                  id_list: np.ndarray,
                  cor: np.ndarray, 
                  cor_len: int):
    """
    Function for updating all inter-robot correlations when a private measurement is made, 
    between participating and non-participating robots (sigma_ik).
    Uses the block-diagonal approximation introduced by Luft et al.
    Args:
        Pii_new (np.ndarray): The updated covariance matrix
        Pii_old (np.ndarray): The non-updated covariance matrix
        other_id (int): ID of participating robot. Used for ignoring approximating this correlation, since we already know it exact
        id_list (np.ndarray): 
        cor (np.ndarray): array of inter-robot correlations
        cor_len (int): number of inter-robot correlations
    """
    P_approx = Pii_new @ np.linalg.inv(Pii_old)
    for i in range(cor_len):
        # make sure to ignore the ID of the participating robot:
        if not id_list[i] == other_id:
            cor[:,:,i] = P_approx @ cor[:,:,i]


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
    


#### ML estimators ####

def _pdf_meas_1D(P: np.ndarray, 
        H: np.ndarray, 
        r: float, 
        z: float,
        zpred: float,
        radian_sel: int):
    """
    Calculate the probability for the measurement, to be used in ML.
    This function is only for use for single-variable distributions
    Args:
        P (np.ndarray): Process noise
        H (np.ndarray): Jacobian of output matrix (only for bearing!!)
        r (float): Measured output noise
        y (np.ndarray): Measured output
        radian_sel (int): 0 for normal substraction, 1 for radian substraction
    Returns:
        prob (float): Probability of measurement
    """
    
    # Special substraction for radians:
    if radian_sel == 1:
        diff = ((z - zpred + np.pi) % (2*np.pi)) - np.pi #TODO maybe reuse wrappingpi here
    else:
        diff = z - zpred
    prob = (1/np.sqrt(2*np.pi*r))*np.exp(-(diff**2)/(2*r))
    
    return prob

def _pdf_meas(S: np.ndarray,
        inno: np.ndarray):
    """
    Calculate the probability for the measurement, to be used in ML.
    This function is for multi-variable distributions
    Args:
        S (np.ndarray): 
        inno (np.ndarray): Difference between y and ypred
    Returns:
        prob (float): Probability of measurement
    """
    prob = (1/2*np.pi)*(1/np.sqrt(np.linalg.det(S))) * np.exp(-(1/2)*np.transpose(inno) @ np.linalg.inv(S) @ inno)
    
    return prob

def ML_bearing(b, 
               r, 
               moti: MotionModel, 
               motj: MotionModel,
               measi: MeasModel,
               measj: MeasModel):
    """
    Uses ML to find the most likely bearing, based on the prediction 
    from the motion model 
    """
    P = moti.P
    H = measi.get_jacobian_bearing()
    r_phi = measi.R[Z_PHI, Z_PHI] #TODO: right now, use the same noise for all possible measurements
    zpred_phi = measi.h_bearing(moti.x, motj.x, measj.t)

    z = np.zeros((2,1))
    z_n = b.shape[0]
    ml = 0
    for bi in b:
        p = _pdf_meas_1D(P, H, r_phi, bi, zpred_phi, 1)
        if p > ml:
            ml = p
    z[0] = ml
    z[1] = r

    return z

def ML_rb(b, 
               r, 
               moti: MotionModel, 
               motj: MotionModel,
               measi: MeasModel,
               measj: MeasModel):
    """
    Uses ML to find the most likely range/bearing, based on the prediction 
    from the motion model 
    """
    P = moti.P
    H = measi.get_jacobian_rb(moti.x, motj.x, measj.t)
    zpred = measi.h_rb(moti.x, motj.x, measj.t)
    rad_sel = measi.radian_sel[:Z_W]

    R = measi.R[:Z_W, :Z_W] #TODO: right now, use the same noise for all possible measurements
    S = H @ P @ np.transpose(H) + R

    z = np.zeros((2,1))
    z[1] = r
    b_final = 0
    ml = 0
    for bi in b:
        z[0] = bi
        inno = subtractState(z, zpred, rad_sel)
        # Calculate likelihood
        p = _pdf_meas(S, inno)
        # If larger than previous maximum, then update the bearing to be used
        if p > ml:
            ml = p
            b_final = bi
    # Use the most likely bearing for final measurement
    z[0] = b_final

    return z, ml