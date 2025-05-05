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
    # Update covariance (with Joseph form):
    IKC = (np.eye(P.shape[0]) - K @ H)
    Pnew = IKC @ P @ IKC.T + K @ R @ K.T
    # Return new values:
    return xnew, Pnew, inno, K

def _KF_ml(x: np.ndarray, 
       P: np.ndarray, 
       H: np.ndarray, 
       R: np.ndarray, 
       ys: np.ndarray,
       ypred: np.ndarray,
       radian_sel: np.ndarray):
    """
    Computes the Kalman gain and updates the states and covariance.
    Includes the ML step
    Args: 
        x (np.ndarray): States to update
        P (np.ndarray): Process noise to update
        H (np.ndarray): Jacobian of output matrix
        ys (np.ndarray): matrix of possible measured outputs
        radian_sel (np.ndarray): Array indicating which states are given in radians
    """
    S = H @ P @ np.transpose(H) + R
    # Run ML step:
    r_sel = radian_sel
    y, _ = ML_rb_gen(ys, ypred, S, r_sel)
    # Then carry on with the Kalman Filter
    inno = subtractState(y, ypred, r_sel)
    # Compute gain:
    K = P @ np.transpose(H) @ np.linalg.inv(S)
    # Update state estimate
    xnew = x + K @ inno 
    # Update covariance (with Joseph form):
    IKC = (np.eye(P.shape[0]) - K @ H)
    Pnew = IKC @ P @ IKC.T + K @ R @ K.T
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
        H (np.ndarray): Output matrix
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

    return inno, K, H

def KF_rb(moti: MotionModel, 
            xj: MotionModel, 
            measi: MeasModel,
            tj: np.ndarray, 
            ys: np.ndarray) -> np.ndarray:
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
        H (np.ndarray):
    """
    xi = moti.x
    P = moti.P
    H = measi.get_jacobian_rb(xi, xj, tj)
    R = measi.R[:Z_W, :Z_W] # Use only noise related to range/bearing:
    ypred = measi.h_rb(xi, xj, tj)
    radian_sel = measi.radian_sel[:Z_W]
    xnew, Pnew, inno, K = _KF_ml(xi, P, H, R, ys, ypred, radian_sel)
    moti.x = xnew
    moti.P = Pnew

    return inno, K, H

def KF_rb_ext(moti: MotionModel, 
            motj: MotionModel, 
            measi: MeasModel,
            tj: np.ndarray, 
            ys: np.ndarray) -> np.ndarray:
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
        H (np.ndarray):
    """
    xi = moti.x
    xj = motj.x
    P = np.zeros((2*STATE_LEN, 2*STATE_LEN))
    P[:STATE_LEN, :STATE_LEN] = moti.P #Pii
    P[STATE_LEN:, STATE_LEN:] = motj.P #Pjj
    H = measi.get_jacobian_rb_ext(xi, xj, tj)
    R = measi.R[:Z_W, :Z_W] # Use only noise related to range/bearing:
    ypred = measi.h_rb(xi, xj, tj)
    radian_sel = measi.radian_sel[:Z_W]
    # Append both state vectors to get x:
    x = np.append(xi, xj, axis=0)
    xnew, Pnew, inno, K = _KF_ml(x, P, H, R, ys, ypred, radian_sel)
    moti.x = xnew[:STATE_LEN,:] # Update xi
    motj.x = xnew[STATE_LEN:,:] # Update xj
    moti.P = Pnew[:STATE_LEN, :STATE_LEN] # Update Pii
    motj.P = Pnew[STATE_LEN:,STATE_LEN:] # Update Pjj

    return inno, K, H

def KF_full(moti: MotionModel, 
            motj: MotionModel, 
            measi: MeasModel,
            measj: MeasModel, 
            ys: np.ndarray) -> np.ndarray:
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
    xnew, Pnew, inno, K = _KF_ml(xi, P, H, R, ys, ypred, radian_sel)
    moti.x = xnew
    moti.P = Pnew

    return inno, K


#### KF for collaborative localization ####

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
        cor[:,:,i] = (np.eye(cor.shape[0]) - K @ H) @ cor[:,:,i]

def luft_relative(Pii_new: np.ndarray, 
                  Pii_old: np.ndarray, 
                  other_id: int,
                  id_list: dict,
                  cor: np.ndarray):
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
    # NOTE: P has to be populated before it can be inverted. Dont run a relative measurement before this has happened!
    if len(id_list) > 1:
        # If there is more than 1 entry in the correlation list, 
        # that most mean there are non-participating robots in this measurement
        P_approx = Pii_new @ np.linalg.inv(Pii_old)
        for i in id_list:
            # make sure to ignore the ID of the participating robot:
            if not i == other_id:
                idx = id_list[i]
                cor[:,:,idx] = P_approx @ cor[:,:,idx]
    return

def KF_IMU_rom(mot: MotionModel, 
               meas: MeasModel, 
               y: np.ndarray,
               cor_list: np.ndarray,
               cor_num: int) -> np.ndarray:
    """
    KF for when IMU measurements come in. 
    Also updates all inter-robot correlations 
    Args:
        mot (MotionModel): Motion model of the robot 
        meas (MeasModel): Measurement model of the robot
        y (np.ndarray): Incoming measurement
    Returns:
        inno (np.ndarray): Innovation
        K (np.ndarray): Kalman gain vector
    """
    # Run normal KF IMU 
    inno, K, H = KF_IMU(mot, meas, y)
    # Update correlations between robots
    rom_private(K, H, cor_list, cor_num)

    return inno, K

def KF_rb_rom(moti: MotionModel, 
            xj: np.ndarray, 
            measi: MeasModel,
            tj: np.ndarray, 
            ys: np.ndarray,
            cor_list: np.ndarray,
            cor_num: int) -> np.ndarray:
    """
    KF for and range/bearing measurement between robot i and anchor j
    Assumes no uncertainty about state vector of j!
    Uses Rom method for updating all inter-robot correlations
    Args:
        moti (MotionModel): Motion model of the ego robot i 
        motj (MotionModel): Motion model of the other robot j
        measi (MeasModel): Measurement model of the robot i
        measj (MeasModel): Meadurement model of the robot j
        y (np.ndarray): Incoming measurement
    Returns:
        inno (np.ndarray): Innovation
        K (np.ndarray): Kalman gain vector
        H (np.ndarray):
    """
    # Run normal KF robot to anchor measurement
    inno, K, H = KF_rb(moti, xj, measi, tj, ys)
    # Then update correlations
    rom_private(K, H, cor_list, cor_num)

    return inno, K

def _KF_relative_decen(moti: MotionModel,  
            measi: MeasModel,
            idj: int,
            xj: np.ndarray,
            Pjj: np.ndarray,
            sigmaji: np.ndarray,
            tj: np.ndarray,
            ys: np.ndarray,
            id_list: dict,
            cor_list: np.ndarray,
            cor_num: int):
    # First, check if we have made a measurement to this robot before:
    if idj not in id_list.keys():
        print("Adding new robot: " + str(idj))
        idx = cor_num
        id_list[idj] = idx
        cor_num += 1 # TODO: No limit on growt, but np array has a max! Add limit check here
        Pij = np.zeros((STATE_LEN, STATE_LEN))
    else: 
        idx = id_list[idj]
        Pij = cor_list[:,:,idx] @ sigmaji.T
    # DEBUG:
    #Pij = np.zeros((STATE_LEN, STATE_LEN)) # DEBUG: always set to zero, to see if correlations cause problems
    # Put together Paa matrix:
    Pii = moti.P
    Paa = np.zeros((STATE_LEN*2, STATE_LEN*2))
    Paa[:STATE_LEN, :STATE_LEN] = Pii
    Paa[:STATE_LEN, STATE_LEN:] = Pij
    Paa[STATE_LEN:, :STATE_LEN] = Pij.T
    Paa[STATE_LEN:, STATE_LEN:] = Pjj    
    # Ensure that Paa is PD: (maybe this is only needed on Pii/Pjj)
    Paa = 1/2*(Paa + Paa.T)
    # Get Ha matrix:
    Ha = measi.get_jacobian_rb_ext(moti.x, xj, tj)
    # Get the argumented state vector
    xa = np.append(moti.x, xj, axis=0)
    # calculate 
    ypred = measi.h_rb(moti.x, xj, tj)
    R = measi.R[:Z_W, :Z_W] # Use only noise related to range/bearing
    rad_sel = measi.radian_sel[:Z_W]
    xnew, Pnew, inno, K = _KF_ml(xa, Paa, Ha, R, ys, ypred, rad_sel)
    print(np.linalg.eig(Pnew)[0]) # DEBUG: Check when the matrix is no longer pd
    # Now, split up the results:
    moti.x = xnew[:STATE_LEN, 0:1]
    xj_new = xnew[STATE_LEN:, 0:1]
    Pii_new = Pnew[:STATE_LEN, :STATE_LEN]
    Pjj_new = Pnew[STATE_LEN:, STATE_LEN:]
    Pij_new = Pnew[:STATE_LEN, STATE_LEN:]
    moti.P = Pii_new
    cor_list[:,:,idx] = Pij_new # update the ij correlation
    return xj_new, Pii_new, Pii, Pjj_new, cor_num, inno, K

def KF_relative_luft(moti: MotionModel,  
            measi: MeasModel,
            idj: int,
            xj: np.ndarray,
            Pjj: np.ndarray,
            sigmaji: np.ndarray,
            tj: np.ndarray,
            ys: np.ndarray,
            id_list: dict,
            cor_list: np.ndarray,
            cor_num: int):
    """
    Luft et als algorithm, that approximates inter-robot correlations for non-participating robots
    Args:
        moti (MotionModel): Motion model of the ego robot i 
        measi (MeasModel): Measurement model of the robot i
        idj (int): id of other robot j
        xj (np.ndarray): State vector of other robot j
        Pjj (np.ndarray): Covariance of other robot j
        sigmaji (np.ndarray): correlation to other robot j
        y (np.ndarray): Incoming measurement
        id_list (dict): List of ids
        cor_list (np.ndarray): List of inter-robot correlations
        cor_num (int): length of list 
    Returns:
        xj_new (np.ndarray): New xj
        Pjj_new (np.ndarray): new Pjj. Use to update robot j
        cor_num (int): new length of list
        inno (np.ndarray): Innovation
        K (np.ndarray): Kalman gain vector
    """
    # first, perform the first part of the algorithm:
    xj_new, Pii_new, Pii, Pjj_new, cor_num, inno, K = _KF_relative_decen(moti, 
                                                                        measi, 
                                                                        idj, 
                                                                        xj, 
                                                                        Pjj, 
                                                                        sigmaji, 
                                                                        tj, 
                                                                        ys, 
                                                                        id_list, 
                                                                        cor_list, 
                                                                        cor_num)
    # Finally, approximate inter-robot correlations
    luft_relative(Pii_new, Pii, idj, id_list, cor_list)
    #TODO: check if id_list and cor_list gets updated correctly
    # xj_new and Pjj_new should be transmitted to the other robot
    return xj_new, Pjj_new, cor_num, inno, K 

def KF_relative_rom(moti: MotionModel,  
            measi: MeasModel,
            idj: int,
            xj: np.ndarray,
            Pjj: np.ndarray,
            sigmaji: np.ndarray,
            tj: np.ndarray,
            ys: np.ndarray,
            id_list: dict,
            cor_list: np.ndarray,
            cor_num: int):
    """
    Roumeliotis et als algorithm, that uses the exact inter-robot correlations
    """
    pass

def recieve_meas(moti: MotionModel,
                idj: int,
                xi_new: np.ndarray,
                Pii_new: np.ndarray,
                id_list: dict,
                cor_list: np.ndarray,
                cor_num: int):
    """
    Function for the robot on the recieving end of an measurement
    """
    Pii = moti.P
    moti.x = xi_new
    moti.P = Pii_new
    if idj not in id_list.keys():
        print("ERROR: this id should exist:" + str(idj))
    else: 
        idx = id_list[idj]
    cor_list[:,:,idx] = np.eye(STATE_LEN) # This is due to the chosen decomposition of the correlation
    # Then approximate all inter-robot correlations:
    luft_relative(Pii_new, Pii, idj, id_list, cor_list)

    return # robot should not send anything back at this point

def request_meas(mot: MotionModel, 
                 id_list: dict,
                 cor_list: np.ndarray,
                 cor_num: int, 
                 idj: int):
    """
    Function that should run before the measurement is made
    """
    xi = mot.x
    Pii = mot.P
    # First, check if we have made a measurement to this robot before:
    if idj not in id_list.keys():
        print("Adding new robot: " + str(idj))
        idx = cor_num
        id_list[idj] = idx
        cor_num += 1 # TODO: No limit on growt, but np array has a max! Add limit check here
        sigmaij = np.zeros((STATE_LEN, STATE_LEN))
        cor_list[:,:,idx] = sigmaij # Mostly uncesesarry, since there is already zeros here 
    else: 
        idx = id_list[idj]
        sigmaij = cor_list[:,:,idx]
    
    return xi, Pii, sigmaij, cor_num

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

def ML_rb(ys: np.ndarray, 
               moti: MotionModel, 
               xj: np.ndarray,
               measi: MeasModel,
               tj: np.ndarray):
    """
    Uses ML to find the most likely range/bearing, based on the prediction 
    from the motion model 
    """
    P = moti.P
    H = measi.get_jacobian_rb(moti.x, xj, tj)
    zpred = measi.h_rb(moti.x, xj, tj)
    rad_sel = measi.radian_sel[:Z_W]

    R = measi.R[:Z_W, :Z_W] #TODO: right now, use the same noise for all possible measurements
    S = H @ P @ np.transpose(H) + R

    ylen = ys.shape[1]
    if ylen == 1:
        # if there is only one measurement, then just return
        ml = -1
        return ys, ml
    # Otherwise carry on
    i_final = -1
    ml = 0
    for i in range(ylen):
        inno = subtractState(ys[:,i:i+1], zpred, rad_sel)
        # Calculate likelihood
        p = _pdf_meas(S, inno)
        # If larger than previous maximum, then update the bearing to be used
        if p > ml:
            ml = p
            i_final = i
    # Use the most likely bearing for final measurement
    z = ys[:,i_final:i_final+1]

    return z, ml

def ML_rb_gen(ys,
            zpred,
            S, 
            rad_sel):
    """
    Uses ML to find the most likely range/bearing, based on the prediction 
    from the motion model.
    More general form. Reuses S, zpred calculated for Kalman filtering
    """
    ylen = ys.shape[1]
    if ylen == 1:
        # if there is only one measurement, then just return
        ml = -1
        return ys, ml
    # Otherwise carry on
    i_final = -1
    ml = 0
    for i in range(ylen):
        inno = subtractState(ys[:,i:i+1], zpred, rad_sel)
        # Calculate likelihood
        p = _pdf_meas(S, inno)
        # If larger than previous maximum, then update the bearing to be used
        if p > ml:
            ml = p
            i_final = i
    # Use the most likely bearing for final measurement
    z = ys[:,i_final:i_final+1]

    return z, ml