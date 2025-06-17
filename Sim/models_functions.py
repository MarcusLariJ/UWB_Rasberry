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
RB2_LEN = 3
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
Z_PHI2 = 1
Z_R = 1
Z_R2 = 2
Z_W = 2
Z_A = slice(3,5)

# ENUMS for input space:
U_ETAW = 0
U_ETAA = slice(1,3)
U_ETABW = 3
U_ETABA = slice(4,6)

#### Helper functions ####

def normalize_angle(angle: float):
    # Normalize the angle to be within the range -pi to pi
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return angle

def wrappingPi(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ Takes in two arrays of radians and computes the difference when the scale is -pi to pi

    Args:
        x (np.ndarray): First vector
        y (np.ndarray): Second vector

    Returns:
        np.ndarray: Difference of vector
    """
    if x.size:
        return ((x - y + np.pi) % (2*np.pi)) - np.pi
    else:
        return np.zeros_like(x)

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
    def __init__(self, 
                 t: np.ndarray = np.array([[0],[0]]), 
                 R: np.ndarray = np.diag([0.0009, 0.001, 0.0002, 0.004, 0.004])):
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

    def h_rb2(self, xi: np.ndarray, xj: np.ndarray, tj: np.ndarray) -> np.ndarray:
        """ 
        Computes the masurement update when we have access to measurement with other robot
        Uses both AoA and AoD

        Args:
            xi (np.ndarray): The information of the state of the ego robot i at time 'k'
            xj (np.ndarray): The information of the state at the measured robot j at time 'k'
            
        Returns:
            (np.ndarray): The measurement of state xi, xj 
        """
        z = np.zeros((RB2_LEN,1))
        # Precompute q
        thetaj = xj[X_THETA][0]
        thetai = xi[X_THETA][0]
        q = (xj[X_P] + RM(thetaj) @ tj - xi[X_P] - RM(thetai) @ self._t)
        # Range and bearing:
        z[Z_PHI] = np.arctan2(q[1], q[0]) - xi[X_THETA]
        z[Z_PHI2] = np.arctan2(-q[1], -q[0]) - xj[X_THETA]
        z[Z_R2] = np.sqrt(np.transpose(q) @ q)

        return z

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

    def get_jacobian_rb2(self, xi0: np.ndarray, xj0: np.ndarray, tj: np.ndarray) -> np.ndarray:
        """
        Computes the jacobian at time k for the measurement
        Takes into account both the AoA and AoD

        Args:
            xi0 (np.ndarray): The stationary point for the ego bot
            xj0 (np.ndarray): The stationary point for the measurement bot
            tj: (np.ndarray): The offset of the measurement bot

        Returns:
            (np.ndarray): The Jacobian matrix evaluated at xi0, xj0
        """
        # TODO: Check if output H is correct
        H2 = np.zeros((RB2_LEN, STATE_LEN))

        ti = self._t
        thetai = xi0[X_THETA][0]
        thetaj = xj0[X_THETA][0]
        # Precompute q
        q = (xj0[X_P] + RM(thetaj) @ tj - xi0[X_P] - RM(thetai) @ ti)
        # Compute the Jacobian for range/angle measurement:
        # 1st row
        H2[Z_PHI, X_THETA] = (RMdot(thetai)[0,:] @ ti * q[1] - RMdot(thetai)[1,:] @ ti * q[0])/(np.transpose(q) @ q) - 1
        H2[Z_PHI, X_P] = np.array([q[1], -q[0]]).reshape(1,-1) / (np.transpose(q) @ q)
        # 2nd row
        H2[Z_PHI2, X_THETA] = (RMdot(thetai)[0,:] @ ti * q[1] - RMdot(thetai)[1,:] @ ti * q[0])/(np.transpose(q) @ q)
        H2[Z_PHI2, X_P] = np.array([q[1], -q[0]]).reshape(1,-1) / (np.transpose(q) @ q)
        # 3rd row
        H2[Z_R2, X_THETA] = (np.transpose(q) @ (-RMdot(thetai)) @ ti + np.transpose(-RMdot(thetai) @ ti) @ q) / (2*np.sqrt(np.transpose(q) @ q))
        H2[Z_R2, X_P] = -np.transpose(q) / np.sqrt(np.transpose(q) @ q)

        return H2

    def get_jacobian_rb_ext2(self, xi0: np.ndarray, xj0: np.ndarray, tj: np.ndarray) -> np.ndarray:
        """
        Computes the jacobian at time k for the measurement of both robots
        Takes into account both the AoA and AoD

        Args:
            xi0 (np.ndarray): The stationary point for the ego bot
            xj0 (np.ndarray): The stationary point for the measurement bot
            tj: (np.ndarray): The offset of the measurement bot

        Returns:
            (np.ndarray): The Jacobian matrix evaluated at xi0, xj0
        """
        Hij2 = np.zeros((RB2_LEN, STATE_LEN*2))
        ti = self._t
        thetai = xi0[X_THETA][0]
        thetaj = xj0[X_THETA][0]
        # Precompute q
        q = (xj0[X_P] + RM(thetaj) @ tj - xi0[X_P] - RM(thetai) @ ti)
        # Compute the normal jacobian first
        Hij2[:,:STATE_LEN] = self.get_jacobian_rb2(xi0=xi0, xj0=xj0, tj=tj)
        # Then the extended part:
        # 1st row
        Hij2[Z_PHI, X_THETA_EXT] = (-RMdot(thetaj)[0,:] @ tj * q[1] + RMdot(thetaj)[1,:] @ tj * q[0])/(np.transpose(q) @ q)
        Hij2[Z_PHI, X_P_EXT] = np.array([-q[1], q[0]]).reshape(1,-1) / (np.transpose(q) @ q)
        # 2nd row 
        Hij2[Z_PHI2, X_THETA_EXT] = (-RMdot(thetaj)[0,:] @ tj * q[1] + RMdot(thetaj)[1,:] @ tj * q[0])/(np.transpose(q) @ q) - 1
        Hij2[Z_PHI2, X_P_EXT] = np.array([-q[1], q[0]]).reshape(1,-1) / (np.transpose(q) @ q)
        # 3rd row
        Hij2[Z_R2, X_THETA_EXT] = (np.transpose(q) @ (RMdot(thetaj)) @ tj + np.transpose(RMdot(thetaj) @ tj) @ q) / (2*np.sqrt(np.transpose(q) @ q))
        Hij2[Z_R2, X_P_EXT] = np.transpose(q) / np.sqrt(np.transpose(q) @ q)

        return Hij2

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
        # Compute the Jacobian for IMU measurement:
        self.get_jacobian_IMU(xi0)
        # Compute the Jacobian for range/angle measurement:
        self.get_jacobian_rb(xi0, xj0, tj)

        return self._H

class MotionModel:
    """ Base measurement model class
    """
    def __init__(self, dt: float = 1.0,  
                x0: np.ndarray = np.zeros((STATE_LEN, 1)),
                P: np.ndarray = np.diag([0.3, 0.002, 3.0, 3.0, 0.1, 0.1, 0.04, 0.04, 0.0001, 0.1, 0.1]),
                Q: np.ndarray = np.diag([0.1, 8.0, 8.0, 0.000001, 0.00001, 0.00001])):
        """"
        Inits the measurement model
        Args:
            dt float: time difference between k
            P (np.ndarray): State covariance
            Q (np.ndarray): Process noise covariance
        """
        # Initial uncertainty:
        # +-90 on orientation
        # +-8 degrees pr second
        # +- 5 m uncertainty on position
        # +- 1 m/s on velocity
        # +- 0.6 m/s^2 on acceleration
        # +- 1.7 deg/s on rate bias 
        # +- 0.95 m/s^2 on acc bias 
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
    Includes the ML (now MD) step
    Args: 
        x (np.ndarray): States to update
        P (np.ndarray): Process noise to update
        H (np.ndarray): Jacobian of output matrix
        ys (np.ndarray): matrix of possible measured outputs
        radian_sel (np.ndarray): Array indicating which states are given in radians
    """
    S = H @ P @ np.transpose(H) + R
    Sinv = np.linalg.inv(S)
    # Run ML step:
    r_sel = radian_sel
    y, nis = MD_rb_gen(ys, ypred, Sinv, r_sel)
    # filter out unlikely measurements

    # Then carry on with the Kalman Filter
    inno = subtractState(y, ypred, r_sel)
    # Compute gain:
    K = P @ np.transpose(H) @ Sinv
    # Update state estimate
    xnew = x + K @ inno 
    # Update covariance (with Joseph form):
    IKC = (np.eye(P.shape[0]) - K @ H)
    Pnew = IKC @ P @ IKC.T + K @ R @ K.T
    # Return new values:
    return xnew, Pnew, nis[0,0], K

def KF_IMU(mot: MotionModel, meas: MeasModel, y: np.ndarray, thres: float = 0) -> np.ndarray:
    """
    Simple KF for when IMU measurements come in
    Args:
        mot (MotionModel): Motion model of the robot 
        meas (MeasModel): Measurement model of the robot
        y (np.ndarray): Incoming measurement
    Returns:
        nis float: Normalized innovation squared
        K (np.ndarray): Kalman gain vector
        H (np.ndarray): Output matrix
    """
    x = mot.x
    P = mot.P
    H = meas.get_jacobian_IMU(x)
    R = meas.R[Z_W:,Z_W:] # Only use noise related to IMU
    ypred = meas.h_IMU(x)
    radian_sel = meas.radian_sel[Z_W:]
    xnew, Pnew, nis, K = _KF_ml(x, P, H, R, y, ypred, radian_sel) # only for filtering outliers TODO: not very efficient 
    if thres > 0 and nis > thres:
        print("NIS too large. Skipping IMU update")
        return nis, K, H
    mot.x = xnew
    mot.P = Pnew

    return nis, K, H

def KF_rb(moti: MotionModel, 
            xj: MotionModel, 
            measi: MeasModel,
            tj: np.ndarray, 
            ys: np.ndarray,
            thres=0) -> np.ndarray:
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
        nis (np.ndarray): Normalized innovation squared
        K (np.ndarray): Kalman gain vector
        H (np.ndarray):
    """
    xi = moti.x
    P = moti.P
    H = measi.get_jacobian_rb(xi, xj, tj)
    R = measi.R[:Z_W, :Z_W] # Use only noise related to range/bearing:
    ypred = measi.h_rb(xi, xj, tj)
    radian_sel = measi.radian_sel[:Z_W]
    xnew, Pnew, nis, K = _KF_ml(xi, P, H, R, ys, ypred, radian_sel)
    if thres > 0 and nis > thres:
        print("NIS too large. Skipping RB update")
        return nis, K, H
    moti.x = xnew
    moti.P = Pnew

    return nis, K, H

def KF_rb2(moti: MotionModel, 
            xj: MotionModel, 
            measi: MeasModel,
            tj: np.ndarray, 
            ys: np.ndarray,
            thres=0) -> np.ndarray:
    """
    KF for and range/bearing measurement between robot i and anchor j
    Assumes no uncertainty about state vector of j!
    Uses both AoA and AoD measurements
    Args:
        moti (MotionModel): Motion model of the ego robot i 
        motj (MotionModel): Motion model of the other robot j
        measi (MeasModel): Measurement model of the robot i
        measj (MeasModel): Meadurement model of the robot j
        y (np.ndarray): Incoming measurement
    Returns:
        nis (np.ndarray): Normalized innovation squared
        K (np.ndarray): Kalman gain vector
        H (np.ndarray):
    """
    xi = moti.x
    P = moti.P
    H = measi.get_jacobian_rb2(xi, xj, tj)
    ypred = measi.h_rb2(xi, xj, tj)

    radian_sel = np.array([[True],[True],[False]]) #TODO: not the best that this is hardcoded
    R = np.diag([measi.R[Z_PHI, Z_PHI], measi.R[Z_PHI, Z_PHI], measi.R[Z_R, Z_R]]) # TODO: neither is this
    
    xnew, Pnew, nis, K = _KF_ml(xi, P, H, R, ys, ypred, radian_sel)
    if thres > 0 and nis > thres:
        print("NIS too large. Skipping RB2 update")
        return nis, K, H
    moti.x = xnew
    moti.P = Pnew

    return nis, K, H

def KF_rb_ext(moti: MotionModel, 
            motj: MotionModel, 
            measi: MeasModel,
            tj: np.ndarray, 
            ys: np.ndarray,
            thres=0) -> np.ndarray:
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
        nis (np.ndarray): Normalized innovation squared
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
    xnew, Pnew, nis, K = _KF_ml(x, P, H, R, ys, ypred, radian_sel)
    if thres > 0 and nis > thres:
        print("NIS too large. Skipping RB_ext2 update")
        return nis, K, H
    moti.x = xnew[:STATE_LEN,:] # Update xi
    motj.x = xnew[STATE_LEN:,:] # Update xj
    moti.P = Pnew[:STATE_LEN, :STATE_LEN] # Update Pii
    motj.P = Pnew[STATE_LEN:,STATE_LEN:] # Update Pjj

    return nis, K, H

def KF_rb_ext2(moti: MotionModel, 
            motj: MotionModel, 
            measi: MeasModel,
            tj: np.ndarray, 
            ys: np.ndarray,
            thres=0) -> np.ndarray:
    """
    KF for and range/bearing measurement between robot i and j,
    that takes both noise sources into account.
    Used for the naive implementation of collaborative localization
    Uses both AoA and AoD
    Args:
        moti (MotionModel): Motion model of the ego robot i 
        motj (MotionModel): Motion model of the other robot j
        measi (MeasModel): Measurement model of the robot i
        measj (MeasModel): Meadurement model of the robot j
        y (np.ndarray): Incoming measurement
    Returns:
        nis (np.ndarray): Normalized innovation squared
        K (np.ndarray): Kalman gain vector
        H (np.ndarray):
    """
    xi = moti.x
    xj = motj.x
    P = np.zeros((2*STATE_LEN, 2*STATE_LEN))
    P[:STATE_LEN, :STATE_LEN] = moti.P #Pii
    P[STATE_LEN:, STATE_LEN:] = motj.P #Pjj
    H = measi.get_jacobian_rb_ext2(xi, xj, tj)
    ypred = measi.h_rb2(xi, xj, tj)

    radian_sel = np.array([[True],[True],[False]]) #TODO: not the best that this is hardcoded
    R = np.diag([measi.R[Z_PHI, Z_PHI], measi.R[Z_PHI, Z_PHI], measi.R[Z_R, Z_R]]) # TODO: #neither is this    

    # Append both state vectors to get x:
    x = np.append(xi, xj, axis=0)
    xnew, Pnew, nis, K = _KF_ml(x, P, H, R, ys, ypred, radian_sel)
    if thres > 0 and nis > thres:
        print("NIS too large. Skipping RB2_ext update")
        return nis, K, H
    moti.x = xnew[:STATE_LEN,:] # Update xi
    motj.x = xnew[STATE_LEN:,:] # Update xj
    moti.P = Pnew[:STATE_LEN, :STATE_LEN] # Update Pii
    motj.P = Pnew[STATE_LEN:,STATE_LEN:] # Update Pjj

    return nis, K, H

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
               cor_num: int,
               thres: float = 0) -> np.ndarray:
    """
    KF for when IMU measurements come in. 
    Also updates all inter-robot correlations 
    Args:
        mot (MotionModel): Motion model of the robot 
        meas (MeasModel): Measurement model of the robot
        y (np.ndarray): Incoming measurement
    Returns:
        nis (np.ndarray): Normalized innovation squared
        K (np.ndarray): Kalman gain vector
    """
    # Run normal KF IMU 
    nis, K, H = KF_IMU(mot, meas, y, thres=thres)
    if thres > 0 and nis > thres:
        print("NIS too large: Skipping private update")
        return nis, K
    # Update correlations between robots
    rom_private(K, H, cor_list, cor_num)

    return nis, K

def KF_rb_rom(moti: MotionModel, 
            xj: np.ndarray, 
            measi: MeasModel,
            tj: np.ndarray, 
            ys: np.ndarray,
            cor_list: np.ndarray,
            cor_num: int,
            thres=0) -> np.ndarray:
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
    nis, K, H = KF_rb(moti, xj, measi, tj, ys, thres=thres)
    # Then update correlations
    rom_private(K, H, cor_list, cor_num)

    return nis, K

def KF_rb_rom2(moti: MotionModel, 
            xj: np.ndarray, 
            measi: MeasModel,
            tj: np.ndarray, 
            ys: np.ndarray,
            cor_list: np.ndarray,
            cor_num: int,
            thres=0) -> np.ndarray:
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
    nis, K, H = KF_rb2(moti, xj, measi, tj, ys, thres=thres)
    # Then update correlations
    rom_private(K, H, cor_list, cor_num)

    return nis, K

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
            cor_num: int,
            thres=0):
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
    xnew, Pnew, nis, K = _KF_ml(xa, Paa, Ha, R, ys, ypred, rad_sel)
    if thres > 0 and nis > thres:
        print("NIS too large. Skipping DECEN update")
        return xj, Pii, Pii, Pjj, cor_num, nis, K
    # Now, split up the results:
    moti.x = xnew[:STATE_LEN, 0:1]
    xj_new = xnew[STATE_LEN:, 0:1]
    Pii_new = Pnew[:STATE_LEN, :STATE_LEN]
    Pjj_new = Pnew[STATE_LEN:, STATE_LEN:]
    Pij_new = Pnew[:STATE_LEN, STATE_LEN:]
    moti.P = Pii_new
    cor_list[:,:,idx] = Pij_new # update the ij correlation
    return xj_new, Pii_new, Pii, Pjj_new, cor_num, nis, K

def _KF_relative_decen2(moti: MotionModel,  
            measi: MeasModel,
            idj: int,
            xj: np.ndarray,
            Pjj: np.ndarray,
            sigmaji: np.ndarray,
            tj: np.ndarray,
            ys: np.ndarray,
            id_list: dict,
            cor_list: np.ndarray,
            cor_num: int,
            thres=0):
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
    Ha = measi.get_jacobian_rb_ext2(moti.x, xj, tj)
    # Get the argumented state vector
    xa = np.append(moti.x, xj, axis=0)
    # calculate 
    ypred = measi.h_rb2(moti.x, xj, tj)

    rad_sel = np.array([[True],[True],[False]]) #TODO: not the best that this is hardcoded
    R = np.diag([measi.R[Z_PHI, Z_PHI], measi.R[Z_PHI, Z_PHI], measi.R[Z_R, Z_R]]) # TODO: #neither is this

    xnew, Pnew, nis, K = _KF_ml(xa, Paa, Ha, 10*R, ys, ypred, rad_sel) # TODO: multiplying R with 10, to prioritize this measurements lower than anchors
    # Now, split up the results:
    if thres > 0 and nis > thres:
        print("NIS too large. Skipping DECEN2 update")
        return xj, Pii, Pii, Pjj, cor_num, nis, K
    moti.x = xnew[:STATE_LEN, 0:1]
    xj_new = xnew[STATE_LEN:, 0:1]
    Pii_new = Pnew[:STATE_LEN, :STATE_LEN]
    Pjj_new = Pnew[STATE_LEN:, STATE_LEN:]
    Pij_new = Pnew[:STATE_LEN, STATE_LEN:]
    moti.P = Pii_new
    cor_list[:,:,idx] = Pij_new # update the ij correlation
    return xj_new, Pii_new, Pii, Pjj_new, cor_num, nis, K

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
            cor_num: int,
            thres=0,
            meas2=False):
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
    if meas2:
        xj_new, Pii_new, Pii, Pjj_new, cor_num, nis, K = _KF_relative_decen2(moti, 
                                                                            measi, 
                                                                            idj, 
                                                                            xj, 
                                                                            Pjj, 
                                                                            sigmaji, 
                                                                            tj, 
                                                                            ys, 
                                                                            id_list, 
                                                                            cor_list, 
                                                                            cor_num,
                                                                            thres=thres)
    else:
        xj_new, Pii_new, Pii, Pjj_new, cor_num, nis, K = _KF_relative_decen(moti, 
                                                                            measi, 
                                                                            idj, 
                                                                            xj, 
                                                                            Pjj, 
                                                                            sigmaji, 
                                                                            tj, 
                                                                            ys, 
                                                                            id_list, 
                                                                            cor_list, 
                                                                            cor_num,
                                                                            thres=thres)
    # Finally, approximate inter-robot correlations
    if thres > 0 and nis > thres:
        print("Skipping relative update")
        return xj_new, Pjj_new, cor_num, nis, K 
    luft_relative(Pii_new, Pii, idj, id_list, cor_list)
    # xj_new and Pjj_new should be transmitted to the other robot
    return xj_new, Pjj_new, cor_num, nis, K 

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

def inflate_P(mot: MotionModel, cor: np.ndarray, cor_len: int, a: float):
    """
    Scales the covariance and correlations with a factor 'a'.
    Can be used to reset the filter, in case it diverges too much
    """
    mot.P = mot.P*(a**2)
    for i in range(cor_len):
        cor[:,:,i] = a*cor[:,:,i] # the sqrt of the scaling on P
    return

#### ML and MD estimators ####

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
    prob = (1/2*np.pi)*(1/np.sqrt(np.linalg.det(S))) * np.exp(-(1/2)*inno.T @ np.linalg.inv(S) @ inno)
    
    return prob

def _MD_meas(Sinv: np.ndarray,
             inno: np.ndarray):
    """
    Calculate the Mahalanobis distance
    Args:
        Sinv (np.ndarray): The S matrix already inverted
        inno (np.ndarray): Difference between y and ypred
    returns:
        md (float): The Mahalanobis distance
    """

    md = inno.T @ Sinv @ inno

    return md

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
    i_final = -1
    ml = -1
    for i in range(ylen):
        inno = subtractState(ys[:,i:i+1], zpred, rad_sel)
        # Calculate likelihood
        p = _pdf_meas(S, inno)
        # If larger than previous maximum, then update the bearing to be used
        if p > ml:
            ml = p
            i_final = i
    # Debug: Notify us when wrong measurement is used:
    #if not i_final == 0:
    #    print("Wrong measurement used!! Bearing: " + str(ys[0,i_final]) + " was used instead of: " + str(ys[0,0]))
    #if ml == 0.0:
    #    print("Zero! how?")
    # Use the most likely bearing for final measurement
    z = ys[:,i_final:i_final+1]

    return z, ml

def MD_rb_gen(ys,
            zpred,
            Sinv, 
            rad_sel):
    """
    Uses MD to find the most likely range/bearing, based on the prediction 
    from the motion model.
    More general form. Reuses Sinv, zpred calculated for Kalman filtering
    """
    ylen = ys.shape[1]
    i_final = -1
    for i in range(ylen):
        inno = subtractState(ys[:,i:i+1], zpred, rad_sel)
        # Calculate MD
        md = _MD_meas(Sinv, inno)
        if i==0:
            md_min = md
            i_final = 0
        elif md < md_min:
            md_min = md
            i_final = i
    # Debug: Notify us when wrong measurement is used:
    #if not i_final == 0:
    #    print("Wrong measurement used!! Bearing: " + str(ys[0,i_final]) + " was used instead of: " + str(ys[0,0]))
    # Use the most likely bearing for final measurement
    z = ys[:,i_final:i_final+1]

    return z, md_min