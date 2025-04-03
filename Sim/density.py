from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
from scipy.signal import cont2discrete


class MeasModel(ABC):
    """ Base measurement model class
    """
    
    @property
    @abstractmethod
    def R(self) -> np.ndarray:
        """ Measurement covariance matrix
        """
        pass
    
    @property
    @abstractmethod
    def nx(self) -> int:
        """ The size of measurement space
        """
        pass
    
    @property
    @abstractmethod
    def nz(self) -> int:
        """ The size of output space
        """
        pass
    
    @property
    @abstractmethod
    def state_signature(self) -> int:
        """ The state signature of the system
        """
        pass
    
    @property
    @abstractmethod
    def output_unit(self) -> np.array:
        """ The output units. Applicable when the dealing with radians.
        """
        pass
    
    @property
    @abstractmethod
    def radian_sel(self) -> np.array:
        """ Selects the outputs with radians as units.
        """
        pass
    
    @property
    @abstractmethod
    def invertOrder(self) -> bool:
        """ Returns whether the order should be reversed, such as for the pin-hole model

        Returns:
            bool: True if it should be inverted
        """
        
    @property
    @abstractmethod
    def under_determined(self) -> bool:
        """ Returns True if the measurement model is under determined
        """
        pass
    
    @abstractmethod
    def h(self, x: np.ndarray, u: np.ndarray, *args: List) -> np.ndarray:
        """ Computes the measurement of state x at time 'k'

        Args:
            x (np.ndarray): The information of the state at time 'k'
            u (np.ndarray): The input at time 'k'
            
        Returns:
            (np.ndarray): The measurement of state x 
        """
        pass
    
    @abstractmethod
    def linearize(self, x: np.ndarray, u: np.ndarray, *args: List) -> Tuple[np.ndarray, np.ndarray]:
        """ Computes evaluation of the Jacobian of the motion model at specific point in state space

        Args:
            x (np.ndarray): The state information
            u (np.ndarray): The input information

        Returns:
            np.ndarray: Returns the system matrix and input matrix in both continuous and discrete time
        """
        H, D = JacobianFD(self.h, self.h(x, u, *args), x, u, self.radian_sel, *args)
        n, nx, nu = D.shape
        return H, D
    
    
    

class MotionModel(ABC):
    """ Base motionmodel class
    """
    
    @property
    @abstractmethod
    def T(self) -> float:
        """ Sampling period
        """
        pass
    
    @property
    @abstractmethod
    def Q(self) -> np.ndarray:
        """ Process noise
        """
        pass
    
    @property
    @abstractmethod
    def nx(self) -> int:
        """ The size of state space
        """
        pass
    
    @property
    @abstractmethod
    def nu(self) -> int:
        """ The input size
        """
        pass
    
    @property
    @abstractmethod
    def nd(self) -> int:
        """ The disturbance size
        """
        pass
    
    @property
    @abstractmethod
    def state_signature(self) -> int:
        """ Selects states that have radians as unit
        """
        pass
    
    @property
    @abstractmethod
    def rad_sel(self) -> int:
        """ Selects states that have radians as unit
        """
        pass

    @abstractmethod
    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """ Computes the prediction of the object from time 'k' to time 'k+1'

        Args:
            x (np.ndarray): The information of the state at time 'k'
            u (np.ndarray): The input and disturbance at time 'k'
            
        Returns:
            (np.ndarray): The predicted state information at time 'k+1'
        """
        pass
    
    @abstractmethod
    def linearize(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Computes evaluation of the Jacobian of the motion model at specific point in state space

        Args:
            x (np.ndarray): The state information
            u (np.ndarray): The input information

        Returns:
            np.ndarray: Returns the system matrix and input matrix in both continuous and discrete time
        """
        A, B = JacobianFD(self.f, self.f(x, u), x, u, self.rad_sel)
        nx, nu_nd = B.shape
        F, G, _, _, _ = cont2discrete((A, B, np.zeros((nx, 0)), np.zeros((0, nu_nd))), self._T)
        B_d = B[:,self.nu:]
        B = B[:,:self.nu]
        G_d = G[:,self.nu:]
        G = G[:,:self.nu]
        return A, B, B_d, F, G, G_d
    
    

def JacobianFD(f: callable, f_eval: np.ndarray, x: np.ndarray, u: np.ndarray,
               rad_sel: np.ndarray, *args: List) -> Tuple[np.ndarray]:
    """ Determines the Jacobians given a function [f] with respect to both [x] and [u]. EX_Amples:
    
            x_dot = f(x, u)
            
            y = h(x, u)
            
        Results in i.e.
        
            x_dot = A * x + B * u
            
            y = H * x + D * u
    
    Args:
        f (callable): The callable function 
        f_eval (np.ndarray): The function evaluated at [x], and [u]
        x (np.ndarray): The state information
        u (np.ndarray): The input
        rad_sel: The elements with unit radian (used to wrap the values between -np.pi to np.pi)

    Returns:
        Tuple[np.ndarray, np.ndarray]: The two Jacobians 
    """
    # Prepare smallest step, determine sizes and initialize Jacobians
    pert = np.sqrt(np.finfo(np.float64).eps)
    n, nx, _ = x.shape
    _, nu, _ = u.shape
    _, nf, _ = f_eval.shape
    A = np.zeros((n, nf, nx))
    B = np.zeros((n, nf, nu))
    # Determine entries of A
    for i in range(nx):
        temp = np.ones((n, 2))
        temp[:,1] = np.abs(x[:,i,0])
        h = (pert*np.max(temp, axis=1))[:,None]
        xh = np.copy(x)
        xh[:,i] = xh[:,i] + h
        h = xh[:,i] - x[:,i]
        f_h_x = f(xh, u, *args)
        f_diff = subtractState(f_h_x, f_eval, rad_sel)
        A[:, :, i] = np.squeeze(f_diff/h[:,None], axis=2)
    # Determine entries of B  
    for j in range(nu):
        temp = np.ones((n, 2))
        temp[:,1] = np.abs(u[:,j,0])
        h = (pert*np.max(temp, axis=1))[:,None]
        uh = np.copy(u)
        uh[:,j] = uh[:,j] + h
        h = uh[:,j] - u[:,j]
        f_h_u = f(x, uh, *args)
        f_diff = subtractState(f_h_u, f_eval, rad_sel)
        B[:, :, j] = np.squeeze(f_diff/h[:,None], axis=2)
    # Return values
    return A, B


    
    

        
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



@dataclass
class GaussianParam:
    x: np.ndarray = np.array([])
    P: np.ndarray = np.array([])
    
    @property
    def n(self):
        return self.x.shape[0]
    
    @property
    def nx(self):
        return self.x.shape[1]
    
    def __getitem__(self, _slice):
        if type(_slice) == int or type(_slice) == np.int64:
            x = self.x[None,_slice, :]
            P = self.P[None, _slice, :, :]
            n = 1
        else:
            x = self.x[_slice,:]
            P = self.P[_slice, :, :]
        return GaussianParam(x, P)
    
    def __setitem__(self, _slice, other: "GaussianParam"):
        if type(_slice) == int or type(_slice) == np.int64:
            self.x[_slice,:] = other.x
            self.P[_slice,:,:] = other.P
        else:
            self.x[_slice,:] = other.x
            self.P[_slice,:,:] = other.P
  
            
    def extend(self, other: "GaussianParam") -> None:
        self.x = np.concatenate((self.x, other.x), axis=0)
        self.P = np.concatenate((self.P, other.P))
        
    def extendByElement(self, x: np.ndarray, P: np.ndarray, opt_info: List) -> None:
        self.x = np.concatenate((self.x, x), axis=0)
        self.P = np.concatenate((self.P, P))
        
    def delete(self, idx: np.ndarray):
        self.x = np.delete(self.x, idx, axis=0)
        self.P = np.delete(self.P, idx, axis=0)




class GaussianDensity:
    
    @staticmethod
    def eX_Pected_value(state: GaussianParam) -> np.array:
        return state.x
    
    @staticmethod
    def covariance(state: GaussianParam) -> np.array:
        return state.P
    
    @staticmethod
    def predict(state: GaussianParam, u: np.ndarray, motion_model: MotionModel) -> GaussianParam:
        """Kalman prediction

        Args:
            state (State): State structure containing information about posterior from previous time step
            u (np.ndarray): Input information
            d (np.ndarray): Disturbance information 
            motionmodel (ConstantVelocity): Motionmodel describing transition density

        Returns:
            State: Prior in update step
        """
        _, _, _, F, G, G_d = motion_model.linearize(state.x, u)
        P_kkm1 = F @ state.P @ np.transpose(F, axes=(0,2,1))
        state_pred = GaussianParam(
                           motion_model.f(state.x, u),
                           P_kkm1 + motion_model.Q
                          )
        return state_pred, P_kkm1
    
    @staticmethod
    def computePredictedPriorParameters(state_pred: GaussianParam, meas_model: MeasModel,
                                        meas: np.ndarray, *args):
            H, _ = meas_model.linearize(state_pred.x, np.zeros((0, 0)), *args)
            S = (H @ state_pred.P @ np.transpose(H, axes=(0,2,1))) + meas_model.R
            # S = (S + S.T)/2
            z_hat = meas_model.h(state_pred.x, np.array([]), *args)
            S_inv = np.linalg.inv(S)
            return z_hat, S, S_inv, H
        
    @staticmethod
    def update(state_pred: GaussianParam, meas: np.ndarray, meas_model: MeasModel,
               z_pred: np.ndarray, S_inv: np.ndarray, H: np.ndarray, *args) -> GaussianParam:
        K = (state_pred.P @ np.transpose(H, axes=(0,2,1))) @ S_inv
        # Handle states with radians as units
        r_sel = meas_model.radian_sel
        z_diff = subtractState(meas, z_pred, r_sel)
        x_upd = state_pred.x + K @ z_diff
        P_upd = (np.eye(state_pred.nx) - K @ H) @ state_pred.P
        
        state_upd = GaussianParam(x_upd, P_upd)
        return state_upd
