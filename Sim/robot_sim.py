from dataclasses import dataclass
import models_functions as mf
import traj 
import numpy as np
import robot_plotter as rp

# Module that features different robot models

class Anchor:
    def __init__(self, x0: np.ndarray,
                 t: np.ndarray = np.array([[0],[0]])
                 ):
        """
        Inits a very simple anchor model
        """
        self.mot = mf.MotionModel(x0=x0)
        self.meas = mf.MeasModel(t=t)

    @property
    def x(self) -> np.ndarray:
        return self.mot.x
    
    @property
    def P(self) -> np.ndarray:
        return self.mot.P
    
    @property
    def t(self) -> np.ndarray:
        return self.meas.t
    
    def draw_position(self, ax, color = 'b'):
        # Qucikly plot the position (without covariance matrix)
        temp = np.concatenate((self.x[mf.X_W:mf.X_W+1,:], self.x[mf.X_P,:]), axis=0)
        rp.plot_position(ax, temp, color=color)

class Robot_single:
    def __init__(self, x0: np.ndarray,
                 path: np.ndarray, 
                 imu: np.ndarray,
                 dt: float = 1.0,
                 P: np.ndarray = np.zeros((mf.STATE_LEN, mf.STATE_LEN)),
                 Q: np.ndarray = np.diag([0.5]*mf.INPUT_LEN),
                 R: np.ndarray = np.diag([0.1]*mf.MEAS_LEN),
                 t: np.ndarray = np.array([[0],[0]])
                 ):
        """"
        Inits the robot model
        Args:
            path (np.ndarray): trajectory to follow
            imu (np.ndarray): imu measurements of path
        """
        self.path = path
        self.imu = imu
        self.p_i = 0 # index of path
        self.p_len = path.shape[1] # number of indicies of path
        self.x_log = np.zeros((mf.STATE_LEN, self.p_len)) # keeps a record of all states
        self.x_log[:,0:1] = x0
        self.P_log = np.zeros((mf.STATE_LEN, mf.STATE_LEN, self.p_len)) # keeps a record of all covariances
        self.P_log[:,:,0] = P

        self.mot = mf.MotionModel(x0 = x0, dt=dt, P=P, Q=Q)
        self.meas = mf.MeasModel(t=t, R=R)
    
    @property
    def x(self) -> np.ndarray:
        return self.mot.x
    
    @property
    def P(self) -> np.ndarray:
        return self.mot.P

    @property
    def t(self) -> np.ndarray:
        return self.meas.t

    def predict(self, imu_correct = True):
        """
        Predict one timestep
        """
        inno = 0
        if self.p_i < self.p_len-1:
            self.mot.predict()
            self.mot.propagate()
            if imu_correct:
                inno = mf.KF_IMU(self.mot, self.meas, self.imu[:,self.p_i:self.p_i+1])

            self.p_i += 1 

            # Log updated quantities:        
            self.x_log[:,self.p_i:self.p_i+1] = self.x
            self.P_log[:,:,self.p_i] = self.P
        else:
            print("End of trajectory!!")

        return inno

    def draw_position(self, ax, color = 'b'):
        # Qucikly plot the position
        rp.plot_position2(ax, self.x, self.P, color=color)

    def anchor_meas(self, a: Anchor, ax = None, sr=0, sb=0):
        # Get ambigious measurement
        y_b, y_r = traj.gen_rb_amb(self.path[0,self.p_i], 
                           a.x[mf.X_THETA, 0], 
                           self.path[1:3, self.p_i:self.p_i+1], 
                           a.x[mf.X_P],
                           self.t,
                           a.t,
                           sr=sr,
                           sb=sb)
        
        # ML to handle front-back ambiguity
        y_rb, _ = mf.ML_rb(y_b, y_r, self.mot, a.mot, self.meas, a.meas)
        inno = mf.KF_rb(self.mot, a.mot, self.meas, a.meas, y_rb)
        if not (ax==None):
            rp.plot_measurement(ax, self.x, a.x)
        return inno
    
    def robot_meas(self, r: 'Robot_single', ax = None):
        """
        Implements naive localization - does not take correlations into account.
        """
        # Get ambigious measurement
        y_b, y_r = traj.gen_rb_amb(self.path[0,self.p_i], 
                           r.x[mf.X_THETA, 0], 
                           self.path[1:3, self.p_i:self.p_i+1], 
                           r.x[mf.X_P],
                           self.t,
                           r.t)
        
        # ML to handle front-back ambiguity
        y_rb, _ = mf.ML_rb(y_b, y_r, self.mot, r.mot, self.meas, r.meas)
        inno = mf.KF_rb_ext(self.mot, r.mot, self.meas, r.meas, y_rb)
        if not (ax==None):
            rp.plot_measurement(ax, self.x, r.x)
        return inno


class robot_rom():
    pass

class robot_luft():
    pass

def MSE_pos(x_true, x_predicted):
    """
    Calculate the mean squared error for orientation and position
    """
    # Extract theta, pos from full state:
    x_predicted2 = x_predicted[[mf.X_THETA, mf.X_P.start, mf.X_P.start+1],:]
    MSE = np.mean(np.square(x_true - x_predicted2), axis=1)
    return MSE