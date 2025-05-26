from dataclasses import dataclass
import models_functions as mf
import traj 
import numpy as np
import robot_plotter as rp
from scipy.stats.distributions import chi2

# Module that features different robot models
# and error metrics

class Anchor:
    def __init__(self, x0: np.ndarray,
                 t: np.ndarray = np.array([[0],[0]]),
                 id: int = 0
                 ):
        """
        Inits a very simple anchor model
        """
        self.mot = mf.MotionModel(x0=x0)
        self.meas = mf.MeasModel(t=t)
        self.id = id

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
        temp = np.concatenate((self.x[mf.X_THETA:mf.X_THETA+1,:], self.x[mf.X_P,:]), axis=0)
        rp.plot_position(ax, temp, color=color, draw_arrow=False, marker='x')

class Robot_single:
    def __init__(self, x0: np.ndarray,
                 path: np.ndarray, 
                 imu: np.ndarray,
                 id: int, 
                 dt: float = 1.0,
                 P: np.ndarray = np.diag([0.3, 0.002, 3.0, 3.0, 0.1, 0.1, 0.04, 0.04, 0.0001, 0.1, 0.1]),
                 Q: np.ndarray = np.diag([0.1, 8.0, 8.0, 0.000001, 0.00001, 0.00001]),
                 R: np.ndarray = np.diag([0.0009, 0.001, 0.0002, 0.004, 0.004]),
                 t: np.ndarray = np.array([[0],[0]])
                 ):
        """"
        Inits the robot model
        Args:
            path (np.ndarray): trajectory to follow
            imu (np.ndarray): imu measurements of path
        """
        self.id = id
        self.path = path
        self.imu = imu
        self.p_i = 0 # index of path
        self.p_len = path.shape[1] # number of indicies of path
        self.x_log = np.zeros((mf.STATE_LEN, self.p_len)) # keeps a record of all states
        self.x_log[:,0:1] = x0
        self.P_log = np.zeros((mf.STATE_LEN, mf.STATE_LEN, self.p_len)) # keeps a record of all covariances
        self.P_log[:,:,0] = P
        self.dt = dt
        self.nis_IMU = np.zeros(self.p_len) # Keeps all recoreded IMU NIS values
        self.nis_rb = np.zeros(self.p_len) # Keeps all recorded RB NIS
        self.rb_ids = np.zeros(self.p_len) # Keeps track of the robot/anchors we communicated with

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

    def predict(self, imu_correct = True, thres=0):
        """
        Predict one timestep
        """
        nis = 0
        if self.p_i < self.p_len-1:
            self.mot.predict()
            self.mot.propagate()
            if imu_correct:
                nis, _, _ = mf.KF_IMU(self.mot, self.meas, self.imu[:,self.p_i:self.p_i+1], thres=thres)

            self.p_i += 1 
        else:
            print("End of trajectory!!")
            return nis

        # Log updated quantities:        
        self.x_log[:,self.p_i:self.p_i+1] = self.x
        self.P_log[:,:,self.p_i] = self.P
        self.nis_IMU[self.p_i - 1] = nis # -1, because it was due to IMU meas at that point

        return nis

    def draw_position(self, ax, color = 'b'):
        # Qucikly plot the position
        rp.plot_position2(ax, self.x, self.P, color=color)

    def anchor_meas(self, a: Anchor, ax = None, sr=0, sb=0):
        # Get ambigious measurement
        ys = traj.gen_rb_amb(self.path[0,self.p_i], 
                           a.x[mf.X_THETA, 0], 
                           self.path[1:3, self.p_i:self.p_i+1], 
                           a.x[mf.X_P],
                           self.t,
                           a.t,
                           sr=sr,
                           sb=sb)
        
        # ML to handle front-back ambiguity
        #y_rb, _ = mf.ML_rb(ys, self.mot, a.mot.x, self.meas, a.meas.t)
        inno, _, _ = mf.KF_rb(self.mot, a.mot.x, self.meas, a.meas.t, ys)
        if not (ax==None):
            rp.plot_measurement(ax, self.x, a.x)
        # Log updated quantities:        
        self.x_log[:,self.p_i:self.p_i+1] = self.x
        self.P_log[:,:,self.p_i] = self.P
        return inno
    
    def robot_meas(self, r: 'Robot_single', ax = None, sr=0, sb=0):
        """
        Implements naive localization - does not take correlations into account.
        But updates xi, xj, Pii and Pjj
        """
        # Get ambigious measurement
        ys = traj.gen_rb_amb(self.path[0,self.p_i], 
                           r.path[0,r.p_i], 
                           self.path[1:3, self.p_i:self.p_i+1], 
                           r.path[1:3, r.p_i:r.p_i+1],
                           self.t,
                           r.t,
                           sr=sr,
                           sb=sb)
        
        # ML to handle front-back ambiguity
        #y_rb, _ = mf.ML_rb(ys, self.mot, r.mot.x, self.meas, r.meas.t)
        inno, _, _ = mf.KF_rb_ext(self.mot, r.mot, self.meas, r.meas.t, ys)
        if not (ax==None):
            rp.plot_measurement(ax, self.x, r.x)
        # Log updated quantities:        
        self.x_log[:,self.p_i:self.p_i+1] = self.x
        self.P_log[:,:,self.p_i] = self.P

        # TODO: update logs of other robots
        r.x_log[:,r.p_i:r.p_i+1] = r.x
        r.P_log[:,:,r.p_i] = r.P

        return inno


class robot_luft(Robot_single):
    def __init__(self, x0: np.ndarray,
                 path: np.ndarray, 
                 imu: np.ndarray,
                 id: int,
                 dt: float = 1.0,
                 P: np.ndarray = np.diag([0.3, 0.002, 3.0, 3.0, 0.1, 0.1, 0.04, 0.04, 0.0001, 0.1, 0.1]),
                 Q: np.ndarray = np.diag([0.1, 8.0, 8.0, 0.000001, 0.00001, 0.00001]),
                 R: np.ndarray = np.diag([0.0009, 0.001, 0.0002, 0.004, 0.004]),
                 t: np.ndarray = np.array([[0],[0]])
                 ):
        """"
        Inits the robot model, used for lufts implementation of collaborative localization
        Args:
            path (np.ndarray): trajectory to follow
            imu (np.ndarray): imu measurements of path
        """
        super().__init__(x0=x0, path=path, imu=imu, dt=dt, P=P, Q=Q, R=R, t=t, id=id)
        # Setup list of robots we have met
        self.id_len = 10 # Allocate memory for 10 robots
        self.id_num = 0 # current length of dictionary
        self.id_list = {}
        self.s_list = np.zeros((mf.STATE_LEN, mf.STATE_LEN, self.id_len)) # list for interrobot correleations (sigmaij)

    def predict(self, imu_correct=True, thres=0):
        """
        Predict one timestep
        """
        inno = 0
        if self.p_i < self.p_len-1:
            self.mot.predict()
            self.mot.propagate_rom(self.s_list, self.id_num) #<--- notice rom function here
            if imu_correct:
                nis, _= mf.KF_IMU_rom(self.mot, self.meas, self.imu[:,self.p_i:self.p_i+1], self.s_list, self.id_num, thres=thres)

            self.p_i += 1 
        else:
            print("End of trajectory!!")
            return nis

        # Log updated quantities:        
        self.x_log[:,self.p_i:self.p_i+1] = self.x
        self.P_log[:,:,self.p_i] = self.P
        self.nis_IMU[self.p_i - 1] = nis # -1, because it was due to IMU meas at that point

        return nis
    
    def anchor_meas(self, a: Anchor, sr=0, sb=0, thres=0, max_dist=-1, amb=True, pout_r=0, pout_b=0):
        """
        Make measurement to anchor and use Rom methods for updating correlations
        """
        # Get ambigious measurement
        ys = traj.gen_rb_amb(self.path[0,self.p_i], 
                           a.x[mf.X_THETA, 0], 
                           self.path[1:3, self.p_i:self.p_i+1], 
                           a.x[mf.X_P],
                           self.t,
                           a.t,
                           sr=sr,
                           sb=sb,
                           amb=amb,
                           pout_r=pout_r,
                           pout_b=pout_b)
        
        if (max_dist > 0 and ys[1,0] > max_dist):
            print("Anchor " + str(a.id) + " out of range for robot " + str(self.id) + " at time " + str(self.p_i*self.dt))
            return None
        # Else: anchor within range:
        print("Robot " + str(self.id) + " sees anchor " + str(a.id) + " at time " + str(self.p_i*self.dt))
        nis, _ = mf.KF_rb_rom(self.mot, a.mot.x, self.meas, a.meas.t, ys, self.s_list, self.id_num, thres=thres)
        # Log updated quantities:        
        self.x_log[:,self.p_i:self.p_i+1] = self.x
        self.P_log[:,:,self.p_i] = self.P
        self.nis_rb[self.p_i] = nis
        self.rb_ids[self.p_i] = a.id
        return nis
    
    def anchor_meas2(self, a: Anchor, sr=0, sb=0, thres=0, max_dist=-1, amb=True, pout_r=0, pout_b=0):
        """
        Make measurement to anchor and use Rom methods for updating correlations
        """
        # Get ambigious measurement
        ys = traj.gen_rb2_amb(self.path[0,self.p_i], 
                           a.x[mf.X_THETA, 0], 
                           self.path[1:3, self.p_i:self.p_i+1], 
                           a.x[mf.X_P],
                           self.t,
                           a.t,
                           sr=sr,
                           sb=sb,
                           amb=amb,
                           pout_r=pout_r,
                           pout_b=pout_b)
        
        if (max_dist > 0 and ys[1,0] > max_dist):
            print("Anchor " + str(a.id) + " out of range for robot " + str(self.id) + " at time " + str(self.p_i*self.dt))
            return None
        # Else: anchor within range:
        print("Robot " + str(self.id) + " sees anchor " + str(a.id) + " at time " + str(self.p_i*self.dt))
        nis, _ = mf.KF_rb_rom2(self.mot, a.mot.x, self.meas, a.meas.t, ys, self.s_list, self.id_num, thres=thres)
        # Log updated quantities:        
        self.x_log[:,self.p_i:self.p_i+1] = self.x
        self.P_log[:,:,self.p_i] = self.P
        self.nis_rb[self.p_i] = nis
        self.rb_ids[self.p_i] = a.id
        return nis

    def robot_meas_luft(self, r: 'robot_luft', sr=0, sb=0, thres=0, max_dist=-1, amb=True, pout_r=0, pout_b=0):
        """
        Implements Lufts et al algorithm for CL localization
        """
        # Get ambigious measurement
        ys = traj.gen_rb_amb(self.path[0,self.p_i], 
                           r.path[0,r.p_i], 
                           self.path[1:3, self.p_i:self.p_i+1], 
                           r.path[1:3, r.p_i:r.p_i+1],
                           self.t,
                           r.t,
                           sr=sr,
                           sb=sb,
                           amb=amb,
                           pout_r=pout_r,
                           pout_b=pout_b)
        if (max_dist > 0 and ys[1,0] > max_dist):
            print("Robot " + str(r.id) + " out of range for robot " + str(self.id) + " at time " + str(self.p_i*self.dt))
            return None
        # Else: robot within range:
        print("Robot " + str(self.id) + " sees robot " + str(r.id) + " at time " + str(self.p_i*self.dt))
        # Request quantities from other robot:
        xj, Pjj, sigmaji, idj, tj = r.send_requested(self.id)
        # KF update
        xj_new, Pjj_new, cor_num, nis, _ = mf.KF_relative_luft(self.mot, 
                                                                self.meas, 
                                                                idj, 
                                                                xj,
                                                                Pjj,
                                                                sigmaji, 
                                                                tj,
                                                                ys, 
                                                                self.id_list,
                                                                self.s_list,
                                                                self.id_num,
                                                                thres=thres)
        self.id_num = cor_num
        # send updated quantities back to j
        r.recieve_update(xj_new, Pjj_new, self.id)

        # Log updated quantities:        
        self.x_log[:,self.p_i:self.p_i+1] = self.x
        self.P_log[:,:,self.p_i] = self.P
        self.nis_rb[self.p_i] = nis
        self.rb_ids[self.p_i] = r.id
        return nis
    
    def robot_meas_luft2(self, r: 'robot_luft', sr=0, sb=0, thres=0, max_dist=-1, amb=True, pout_r=0, pout_b=0):
        """
        Implements Lufts et al algorithm for CL localization
        """
        # Get ambigious measurement
        ys = traj.gen_rb2_amb(self.path[0,self.p_i], 
                           r.path[0,r.p_i], 
                           self.path[1:3, self.p_i:self.p_i+1], 
                           r.path[1:3, r.p_i:r.p_i+1],
                           self.t,
                           r.t,
                           sr=sr,
                           sb=sb,
                           amb=amb,
                           pout_r=pout_r,
                           pout_b=pout_b)
        if (max_dist > 0 and ys[1,0] > max_dist):
            print("Robot " + str(r.id) + " out of range for robot " + str(self.id) + " at time " + str(self.p_i*self.dt))
            return None
        # Else: robot within range:
        print("Robot " + str(self.id) + " sees robot " + str(r.id) + " at time " + str(self.p_i*self.dt))
        # Request quantities from other robot:
        xj, Pjj, sigmaji, idj, tj = r.send_requested(self.id)
        # KF update
        xj_new, Pjj_new, cor_num, nis, _ = mf.KF_relative_luft2(self.mot, 
                                                                self.meas, 
                                                                idj, 
                                                                xj,
                                                                Pjj,
                                                                sigmaji, 
                                                                tj,
                                                                ys, 
                                                                self.id_list,
                                                                self.s_list,
                                                                self.id_num,
                                                                thres=thres)
        self.id_num = cor_num
        # send updated quantities back to j
        r.recieve_update(xj_new, Pjj_new, self.id)

        # Log updated quantities:        
        self.x_log[:,self.p_i:self.p_i+1] = self.x
        self.P_log[:,:,self.p_i] = self.P
        self.nis_rb[self.p_i] = nis
        self.rb_ids[self.p_i] = r.id
        return nis

    def robot_meas_rom(self, raa: 'robot_luft', rbb: list, ax = None, sr=0, sb=0):
        """
        Function that implements Rom's decentralized CL.
        Args:
            raa (robot_luft): The other robot participating in the exchange
            rbb (list): List of robots not participating in the exchange
            ax (Axes): plot measurmenet 
            sr (float): noise on range
            sb (float): noise on bearing
        """
        pass

    def recieve_update(self, xnew, Pnew, idj):
        """
        Update own values after a succesfull measurement
        """
        mf.recieve_meas(self.mot, idj, xnew, Pnew, self.id_list, self.s_list, self.id_num)
        # TODO: log new updated values here!
        self.x_log[:,self.p_i:self.p_i+1] = self.x
        self.P_log[:,:,self.p_i] = self.P

    def send_requested(self, idj):
        """
        Send requested values, when initially making a measurement
        """
        xi, Pii, sigmaij, cor_num = mf.request_meas(self.mot, self.id_list, self.s_list, self.id_num, idj)
        self.id_num = cor_num
        idi = self.id
        ti = self.t
        return xi, Pii, sigmaij, idi, ti



########### Error functions ###############


def RMSE(x_true, x_predicted, rad_sel):
    """
    Calculate the mean squared error for orientation and position
    """
    x_len = x_predicted.shape[0]
    x_num = x_predicted.shape[1]
    N = x_predicted.shape[2]
    rmse = np.zeros((x_len, x_num))
    #square and sum
    for i in range(N):
        rmse += mf.subtractState(x_true, x_predicted[:,:,i], np.repeat(rad_sel, x_num, axis=1))**2
    # normlize and root
    rmse = np.sqrt(1/N*rmse)    
    return rmse

def abs_error(x_true, x_pred, rad_sel):
    """
    calculates the absolute error
    For assessing the absolute performance
    Does not handle multi-dimensional vectors!
    """
    x_num = x_pred.shape[1]
    abs_e = np.abs(mf.subtractState(x_true, x_pred, np.repeat(rad_sel, x_num, axis=1)))
    return abs_e

def error_bp(r: Robot_single, bias):
    """
    Wrapper that handles position + bias error 
    """
    x_true = r.path
    x_pred = r.x_log
    abs_e_idx = np.array([mf.X_THETA, mf.X_P.start, mf.X_P.start+1, mf.X_BW, mf.X_BA.start, mf.X_BA.start+1])

    # Extract only the quantities we are interested in
    x_pred_bp = x_pred[abs_e_idx, :]
    x_num = x_pred_bp.shape[1]
    # Append biases to true
    x_true_bp = np.append(x_true, np.repeat(bias, x_num, axis=1), axis=0)
    # Then calculate error:
    rad_sel = np.array([[True],[False],[False],[False],[False],[False]])
    e_bp = abs_error(x_true_bp, x_pred_bp, rad_sel)
    
    return e_bp

def NEES(x_est: np.ndarray, 
        x_true: np.ndarray, 
        P: np.ndarray, 
        rad_sel: np.ndarray, 
        prob=0.95, 
        dt=1):
    """
    Calculate the NEES (normalized estimation error squared) over time
    TODO: NEES of bearing acts strange at times, make sure that it is calculated correct 
    """
    x_len = x_est.shape[0]
    x_num = x_est.shape[1]
    P_num = P.shape[2]
    x = mf.subtractState(x_true, x_est, np.repeat(rad_sel, x_num, axis=1))
    nees = np.zeros(x_num)
    a = 1-prob
    if x_num == P_num:
        t = np.linspace(0, x_num*dt, x_num)
        check_inv = True
        for i in range(x_num):
            if check_inv:
                # Initially, the P matrix is not invertible (fx all 0s). 
                # Make a check just in case for the first few cases:
                if np.linalg.det(P[:,:,i]) == 0:
                    nees[i] = -1
                    continue
                else:
                    check_inv = False 
            nees[i] = x[:,i:i+1].T @ np.linalg.inv(P[:,:,i]) @ x[:,i:i+1]
        # Calculate thresholds:
        r1 = chi2.ppf(a/2.0, df=x_len)
        r2 = chi2.ppf(1-a/2.0, df=x_len)

        return nees, t, r1, r2
    else:
        print("Number of x did not match number of P")
        return -1, -1, -1, -1

def ANIS(nis,
        df,
        dt=1,
        prob = 95):
    """
    Calculate ANIS
    Note: degrees of freedom (df) has to be set manually. 
    three for IMU, two for r/b
    """
    anis = np.zeros_like(nis[:,0])
    x_num = nis.shape[0]
    N = nis.shape[1]
    t = np.linspace(0, x_num*dt, x_num)
    for i in range(N):
        anis += nis[:,i]
    anis = 1.0/N*anis
    # Plot anis:
    a = 1-prob
    r1 = 1.0/N*chi2.ppf(a/2.0, df=N*df)
    r2 = 1.0/N*chi2.ppf(1-a/2.0, df=N*df)
    
    return anis, t, r1, r2

def ANEES(x_est: np.ndarray, 
        x_true: np.ndarray, 
        P: np.ndarray,
        rad_sel, 
        prob=0.95, 
        dt=1, ):
    """
    Calculate ANEES
    """
    x_len = x_est.shape[0]
    x_num = x_est.shape[1]
    N = x_est.shape[2] # number of independent samples
    anees = np.zeros(x_num)
    a = 1-prob
    for i in range(N):
        # Add all nees up together:
        temp, t, _, _ = NEES(x_est[:,:,i], x_true, P[:,:,:,i], rad_sel=rad_sel, prob=prob, dt=dt)
        anees += temp
    anees = 1.0/N*anees
    # Calculate thresholds:
    r1 = 1.0/N*chi2.ppf(a/2.0, df=N*x_len)
    r2 = 1.0/N*chi2.ppf(1-a/2.0, df=N*x_len)

    return anees, t, r1, r2

    