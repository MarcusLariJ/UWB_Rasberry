import numpy as np
import models_functions as mf

# File that contains various example trajectories
# and other ways to generate measurement data

def gen_circ_traj(R, N, dt=1, theta0=0, xc=0, yc=0, ccw = False):
    # TODO: delete
    # Angle values for each position
    theta = np.linspace(0 + theta0, 2 * np.pi + theta0, N)
    if (ccw):
        # Go in opposite direction
        theta = theta[::-1]
    
    # Speed value
    w = (theta[1] - theta[0])/dt

    # Initialize the 3 x N position matrix
    trajectory = np.zeros((3, N))

    # Top row: orientation (angle)
    if (not ccw):
        trajectory[0, :] = theta + np.pi/2
    else:
        trajectory[0, :] = theta - np.pi/2
    
    # Bottom rows: x and y positions (circular trajectory)
    trajectory[1, :] = R * np.cos(theta) + xc 
    trajectory[2, :] = R * np.sin(theta) + yc 

    # IMU measuerement matrix:
    IMU_meas = np.zeros((mf.IMU_LEN, N))
    IMU_meas[0, :] = np.ones((1, N))*w
    if (not ccw):
        IMU_meas[2, :] = np.ones((1, N))*R*(w**2)
    else: 
        IMU_meas[2, :] = -np.ones((1, N))*R*(w**2)

    # Starting conditions:
    x0 = np.zeros((mf.STATE_LEN, 1))
    x0[mf.X_THETA] = trajectory[0,0]
    x0[mf.X_W] = w
    x0[mf.X_P] = trajectory[1:,0:1]
    x0[mf.X_V] = R*w*np.array([[-np.sin(theta[0])], [np.cos(theta[0])]])
    x0[mf.X_A] = R*(w**2)*np.array([[-np.cos(theta[0])],[-np.sin(theta[0])]])

    return trajectory, IMU_meas, x0

def gen_circ_traj_norot(R, N, dt=1, theta0=0, xc=0, yc=0, ccw = False):
    # TODO: delete
    # Angle values for each position
    theta = np.linspace(0 + theta0, 2 * np.pi + theta0, N)
    if (ccw):
        # Go in opposite direction
        theta = theta[::-1]
    
    # Speed value
    w = (theta[1] - theta[0])/dt

    # Initialize the 3 x N position matrix
    trajectory = np.zeros((3, N))

    # Top row: orientation (angle)
    # They are just zero.
    
    # Bottom rows: x and y positions (circular trajectory)
    trajectory[1, :] = R * np.cos(theta) + xc 
    trajectory[2, :] = R * np.sin(theta) + yc 

    # IMU measuerement matrix:
    IMU_meas = np.zeros((mf.IMU_LEN, N))
    
    # angular speed is zero

    IMU_meas[1,:] = -R*(w**2) * np.cos(theta)
    IMU_meas[2,:] = -R*(w**2) * np.sin(theta)
    
    # Starting conditions:
    x0 = np.zeros((mf.STATE_LEN, 1))
    x0[mf.X_THETA] = trajectory[0,0]
    x0[mf.X_W] = 0
    x0[mf.X_P] = trajectory[1:,0:1]
    x0[mf.X_V] = R*w*np.array([[-np.sin(theta[0])], [np.cos(theta[0])]])
    x0[mf.X_A] = R*(w**2)*np.array([[-np.cos(theta[0])],[-np.sin(theta[0])]])

    return trajectory, IMU_meas, x0

### Functions for generating noisy trajectories

def gen_poly_traj(  x0: np.ndarray, 
                    xf: np.ndarray, 
                    t0: float, 
                    tf: float, 
                    dt: float):
    """
    Generates a quintic polynomial trajectory between x0 and xf between time t0 and tf
    for the position, velocity and acceleration and a cubic polynomial for angular velocity 
    and orientation
    Args:
        x0 (np.ndarray): Initial state
        xf (np.ndarray): Final state
        t0 (float): Start time
        tf (float): End time
        dt (float): Sampling time 
    Returns:
        Trajectory (np.ndarray): Theta and position trajectory
        IMU_meas (np.ndarray): Angular rate and body acceleration
    """    
    num_samples = (1/dt)*(tf-t0)

    # First generate trajectory for position using quintic poly:

    tqM = np.array([   [1, t0, t0**2, t0**3, t0**4, t0**5],
                        [0, 1, 2*t0, 3*t0**2, 4*t0**3, 5*t0**4],
                        [0, 0, 2, 6*t0, 12*t0**2, 20*t0**3],
                        [1, tf, tf**2, tf**3, tf**4, tf**5],
                        [0, 1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4],
                        [0, 0, 2, 6*tf, 12*tf**2, 20*tf**3]
                ])
    
    p0f_x = np.concatenate((x0[2:7:2], xf[2:7:2]), axis=0)
    p0f_y = np.concatenate((x0[3:8:2], xf[3:8:2]), axis=0)

    coeff_px = np.linalg.inv(tqM) @ p0f_x
    coeff_py = np.linalg.inv(tqM) @ p0f_y

    # Then generate cubic trajectory:
    pcM = np.array([   [1, t0, t0**2,  t0**3   ],
                        [0, 1,  2*t0,   3*t0**2 ],
                        [1, tf, tf**2,  tf**3   ],
                        [0, 1,  2*tf,   3*tf**2 ]
                     ]) 
    
    t0f = np.concatenate((x0[:2], xf[:2]), axis=0)

    coeff_t = np.linalg.inv(pcM) @ t0f

    # Setup trajectories:
    tvec = np.linspace(t0, tf, num=round(num_samples)+1)
    time_poly_theta = np.transpose(np.c_[tvec**0, tvec, tvec**2, tvec**3])
    time_poly_w = np.transpose(np.c_[tvec*0, tvec**0, 2*tvec**1, 3*tvec**2])

    time_poly_p = np.transpose(np.c_[tvec**0, tvec, tvec**2, tvec**3, tvec**4, tvec**5])
    time_poly_a = np.transpose(np.c_[tvec*0, tvec*0, 2*tvec**0, 6*tvec**1, 12*tvec**2, 20*tvec**3])

    theta_traj =    coeff_t.T @ time_poly_theta
    w_traj =        coeff_t.T @ time_poly_w
    px_traj =       coeff_px.T @ time_poly_p
    py_traj =       coeff_py.T @ time_poly_p
    ax_traj =       coeff_px.T @ time_poly_a
    ay_traj =       coeff_py.T @ time_poly_a

    # Convert inertial frame accelaration to body acceleration
    a_traj = np.concatenate((ax_traj, ay_traj), axis=0)
    for i in range(a_traj.shape[1]):
        a_traj[:,i:i+1] = mf.RM( -theta_traj[0,i] ) @ a_traj[:,i:i+1]

    # Combine all the trajectories and return them
    IMU_meas = np.concatenate((w_traj, a_traj), axis=0)
    trajectory = np.concatenate((theta_traj, px_traj, py_traj), axis=0)

    return trajectory, IMU_meas

def gen_poly_traj_multi(pos: list, time: list, dt):
    """
    Generate multiple trajectories and concatenate them
    """
    trajs = []
    imu = []
    for i in range(len(pos)-1):
        temptraj, tempimu = gen_poly_traj(pos[i], pos[i+1], t0=time[i], tf=time[i+1], dt=dt)
        trajs += [temptraj]
        imu += [tempimu]
    # Now concatenate everything:
    trajectory = trajs[0]
    IMU_meas = imu[0]
    for i in range(len(imu)-1):
        trajectory = np.append(trajectory[:,:-1], trajs[i+1], axis=1) # Remember to remove repeated position!!
        IMU_meas = np.append(IMU_meas[:,:-1], imu[i+1], axis=1)
    return trajectory, IMU_meas

def gen_noise(y: np.ndarray, bias: np.ndarray = None, sigma: np.ndarray = None, out_freq: np.ndarray = None, dt: float = 1.0):
    """
    Applies noise to the passed measurement
    Args:
        y (np.ndarray): The measurement(s) y_size X y_len 
        bias (np.ndarray): A vector of biases y_size X 1
        sigma (np.ndarray): Covariance matrix y_size X y_size
        out_freq (np.ndarray): How many outliers there are pr second on average. Requires noise is present. Vector of size y_size X 1 
        dt (float): Sampling time
    """
    y_size, y_len = y.shape
    if not (sigma is None):
        # Convert variance to std deviation
        noise_v = np.sqrt(sigma) @ np.random.randn(y_size, y_len)
        if not (out_freq is None):
            # Add outliers to the dataset
            out_num = np.round(out_freq*dt*y_len).astype(int) # get total number of outlier (this is deterministic - might want to randomize this a bit)
            for i in range(y_size):
                outlier_indices = np.random.choice(y_len, size=out_num[i,0], replace=False)
                # Multiply chosen outlier positions with 5, to increase noise to 5 std:
                noise_v[i, outlier_indices] = noise_v[i, outlier_indices]*5
        y += noise_v  
    if not (bias is None):
        bias_v = np.repeat(bias, y_len, axis=1)
        y += bias_v
    return y

### Funcations for generating fake angles:

def pdoa2ang(a):
    # Function for converting pdoa to angle - including all the possible ambiguities
    
    d = 0.026 # 0.0231      # Distance between antennas
    f = 6.4896e9    # frequency
    c = 299792458   # speed of light
    lam = c/f       # wavelength

    # The two thresholds
    rupp = -(2*np.pi*d)/lam + 2*np.pi
    rlow = (2*np.pi*d)/lam - 2*np.pi

    ang1 = np.arcsin((a*lam)/(2*np.pi*d))
    
    if a >= rupp:
        ang2 = np.arcsin(((a-2*np.pi)*lam)/(2*np.pi*d))
    elif a <= rlow:
        ang2 = np.arcsin(((a+2*np.pi)*lam)/(2*np.pi*d))
    else:
        # Simple case of only two ambiguities:
        mang = np.zeros(2)
        mang[0] = ang1
        mang[1] = mf.normalize_angle(np.pi - ang1)

        return mang
    # More advanced case of four ambiguities:

    mang = np.zeros(4)
    mang[0] = ang1
    mang[1] = ang2
    mang[2] = mf.normalize_angle(np.pi - ang1)
    mang[3] = mf.normalize_angle(np.pi - ang2)

    return mang

def gen_adv_amb(a):
    
    d = 0.026 # 0.0231      # Distance between antennas
    f = 6.4896e9    # frequency
    c = 299792458   # speed of light
    lam = c/f       # wavelength
    
    pdoa = mf.normalize_angle(np.sin(a)*(2*np.pi*d)/(lam)) # First generate the equivalent pdoa
    mang = pdoa2ang(pdoa) # Then all the ambiguities
    return mang # Return up to four ambiguities

def gen_rb(thetai, thetaj, posi, posj, ti, tj):
    """
    Generate simple range/bearing measurement
    """
    # Precompute q
    q = (posj + mf.RM(thetaj) @ tj - posi - mf.RM(thetai) @ ti)
    # Range and bearing:
    z = np.zeros((2,1))
    z[0] = np.arctan2(q[1], q[0]) - thetai
    z[1] = np.sqrt(np.transpose(q) @ q)

    return z

def gen_rb_amb(thetai, thetaj, posi, posj, ti, tj, sb=0, sr=0, pout_r=0, pout_b=0, amb=True):
    # TODO: look this function through, make sure it operates correct
    """
    Generate range/bearing measurement with front/back ambiguity.
    This is supposed to be used with the ML before being send to the 
    Kalman filter
    """
    # Precompute q
    q = (posj + mf.RM(thetaj) @ tj - posi - mf.RM(thetai) @ ti)
    
    # Range and bearing:
    b = np.arctan2(q[1], q[0]) - thetai # true measurement
    r = np.sqrt(np.transpose(q) @ q)

    # Generate noise
    # Range noise
    if not (sr==0):
        r_noise = np.sqrt(sr)*np.random.randn()
        if not (pout_r==0) and np.random.rand() < pout_r:
            # If the returned unifrom probability is smaller than the threshold, generate an outlier:
            r_noise = r_noise*5
            print("Range outlier generated") # debug
        r += r_noise
        r = max(r,0) # ranges cannot be smaller than 0
    
    # Bearing noise
    if not (sb==0):
        b_noise = np.sqrt(sb)*np.random.randn()
        if not (pout_b==0) and np.random.rand() < pout_b:
            # If the returned unifrom probability is smaller than the threshold, generate an outlier:
            b_noise = b_noise*5
            print("Bearing outlier generated") # debug
        b += b_noise

    if amb:
        ys = np.zeros((2,2)) # two ambiguities
        ys[0,0] = mf.normalize_angle(b)
        ys[0,1] = mf.normalize_angle(np.pi - b) # bad measurement
        ys[1,0] = r
        ys[1,1] = r
    else:
        ys = np.zeros((2,1)) # No ambiguities
        ys[0,0] = mf.normalize_angle(b)
        ys[1,0] = r

    return ys

def gen_rb2_amb(thetai, thetaj, posi, posj, ti, tj, sb=0, sr=0, pout_r=0, pout_b=0, amb=True):
    """
    Generates two measurements - one for AoA and one for AoD
    """
    y1 = gen_rb_amb(thetai, thetaj, posi, posj, ti, tj, sb, sr, pout_r, pout_b, amb)
    y2 = gen_rb_amb(thetaj, thetai, posj, posi, tj, ti, sb, sr, pout_r, pout_b, amb)
    # Now mix them:
    if not amb:
        # 'perfect' measurement
        y_out = np.zeros((mf.RB2_LEN, 1))
        y_out[mf.Z_PHI, 0] = y1[mf.Z_PHI, 0]
        y_out[mf.Z_PHI2, 0] = y2[mf.Z_PHI, 0]
        y_out[mf.Z_R2, 0] = y1[mf.Z_R, 0]
    else: 
        # Ambigious measurement
        y_out = np.zeros((mf.RB2_LEN, 4))
        
        y_out[mf.Z_PHI, 0] = y1[mf.Z_PHI, 0]
        y_out[mf.Z_PHI2, 0] = y2[mf.Z_PHI, 0]
        y_out[mf.Z_PHI, 1] = y1[mf.Z_PHI, 1]
        y_out[mf.Z_PHI2, 1] = y2[mf.Z_PHI, 0]
        y_out[mf.Z_PHI, 2] = y1[mf.Z_PHI, 0]
        y_out[mf.Z_PHI2, 2] = y2[mf.Z_PHI, 1]
        y_out[mf.Z_PHI, 3] = y1[mf.Z_PHI, 1]
        y_out[mf.Z_PHI2, 3] = y2[mf.Z_PHI, 1]

        y_out[mf.Z_R2, :] = y1[mf.Z_R, 0]

    return y_out
