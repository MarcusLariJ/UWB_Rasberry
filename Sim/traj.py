import numpy as np
import models_functions as mf

# File that contains various example trajectories
# and other ways to generate measurement data

def gen_circ_traj(R, N, dt=1, theta0=0, xc=0, yc=0, ccw = False):
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

def gen_noise(y: np.ndarray, bias: np.ndarray = None, sigma: np.ndarray = None, dt: float = 1.0):
    """
    Applies noise to the passed measurement
    Args:
        y (np.ndarray): The measurement(s) y_size X y_len 
        bias (np.ndarray): A vector of biases y_size X 1
        sigma (np.ndarray): Covariance matrix y_size X y_size
        dt (float): Sampling time
    """
    y_size, y_len = y.shape
    if not (bias is None):
        bias_v = np.repeat(bias, y_len, axis=1)
        y += bias_v
    if not (sigma is None):
        # Convert variance to std deviation
        noise_v = np.sqrt(sigma/dt) @ np.random.randn(y_size, y_len)
        y += noise_v
    return y

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

def gen_rb_amb(thetai, thetaj, posi, posj, ti, tj, sb=0, sr=0):
    """
    Generate range/bearing measurement with front/back ambiguity.
    This is supposed to be used with the ML before being send to the 
    Kalman filter
    """
    # Precompute q
    q = (posj + mf.RM(thetaj) @ tj - posi - mf.RM(thetai) @ ti)
    # Range and bearing:
    b = np.zeros(2) # two ambiguities
    b[0] = np.arctan2(q[1], q[0]) - thetai # true measurement
    b[1] = np.pi - b[0] # bad measurement TODO: can the wrappingpi function handle inputs oustide -180 to 180?
    r = np.sqrt(np.transpose(q) @ q)

    # apply noise
    if not (sr==0):
        r_noise = np.sqrt(sr)*np.random.randn()
        r += r_noise
    if not (sb==0):
        b_noise = np.sqrt(sb)*np.random.randn()
        b[0] += b_noise
        b[1] -= b_noise
    return b, r