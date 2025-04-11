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

def gen_rb(thetai, thetaj, posi, posj, ti, tj):
    # Precompute q
    q = (posj + mf.RM(thetaj) @ tj - posi - mf.RM(thetai) @ ti)
    # Range and bearing:
    z = np.zeros((2,1))
    z[0] = np.arctan2(q[1], q[0]) - thetai
    z[1] = np.sqrt(np.transpose(q) @ q)

    return z