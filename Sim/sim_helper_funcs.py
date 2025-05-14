import numpy as np
import models_functions as mf
import robot_sim as rsim
import traj as traj

########### different anchor setups ##############
def anc_setup1():
    """
        Anchors in each corner of the arena
    """
    xanc1 = np.zeros((mf.STATE_LEN, 1)); xanc1[:4] = np.array([[-0.7854],[0],[0.1],[6.9]])
    xanc2 = np.zeros((mf.STATE_LEN, 1)); xanc2[:4] = np.array([[2.3562],[0],[9.9],[0.1]])
    return xanc1, xanc2

def anc_setup2():
    """
        Anchor in the other corners of the arena
    """
    xanc1 = np.zeros((mf.STATE_LEN, 1)); xanc1[:4] = np.array([[-0.7854],[0],[0.1],[0.1]])
    xanc2 = np.zeros((mf.STATE_LEN, 1)); xanc2[:4] = np.array([[2.3562],[0],[9.9],[6.9]])
    return xanc1, xanc2

def anc_setup3():
    """
        Anchors clumped close together    
    """
    xanc1 = np.zeros((mf.STATE_LEN, 1)); xanc1[:4] = np.array([[-0.7854],[0],[0.1],[6.5]])
    xanc2 = np.zeros((mf.STATE_LEN, 1)); xanc2[:4] = np.array([[2.3562],[0],[0.5],[6.9]])
    return xanc1, xanc2

def anc_setup4():
    """
        Anchors in the middle
    """
    xanc1 = np.zeros((mf.STATE_LEN, 1)); xanc1[:4] = np.array([[-0.7854],[0],[4.5],[3.5]])
    xanc2 = np.zeros((mf.STATE_LEN, 1)); xanc2[:4] = np.array([[2.3562],[0],[5.5],[3.5]])
    return xanc1, xanc2

########### different trajectories pre-defined ##############

# All running over 10 minutes
# Slow trajectories
# Runs two rounds
# waits 30 seconds before lift-off
def path1_slow(dt=1):
    """
    Slow trajectory 1
    Runs for 10 minutes
    Runs all over map
    Two rounds
    """
    # still
    x0 = np.zeros((mf.STATE_LEN, 1)); x0[:6] = np.array([[-1.57],[0],[0.5],[5],[0],[0]])
    
    # Moving
    xb = np.zeros((mf.STATE_LEN, 1)); xb[:6] = np.array([[0],[0],[6],[6.5],[0.05],[0]])
    xc = np.zeros((mf.STATE_LEN, 1)); xc[:6] = np.array([[-1.57],[0],[9.5],[4],[0],[-0.05]])
    xd = np.zeros((mf.STATE_LEN, 1)); xd[:6] = np.array([[0],[0],[8],[1],[-0.05],[0]])
    xe = np.zeros((mf.STATE_LEN, 1)); xe[:6] = np.array([[3.14],[0],[2],[3],[0.05],[0.05]])
    xf = np.zeros((mf.STATE_LEN, 1)); xf[:6] = np.array([[1.57],[0],[9],[5],[0],[-0.05]])
    xg = np.zeros((mf.STATE_LEN, 1)); xg[:6] = np.array([[0],[0],[9],[1],[-0.05],[0]])

    pos, y_IMU = traj.gen_poly_traj_multi(pos=[x0, x0, xb, xc, xd, xe, xf, xg, x0, xb, xc, xd, xe, xf, xg], 
                                          time=[0, 30, 60, 90, 120, 180, 240, 300, 330, 360, 390, 420, 480, 540, 600], dt=dt)

    return pos, y_IMU, x0

def path2_slow(dt=1):
    """
    Slow trajectory 2
    Runs for 10 minutes
    Runs all over map
    Two rounds
    """
    # Stay still here
    xa = np.zeros((mf.STATE_LEN, 1)); xa[:6] = np.array([[0],[0],[0.5],[3],[0],[0]])
    # Then move
    xb = np.zeros((mf.STATE_LEN, 1)); xb[:6] = np.array([[3.14],[0],[7],[5],[0],[0.05]])
    xc = np.zeros((mf.STATE_LEN, 1)); xc[:6] = np.array([[4.71],[0],[3],[5],[-0.05],[-0.05]])
    xd = np.zeros((mf.STATE_LEN, 1)); xd[:6] = np.array([[4.71],[0],[3],[0.5],[0.05],[0]])
    xe = np.zeros((mf.STATE_LEN, 1)); xe[:6] = np.array([[3.14],[0],[8],[4],[0],[0.05]])
    xf = np.zeros((mf.STATE_LEN, 1)); xf[:6] = np.array([[1.57],[0],[0.5],[4],[0],[-0.05]])

    pos, y_IMU = traj.gen_poly_traj_multi(pos=[xa, xa, xb, xc, xd, xe, xf, xa, xb, xc, xd, xe, xf], 
                                          time=[0, 30, 75, 100, 130, 200, 300, 330, 375, 400, 430, 500, 600], dt=dt)

    return pos, y_IMU, xa

def path3_slow(dt=1):
    """
    Slow trajectory 3
    Runs for 10 minutes
    Runs all over map
    Two rounds
    """
    xa = np.zeros((mf.STATE_LEN, 1)); xa[:6] = np.array([[-1.57],[0],[1],[6],[0],[0]])
    xb = np.zeros((mf.STATE_LEN, 1)); xb[:6] = np.array([[0],[0],[2],[0.5],[0.05],[0]])
    xc = np.zeros((mf.STATE_LEN, 1)); xc[:6] = np.array([[1.57],[0],[8],[2],[0],[0.05]])
    xd = np.zeros((mf.STATE_LEN, 1)); xd[:6] = np.array([[3.14],[0],[8.5],[6],[0],[0]])
    xe = np.zeros((mf.STATE_LEN, 1)); xe[:6] = np.array([[3.14],[0],[5],[4],[-0.05],[0]])
    xf = np.zeros((mf.STATE_LEN, 1)); xf[:6] = np.array([[1.57],[0],[4],[5],[0],[0.05]])
    xg = np.zeros((mf.STATE_LEN, 1)); xg[:6] = np.array([[0],[0],[1.5],[6],[0],[-0.05]])
    xh = np.zeros((mf.STATE_LEN, 1)); xh[:6] = np.array([[-1.57],[0],[0.5],[0.5],[-0.05],[0]])

    pos, y_IMU = traj.gen_poly_traj_multi(pos=[xa, xa, xb, xc, xd, xe, xf, xg, xh, xa, xb, xc, xd, xe, xf, xg, xh], 
                                          time=[0, 30, 70, 100, 130, 180, 200, 240, 300, 330, 370, 400, 430, 480, 500, 540, 600], dt=dt)

    return pos, y_IMU, xa

def path4_slow(dt=1):
    """
    Slow trajectory 4
    Runs for 10 minutes
    Runs all over map
    """
    xa = np.zeros((mf.STATE_LEN, 1)); xa[:6] = np.array([[1.57],[0],[0.5],[1],[0],[0]])
    xb = np.zeros((mf.STATE_LEN, 1)); xb[:6] = np.array([[0],[0],[6],[1.5],[0],[-0.05]])
    xc = np.zeros((mf.STATE_LEN, 1)); xc[:6] = np.array([[-1.57],[0],[9.5],[3],[0],[0.05]])
    xd = np.zeros((mf.STATE_LEN, 1)); xd[:6] = np.array([[-1.57],[0],[8],[6.5],[-0.05],[0]])
    xe = np.zeros((mf.STATE_LEN, 1)); xe[:6] = np.array([[-3.14],[0],[6],[2.5],[0],[-0.05]])
    xf = np.zeros((mf.STATE_LEN, 1)); xf[:6] = np.array([[-1.57],[0],[0.5],[2.5],[0],[0.05]])
    xg = np.zeros((mf.STATE_LEN, 1)); xg[:6] = np.array([[0],[0],[2],[6],[0.05],[0]])

    pos, y_IMU = traj.gen_poly_traj_multi(pos=[xa, xa, xb, xc, xd, xe, xf, xg, xa, xb, xc, xd, xe, xf, xg], 
                                          time=[0, 30, 90, 120, 170, 220, 270, 300, 330, 390, 420, 470, 520, 570, 600], dt=dt)

    return pos, y_IMU, xa

def path5_slow(dt=1):
    """
    Slow trajectory 5
    Runs for 10 minutes
    Runs all over map
    2 rounds
    """
    xa = np.zeros((mf.STATE_LEN, 1)); xa[:6] = np.array([[3.14],[0],[9.5],[0.5],[0],[0]])
    xb = np.zeros((mf.STATE_LEN, 1)); xb[:6] = np.array([[1.57],[0],[2.5],[2.5],[0],[0.05]])
    xc = np.zeros((mf.STATE_LEN, 1)); xc[:6] = np.array([[-1.57],[0],[8.5],[4],[0.05],[0]])
    xd = np.zeros((mf.STATE_LEN, 1)); xd[:6] = np.array([[-1.57],[0],[9],[6],[-0.05],[0]])
    xe = np.zeros((mf.STATE_LEN, 1)); xe[:6] = np.array([[-3.14],[0],[4],[6.5],[-0.05],[0]])
    xf = np.zeros((mf.STATE_LEN, 1)); xf[:6] = np.array([[-1.57],[0],[3],[2],[0.05],[0]])

    pos, y_IMU = traj.gen_poly_traj_multi(pos=[xa, xa, xb, xc, xd, xe, xf, xa, xb, xc, xd, xe, xf], 
                                          time=[0, 30, 150, 210, 240, 275, 300, 330, 450, 510, 540, 575, 600], dt=dt)

    return pos, y_IMU, xa

def path6_slow(dt=1):
    """
    Slow trajectory 6
    Runs for 10 minutes
    Runs all over map
    Two rounds
    """
    xa = np.zeros((mf.STATE_LEN, 1)); xa[:6] = np.array([[1.57],[0],[8],[0.5],[0],[0]])
    xb = np.zeros((mf.STATE_LEN, 1)); xb[:6] = np.array([[0],[0],[0.5],[3.5],[0],[0.05]])
    xc = np.zeros((mf.STATE_LEN, 1)); xc[:6] = np.array([[-1.57],[0],[5],[6.5],[0.05],[0]])
    xd = np.zeros((mf.STATE_LEN, 1)); xd[:6] = np.array([[-3.14],[0],[7.5],[2],[0],[-0.05]])

    pos, y_IMU = traj.gen_poly_traj_multi(pos=[xa, xa, xb, xc, xd, xa, xb, xc, xd], 
                                          time=[0, 30, 130, 230, 300, 330, 430, 530, 600], dt=dt)

    return pos, y_IMU, xa

# Fast tracetories
# Multiple rounds
# waits 30 seconds before lift-off

########### Functions to handle updates ##############

def updateAllLuft(robot: rsim.robot_luft, 
                   anchor: list, rob_other: list,
                   ax = None,
                   sr=0,
                   sb=0,
                   thres=0):
    """
        Perform a measurements to all other robots
    """
    inno_anc = []
    inno_rob = []
    for a in anchor:
        inno_anc += [robot.anchor_meas(a, ax, sr, sb, thres)]
    for r in rob_other:
        inno_rob += [robot.robot_meas_luft(r, ax, sr, sb, thres)]
    
    return inno_anc, inno_rob

def updateAllSimple(robot: rsim.Robot_single, 
                   anchor: list, rob_other: list,
                   ax = None,
                   sr=0,
                   sb=0,
                   thres=0):
    """
        Perform a measurements to all other robots
    """
    inno_anc = []
    inno_rob = []
    for a in anchor:
        inno_anc += [robot.anchor_meas(a, ax, sr, sb, thres)]
    for r in rob_other:
        inno_rob += [robot.robot_meas(r, ax, sr, sb, thres)]
    
    return inno_anc, inno_rob
