import numpy as np
import models_functions as mf
import robot_sim as rsim
import traj as traj
import math

########### different anchor setups ##############
def anc_setup1():
    """
        Anchors in each corner of the arena
    """
    xanc1 = np.zeros((mf.STATE_LEN, 1)); xanc1[:4] = np.array([[-np.pi/4],[0],[1],[69]])
    xanc2 = np.zeros((mf.STATE_LEN, 1)); xanc2[:4] = np.array([[3*np.pi/4],[0],[99],[1]])
    return xanc1, xanc2

def anc_setup2():
    """
        Anchor in the other corners of the arena
    """
    xanc1 = np.zeros((mf.STATE_LEN, 1)); xanc1[:4] = np.array([[np.pi/4],[0],[1],[1]])
    xanc2 = np.zeros((mf.STATE_LEN, 1)); xanc2[:4] = np.array([[-3*np.pi/4],[0],[99],[69]])
    return xanc1, xanc2

def anc_setup3():
    """
        Anchors clumped close together    
    """
    xanc1 = np.zeros((mf.STATE_LEN, 1)); xanc1[:4] = np.array([[0],[0],[1],[65]])
    xanc2 = np.zeros((mf.STATE_LEN, 1)); xanc2[:4] = np.array([[0],[0],[5],[69]])
    return xanc1, xanc2

def anc_setup4():
    """
        Anchors in the middle
    """
    xanc1 = np.zeros((mf.STATE_LEN, 1)); xanc1[:4] = np.array([[0],[0],[45],[35]])
    xanc2 = np.zeros((mf.STATE_LEN, 1)); xanc2[:4] = np.array([[0],[0],[55],[35]])
    return xanc1, xanc2

def anc_setup5():
    """
        Anchor in the middle
    """
    xanc1 = np.zeros((mf.STATE_LEN, 1)); xanc1[:4] = np.array([[0],[0],[50],[35]])
    return xanc1

########### different trajectories pre-defined ##############

# All running over 10 minutes
# Slow trajectories
# waits 30 seconds before lift-off
def path1_slow(dt=1):
    """
    Slow trajectory 1
    Runs for 10 minutes
    Runs all over map
    """
    # still
    x0 = np.zeros((mf.STATE_LEN, 1)); x0[:6] = np.array([[-1.57],[0],[5],[50],[0],[0]])
    
    # Moving
    xb = np.zeros((mf.STATE_LEN, 1)); xb[:6] = np.array([[0],[0],[60],[65],[0.5],[0]])
    xc = np.zeros((mf.STATE_LEN, 1)); xc[:6] = np.array([[-1.57],[0],[95],[4],[0],[0]])
    xd = np.zeros((mf.STATE_LEN, 1)); xd[:6] = np.array([[0],[0],[80],[10],[-0.5],[0]])
    xe = np.zeros((mf.STATE_LEN, 1)); xe[:6] = np.array([[3.14],[0],[20],[30],[0.5],[0.5]])
    xf = np.zeros((mf.STATE_LEN, 1)); xf[:6] = np.array([[1.57],[0],[90],[50],[0],[-0.5]])
    xg = np.zeros((mf.STATE_LEN, 1)); xg[:6] = np.array([[0],[0],[90],[10],[0],[0]])

    pos, y_IMU = traj.gen_poly_traj_multi(pos=[x0, x0, xb, xc, xd, xe, xf, xg], 
                                          time=[0, 30, 120, 200, 260, 380, 500, 630], dt=dt)

    return pos, y_IMU, x0

def path2_slow(dt=1):
    """
    Slow trajectory 2
    Runs for 10 minutes
    Runs all over map
    Two rounds
    """
    # Stay still here
    xa = np.zeros((mf.STATE_LEN, 1)); xa[:6] = np.array([[0],[0],[5],[30],[0],[0]])
    # Then move
    xb = np.zeros((mf.STATE_LEN, 1)); xb[:6] = np.array([[3.14],[0],[70],[50],[0],[0.5]])
    xc = np.zeros((mf.STATE_LEN, 1)); xc[:6] = np.array([[4.71],[0],[30],[50],[-0.5],[-0.5]])
    xd = np.zeros((mf.STATE_LEN, 1)); xd[:6] = np.array([[4.71],[0],[30],[5],[0.5],[0]])
    xe = np.zeros((mf.STATE_LEN, 1)); xe[:6] = np.array([[3.14],[0],[80],[40],[0],[0.5]])
    xf = np.zeros((mf.STATE_LEN, 1)); xf[:6] = np.array([[1.57],[0],[5],[40],[0],[-0.25]])

    pos, y_IMU = traj.gen_poly_traj_multi(pos=[xa, xa, xb, xc, xd, xe, xf], 
                                          time=[0, 30, 150, 200, 280, 430, 630], dt=dt)

    return pos, y_IMU, xa

def path3_slow(dt=1):
    """
    Slow trajectory 3
    Runs for 10 minutes
    Runs all over map
    Two rounds
    """
    xa = np.zeros((mf.STATE_LEN, 1)); xa[:6] = np.array([[-1.57],[0],[10],[60],[0],[0]])
    xb = np.zeros((mf.STATE_LEN, 1)); xb[:6] = np.array([[0],[0],[20],[20],[0.5],[0]])
    xc = np.zeros((mf.STATE_LEN, 1)); xc[:6] = np.array([[1.57],[0],[80],[30],[0],[0.5]])
    xd = np.zeros((mf.STATE_LEN, 1)); xd[:6] = np.array([[3.14],[0],[85],[60],[0],[0]])
    xe = np.zeros((mf.STATE_LEN, 1)); xe[:6] = np.array([[3.14],[0],[50],[40],[-0.5],[0]])
    xf = np.zeros((mf.STATE_LEN, 1)); xf[:6] = np.array([[1.57],[0],[40],[50],[0],[0.05]])
    xg = np.zeros((mf.STATE_LEN, 1)); xg[:6] = np.array([[0],[0],[15],[60],[0],[-0.5]])
    xh = np.zeros((mf.STATE_LEN, 1)); xh[:6] = np.array([[-1.57],[0],[5],[5],[-0.5],[0]])

    pos, y_IMU = traj.gen_poly_traj_multi(pos=[xa, xa, xb, xc, xd, xe, xf, xg, xh], 
                                          time=[0, 30, 120, 220, 280, 360, 400, 480, 630], dt=dt)

    return pos, y_IMU, xa

def path4_slow(dt=1):
    """
    Slow trajectory 4
    Runs for 10 minutes
    Runs all over map
    """
    xa = np.zeros((mf.STATE_LEN, 1)); xa[:6] = np.array([[1.57],[0],[5],[10],[0],[0]])
    xb = np.zeros((mf.STATE_LEN, 1)); xb[:6] = np.array([[0],[0],[60],[15],[0],[-0.5]])
    xc = np.zeros((mf.STATE_LEN, 1)); xc[:6] = np.array([[-1.57],[0],[95],[30],[0],[0.5]])
    xd = np.zeros((mf.STATE_LEN, 1)); xd[:6] = np.array([[-1.57],[0],[80],[65],[-0.5],[0]])
    xe = np.zeros((mf.STATE_LEN, 1)); xe[:6] = np.array([[-3.14],[0],[60],[25],[0],[-0.5]])
    xf = np.zeros((mf.STATE_LEN, 1)); xf[:6] = np.array([[-1.57],[0],[5],[25],[0],[0.5]])
    xg = np.zeros((mf.STATE_LEN, 1)); xg[:6] = np.array([[0],[0],[20],[60],[0.5],[0]])

    pos, y_IMU = traj.gen_poly_traj_multi(pos=[xa, xa, xb, xc, xd, xe, xf, xg], 
                                          time=[0, 30, 180, 240, 340, 440, 540, 630], dt=dt)

    return pos, y_IMU, xa

def path5_slow(dt=1):
    """
    Slow trajectory 5
    Runs for 10 minutes
    Runs all over map
    """
    xa = np.zeros((mf.STATE_LEN, 1)); xa[:6] = np.array([[3.14],[0],[95],[5],[0],[0]])
    xb = np.zeros((mf.STATE_LEN, 1)); xb[:6] = np.array([[1.57],[0],[25],[25],[0],[0.25]])
    xc = np.zeros((mf.STATE_LEN, 1)); xc[:6] = np.array([[-1.57],[0],[85],[40],[0.5],[0]])
    xd = np.zeros((mf.STATE_LEN, 1)); xd[:6] = np.array([[-1.57],[0],[90],[60],[-0.5],[0]])
    xe = np.zeros((mf.STATE_LEN, 1)); xe[:6] = np.array([[-3.14],[0],[40],[65],[-0.5],[0]])
    xf = np.zeros((mf.STATE_LEN, 1)); xf[:6] = np.array([[-1.57],[0],[30],[20],[0.5],[0]])

    pos, y_IMU = traj.gen_poly_traj_multi(pos=[xa, xa, xb, xc, xd, xe, xf], 
                                          time=[0, 30, 300, 420, 480, 550, 630], dt=dt)

    return pos, y_IMU, xa

def path6_slow(dt=1):
    """
    Slow trajectory 6
    Runs for 10 minutes
    Runs all over map
    """
    xa = np.zeros((mf.STATE_LEN, 1)); xa[:6] = np.array([[1.57],[0],[80],[5],[0],[0]])
    xb = np.zeros((mf.STATE_LEN, 1)); xb[:6] = np.array([[0],[0],[5],[35],[0],[0.25]])
    xc = np.zeros((mf.STATE_LEN, 1)); xc[:6] = np.array([[-1.57],[0],[50],[65],[0.25],[0]])
    xd = np.zeros((mf.STATE_LEN, 1)); xd[:6] = np.array([[-3.14],[0],[75],[20],[0],[-0.25]])

    pos, y_IMU = traj.gen_poly_traj_multi(pos=[xa, xa, xb, xc, xd], 
                                          time=[0, 30, 260, 460, 630], dt=dt)

    return pos, y_IMU, xa

# Fast tracetories
# Multiple rounds
# waits 30 seconds before lift-off

def path1_fast(dt=1):
    """
    Fast trajectory 1
    Swirly pattern
    """
    xa = np.zeros((mf.STATE_LEN, 1)); xa[:6] = np.array([[0],[0],[0.5],[5],[0],[0]])
    xb = np.zeros((mf.STATE_LEN, 1)); xb[:6] = np.array([[3.14],[0],[4],[6.5],[0.5],[0]])
    xc = np.zeros((mf.STATE_LEN, 1)); xc[:6] = np.array([[0],[0],[9.5],[5],[0],[1.0]])
    xe = np.zeros((mf.STATE_LEN, 1)); xe[:6] = np.array([[-3.14],[0],[1],[2],[0],[1.0]])
    xf = np.zeros((mf.STATE_LEN, 1)); xf[:6] = np.array([[0],[0],[9],[2],[0],[0.5]])

    pos, y_IMU = traj.gen_poly_traj_multi(pos=[xa, xa, xb, xc, xe, xf, xa], 
                                          time=[0, 30, 40, 50, 60, 70, 80, 90], dt=dt)

    return pos, y_IMU, xa

def path2_fast(dt=1):
    """
    Fast trajectory 1
    Start and stop
    """
    xa = np.zeros((mf.STATE_LEN, 1)); xa[:6] = np.array([[0],[0],[5],[0.5],[0],[0]])
    xb = np.zeros((mf.STATE_LEN, 1)); xb[:6] = np.array([[1.57],[0],[5],[6.5],[0],[0]])
    xc = np.zeros((mf.STATE_LEN, 1)); xc[:6] = np.array([[3.14],[0],[1],[4],[0],[0]])
    xe = np.zeros((mf.STATE_LEN, 1)); xe[:6] = np.array([[1.57],[0],[9],[4],[0],[0]])

    pos, y_IMU = traj.gen_poly_traj_multi(pos=[xa, xa, xb, xc, xe, xa], 
                                          time=[0, 30, 40, 50, 60, 70, 85, 90], dt=dt)

    return pos, y_IMU, xa

def path4_fast(dt=1):
    """
    This is just path1_slow modified, so it runs faster
    """
    # Stay still here
    xa = np.zeros((mf.STATE_LEN, 1)); xa[:6] = np.array([[0],[0],[0.5],[3],[0],[0]])
    # Then move
    xb = np.zeros((mf.STATE_LEN, 1)); xb[:6] = np.array([[3.14],[0],[7],[5],[0],[0.05]])
    xc = np.zeros((mf.STATE_LEN, 1)); xc[:6] = np.array([[4.71],[0],[3],[5],[-0.05],[-0.05]])
    xd = np.zeros((mf.STATE_LEN, 1)); xd[:6] = np.array([[4.71],[0],[3],[0.5],[0.05],[0]])
    xe = np.zeros((mf.STATE_LEN, 1)); xe[:6] = np.array([[3.14],[0],[8],[4],[0],[0.05]])
    xf = np.zeros((mf.STATE_LEN, 1)); xf[:6] = np.array([[1.57],[0],[0.5],[4],[0],[-0.05]])

    pos, y_IMU = traj.gen_poly_traj_multi(pos=[xa, xa, xb, xc, xd, xe, xf], 
                                          time=[0, 30, 15, 20, 25, 40, 60], dt=dt)

    return pos, y_IMU, xa

def path5_fast(dt=1):
    """
    This is just path2_slow modified, so it runs faster
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

def path6_fast(dt=1):
    """
    This is just path3_slow modified, so it runs faster
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

########### Functions to handle updates ##############

def updateAllLuft(robots: list, 
                   update_list: list,
                   params: list,
                   i_indx: int,
                   dt: int,
                   delay: float):
    """
        Perform a measurements to all other robots.
        Each indivudal robot have a delay between measurements of exc_time
        # Params have the shape:
        [[sr=0, sb=0, thres_anc=0, thres_rob=0, max_dist=-1, amb=True, meas2_anc=false, meas2_rob=False, pout_r, pout_b],
        ...,
        ...]
    """
    # We are doing something a bit dirty here
    # This also means i == 0 HAS TO EXIST
    if i_indx == 0:
        updateAllLuft.j = 0
        updateAllLuft.robi = 0

    delay_idx = round(delay/dt)
    N = len(robots)
    start_j = updateAllLuft.j
    start_rob = updateAllLuft.robi 
    chk = None

    # if it is time to perform an update
    if (i_indx % delay_idx == 0):
        # First skip all invalid entries in the update list
        while (1):
            j = updateAllLuft.j
            i = updateAllLuft.robi
            # check if entry is none or is the robot itself
            if (update_list[i][j] is not None) and (robots[i] != update_list[i][j]):
                # if the robot is valid, then make the measurement
                if isinstance(update_list[i][j], rsim.Anchor):
                    chk = robots[i].anchor_meas(update_list[i][j], params[i][0], params[i][1], params[i][2], params[i][4], params[i][5], params[i][6], params[i][8], params[i][9])
                elif isinstance(update_list[i][j], rsim.robot_luft):
                    chk = robots[i].robot_meas(update_list[i][j], params[i][0], params[i][1], params[i][3], params[i][4], params[i][5], params[i][7], params[i][8], params[i][9])
                # These functions return none, if the robots are out of range 

            # Move on to next entry in update list and possibly the next robot
            updateAllLuft.j += 1
            if updateAllLuft.j >= len(update_list[updateAllLuft.robi]):
                # Measurements to all the robots in the update list have been made - move on to next robot
                updateAllLuft.j = 0
                updateAllLuft.robi += 1
                if updateAllLuft.robi == N:
                    # loop around
                    updateAllLuft.robi = 0
            
            # A valid measurement has been made - break out of the loop
            if (chk is not None):
                break

            # Security check: If no robots are able to make a measurement, break the loop
            if (updateAllLuft.j == start_j and updateAllLuft.robi == start_rob):
                break
        
    # Return the NIS value
    return chk
        

def updateAllSimple(robot: rsim.Robot_single, 
                   anchor: list, rob_other: list,
                   ax = None,
                   sr=0,
                   sb=0,
                   thres=0):
    """
        Perform a measurements to all other robots
    """
    for a in anchor:
        robot.anchor_meas(a, ax, sr, sb, thres)
    for r in rob_other:
        robot.robot_meas(r, ax, sr, sb, thres)
