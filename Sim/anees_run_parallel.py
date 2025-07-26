# %%
import robot_plotter as rp
import models_functions as mf
import traj
import numpy as np
import matplotlib.pyplot as plt
import robot_sim as sim
import sim_helper_funcs as hfunc
import save_load_data as sldat
from multiprocessing import Pool
import winsound

# %%
# generate reference trajectory
# Sampling time
dt = 0.01

# Define all the paths for the robots and thus also the number of robots
paths = [hfunc.path1_slow, hfunc.path2_slow, hfunc.path3_slow, hfunc.path4_slow, hfunc.path5_slow, hfunc.path6_slow] 
robot_N = len(paths)
pos = []
y_IMU_base = []
x0 = []

end_idx = None # can be modified for shorter trajectories
#end_idx = round(60/dt)

for path in paths:
    p, imu, x = path(dt)
    pos.append(p[:,:end_idx])
    y_IMU_base.append(imu[:,:end_idx])
    x0.append(x)

pos_len = pos[0].shape[1] # Assumes that all paths have the same length

# Generate anchors position:
#x_ancs = [*hfunc.anc_setup1(), *hfunc.anc_setup2()]
x_ancs = [*hfunc.anc_setup1()]
#x_ancs = [hfunc.anc_setup5()]

# %%
R_b =  0.011 # old value from datasheet 0.0009 #
R_r =  0.0008 # old value from datasheet # 0.001
R_w = 0.00002
R_a = 0.001

R = np.diag([R_b, R_r, R_w, R_a, R_a])
Q = np.diag([0.1, 8.0, 8.0, 1e-8, 1e-7, 1e-7])
P = np.diag([0.1, 0.001, 1.0, 1.0, 0.1, 0.1, 0.01, 0.01, 0.0001, 0.1, 0.1])

anchors = []
for i in range(len(x_ancs)):
    anchors.append(sim.Anchor(x0=x_ancs[i], id=(i+1)))

# %%
list_seeds = [1]
#list_seeds = [1061, 1, 19001, 7871871, 2289, 91667, 8, 6119077, 47, 5514]
seeds_num = len(list_seeds)

# Sim settings
sr = R_r
sb = R_b
r_w = R_w
r_a = R_a
uwb_trans = np.array([[0.1],[0.0]])
sim_max_dist = 0 # 35
thres_anc = 15.4 # 99.5 % confidence
thres_rob = 15.4 # 99.5 % confidence else 13.0 for df=2
thres_IMU = 15.4 # 99.5 % confidence
meas2_anc = True
meas2_rob = True
amb = True

out_freq = np.array([[0.0],[0.0],[0.0]]) # an outlier every second on average
pout_r = 0
pout_b = 0
bias_base = np.array([[0.07],[0.2],[0.2]]) # max magnitude of bias

meas_delay = 0.02

# Quick disabling of settings 
thres_anc = 0
thres_rob = 0
thres_IMU = 0
#bias_base = np.array([[0.0],[0.0],[0.0]]) 
#sr=0
#sb=0
#r_w = 0
#r_a = 0


# %%
def run_sim_loop(j):
    print("seed: " + str(list_seeds[j]))
    np.random.seed(list_seeds[j])
    
    # generate IMU measurments:
    y_IMU = []
    bias_vec = []
    robots = []
    for i in range(robot_N):
        bias_vec.append(2*bias_base.copy()*(np.random.rand(3, 1) - 0.5))
        y_IMU.append(traj.gen_noise(y_IMU_base[i].copy(), dt=dt, sigma=np.diag([r_w, r_a, r_a]), bias=bias_vec[i], out_freq=out_freq))

        # Setup robots
        robots.append( sim.robot_luft(x0=x0[i], path=pos[i], imu=y_IMU[i], dt=dt, id=111*(i+1), t=uwb_trans, R=R, P=P, Q=Q) )
        
    # Setup update params:
    update_list = anchors + robots
    #params = [sr, sb, thres_anc, thres_rob, sim_max_dist, amb, meas2_anc, meas2_rob, pout_r, pout_b]

    # Run Lufts algorithm
    for i in range(pos_len-1):
        for r in robots:
            r.predict(imu_correct=True, thres=thres_IMU)
        
        # Special rule: The first 30 seconds is used for proper initialization. It is assumed that the UAVs all have access to anchors here
        if (i*dt < 30):
            params = [sr, sb, thres_anc, thres_rob, 0, amb, meas2_anc, meas2_rob, pout_r, pout_b]
        else:
            params = [sr, sb, thres_anc, thres_rob, sim_max_dist, amb, meas2_anc, meas2_rob, pout_r, pout_b]
       
       # Set up a measurement pattern similar to the used communication protocol
        hfunc.updateAllLuft(robots, [update_list]*robot_N, [params]*robot_N, i, dt, meas_delay)
    
    # Save estimates and covariances:
    return {
        'j': j,
        'robots' : robots,
        'biases' : bias_vec
    }

# %%
def execute_task_parallel():
    with Pool(4) as p:
        results = p.map(run_sim_loop, range(seeds_num))
    return results

# %%
def main():
    # execute loops in parallel
    results = execute_task_parallel()

    # Initialize logs
    x_log = np.zeros((robot_N, mf.STATE_LEN, pos_len, seeds_num))
    P_log = np.zeros((robot_N, mf.STATE_LEN, mf.STATE_LEN, pos_len, seeds_num))
    nis_imu_log = np.zeros((robot_N, pos_len, seeds_num))
    nis_rb_log = np.zeros((robot_N, pos_len, seeds_num))
    rb_ids = np.zeros((robot_N, pos_len, seeds_num))
    biases = np.zeros((robot_N, 3, seeds_num))
    self_ids = np.zeros((robot_N))
    ref_pos = np.zeros((robot_N, 3, pos_len))

    for res in results:
        j = res['j']
        for i in range(len(res['robots'])):
            rob = res['robots'][i]
            x_log[i,:,:,j] = rob.x_log
            P_log[i,:,:,:,j] = rob.P_log
            nis_imu_log[i,:,j] = rob.nis_IMU
            nis_rb_log[i,:,j] = rob.nis_rb
            rb_ids[i,:,j] = rob.rb_ids
            biases[i,:,j] = res['biases'][i].squeeze()
            # One time metrics that gets overwritten anyway
            self_ids[i] = rob.id
            ref_pos[i,:] = rob.path


    # finally save to an pickle object:
    robotData = sldat.RobotData(x_log, P_log, nis_imu_log, nis_rb_log, ref_pos, biases, self_ids, rb_ids, anchors)
    sldat.save_data(robotData, 'no_collab_1anc_0d_meas2')

    # Play sound 
    winsound.MessageBeep()


if __name__ == "__main__":
    main()


