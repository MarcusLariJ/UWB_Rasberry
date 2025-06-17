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

# %%
# generate reference trajectory
dt = 0.01

pos1, y_IMU1_base, xa1 = hfunc.path1_slow(dt)
pos2, y_IMU2_base, xa2 = hfunc.path2_slow(dt)
pos3, y_IMU3_base, xa3 = hfunc.path3_slow(dt)

pos_len = pos1.shape[1] # Assumes that all paths have the same length
#pos_len = round(120/dt) # overwrite for debugging

# Generate anchors position:
xanc1, xanc2 = hfunc.anc_setup1()
xanc3, xanc4 = hfunc.anc_setup2()

# %%
R_b = 0.0009
R_r = 0.001
R_w = 0.0002
R_a = 0.004

anchor1 = sim.Anchor(x0=xanc1, id=1)
anchor2 = sim.Anchor(x0=xanc2, id=2)


# %%
list_seeds = [1061]
#1061, 1, 19001, 7871871, 2289, 91667, 8, 6119077, 47, 5514
seeds_num = len(list_seeds)

# Sim settings
sr = R_r
sb = R_b
r_w = R_w
r_a = R_a
sim_max_dist = 0 # 35
thres_anc = 14.5 # 99.5 % confidence
thres_rob = 12 # 99.5 % confidence else 12 for df=2
thres_IMU = 14.5 # 99.5 % confidence
meas2_anc = True
meas2_rob = True
out_freq = np.array([[0.0],[0.0],[0.0]]) # an outlier every second on average
pout_r = 0
pout_b = 0
bias_base = np.array([[0.07],[0.2],[0.2]]) # max magnitude of bias
amb = True
update_freq = 1.0
meas_delay = 0.1
# Quick disabling of settings 
#thres_anc = 0
#thres_rob = 0
#thres_IMU = 0
#bias_base = np.array([[0.0],[0.0],[0.0]]) 
#sr=0
#sb=0
#r_w = 0
#r_a = 0


# %%
def run_sim_loop(j):
    print("seed: " + str(list_seeds[j]))
    np.random.seed(list_seeds[j])
    # random bias:
    bias_vec1 = 2*bias_base*(np.random.rand(3, 1) - 0.5)
    bias_vec2 = 2*bias_base*(np.random.rand(3, 1) - 0.5)
    bias_vec3 = 2*bias_base*(np.random.rand(3, 1) - 0.5)
    
    # Generate noise
    y_IMU1 = traj.gen_noise(y_IMU1_base.copy(), dt=dt, sigma=np.diag([r_w, r_a, r_a]), bias=bias_vec1, out_freq=out_freq)
    y_IMU2 = traj.gen_noise(y_IMU2_base.copy(), dt=dt, sigma=np.diag([r_w, r_a, r_a]), bias=bias_vec2, out_freq=out_freq)
    y_IMU3 = traj.gen_noise(y_IMU3_base.copy(), dt=dt, sigma=np.diag([r_w, r_a, r_a]), bias=bias_vec3, out_freq=out_freq)

    # Setup LUFT robots
    robotL1 = sim.robot_luft(x0=xa1, path=pos1, imu=y_IMU1, dt=dt, id=111, t=np.array([[1.0],[1.0]]))
    robotL2 = sim.robot_luft(x0=xa2, path=pos2, imu=y_IMU2, dt=dt, id=222, t=np.array([[1.0],[1.0]]))
    robotL3 = sim.robot_luft(x0=xa3, path=pos3, imu=y_IMU3, dt=dt, id=333, t=np.array([[1.0],[1.0]]))

    # Setup update params:
    robots_list = [robotL1, robotL2, robotL3]
    update_list = [anchor1, anchor2, robotL1, robotL2, robotL3]
    params = [sr, sb, thres_anc, thres_rob, sim_max_dist, amb, meas2_anc, meas2_rob, pout_r, pout_b]

    # Run Lufts algorithm
    for i in range(pos_len-1):
        robotL1.predict(imu_correct=True, thres=thres_IMU)
        robotL2.predict(imu_correct=True, thres=thres_IMU)
        robotL3.predict(imu_correct=True, thres=thres_IMU)
        # Set up a measurement pattern similar to the used communication protocol
        hfunc.updateAllLuft(robots_list, [update_list]*3, [params]*3, i, dt, update_freq, meas_delay)
    
    # Save estimates and covariances:
    return {
        'j': j,
        'id1' : robotL1.id,
        'id2' : robotL1.id,
        'id3' : robotL1.id,
        'x_logL1': robotL1.x_log,
        'P_logL1': robotL1.P_log,
        'x_logL2': robotL2.x_log,
        'P_logL2': robotL2.P_log,
        'x_logL3': robotL3.x_log,
        'P_logL3': robotL3.P_log,
        'nis_log1': robotL1.nis_IMU,
        'nis_log2': robotL2.nis_IMU,
        'nis_log3': robotL3.nis_IMU,
        'nis_rb_log1': robotL1.nis_rb,
        'nis_rb_log2': robotL2.nis_rb,
        'nis_rb_log3': robotL3.nis_rb,
        'rb_ids1': robotL1.rb_ids,
        'rb_ids2': robotL2.rb_ids,
        'rb_ids3': robotL3.rb_ids,
        'biases1': bias_vec1[:,0],
        'biases2': bias_vec2[:,0],
        'biases3': bias_vec3[:,0]
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
    x_logL1 = np.zeros((mf.STATE_LEN, pos_len, seeds_num))
    P_logL1 = np.zeros((mf.STATE_LEN, mf.STATE_LEN, pos_len, seeds_num))
    x_logL2 = np.zeros((mf.STATE_LEN, pos_len, seeds_num))
    P_logL2 = np.zeros((mf.STATE_LEN, mf.STATE_LEN, pos_len, seeds_num))
    x_logL3 = np.zeros((mf.STATE_LEN, pos_len, seeds_num))
    P_logL3 = np.zeros((mf.STATE_LEN, mf.STATE_LEN, pos_len, seeds_num))
    nis_log1 = np.zeros((pos_len, seeds_num))
    nis_log2 = np.zeros((pos_len, seeds_num))
    nis_log3 = np.zeros((pos_len, seeds_num))
    nis_rb_log1 = np.zeros((pos_len, seeds_num))
    nis_rb_log2 = np.zeros((pos_len, seeds_num))
    nis_rb_log3 = np.zeros((pos_len, seeds_num))
    rb_ids1 = np.zeros((pos_len, seeds_num))
    rb_ids2 = np.zeros((pos_len, seeds_num))
    rb_ids3 = np.zeros((pos_len, seeds_num))
    biases1 = np.zeros((3, seeds_num))
    biases2 = np.zeros((3, seeds_num))
    biases3 = np.zeros((3, seeds_num))

    for res in results:
        j = res['j']
        x_logL1[:,:,j] = res['x_logL1']
        P_logL1[:,:,:,j] = res['P_logL1']
        x_logL2[:,:,j] = res['x_logL2']
        P_logL2[:,:,:,j] = res['P_logL2']
        x_logL3[:,:,j] = res['x_logL3']
        P_logL3[:,:,:,j] = res['P_logL3']
        nis_log1[:,j] = res['nis_log1']
        nis_log2[:,j] = res['nis_log2']
        nis_log3[:,j] = res['nis_log3']
        nis_rb_log1[:,j] = res['nis_rb_log1']
        nis_rb_log2[:,j] = res['nis_rb_log2']
        nis_rb_log3[:,j] = res['nis_rb_log3']
        rb_ids1[:,j] = res['rb_ids1']
        rb_ids2[:,j] = res['rb_ids2']
        rb_ids3[:,j] = res['rb_ids3']
        biases1[:,j] = res['biases1']
        biases2[:,j] = res['biases2']
        biases3[:,j] = res['biases3']

    # finally save to an pickle object:
    robdat1 = sldat.RobotData(x_logL1, P_logL1, nis_log1, nis_rb_log1, pos1, biases1, rb_ids=rb_ids1, id=res['id1'])
    robdat2 = sldat.RobotData(x_logL2, P_logL2, nis_log2, nis_rb_log2, pos2, biases2, rb_ids=rb_ids2, id=res['id2'])
    robdat3 = sldat.RobotData(x_logL3, P_logL3, nis_log3, nis_rb_log3, pos3, biases3, rb_ids=rb_ids3, id=res['id3'])
    robcol = sldat.RobotCollection([robdat1, robdat2, robdat3], [xanc1, xanc2])
    sldat.save_data(robcol, 'para')


if __name__ == "__main__":
    main()


