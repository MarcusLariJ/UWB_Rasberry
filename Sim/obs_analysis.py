import models_functions as mf
import numpy as np
import control as ct

# Observability analysis

dt = 0.01
mot = mf.MotionModel(dt=dt)
t = np.array([[0.0],[0.0]])
meas = mf.MeasModel(t=t)

F = mot.A

# Perform Monte Carlo simulation
iter = 1000
np.random.seed(0)
rank_imu = 0
rank_rb_a = 0
rank_rb_b = 0
rank_rb2 = 0
mag = np.array([[np.pi], [2*np.pi], [100], [100], [1000], [1000], [10000], [10000], [np.pi], [10], [10]])
for i in range(iter):
    x0 = (np.random.rand(11, 1)-0.5)*mag
    x1 = (np.random.rand(11, 1)-0.5)*mag
    x2 = (np.random.rand(11, 1)-0.5)*mag
    x3 = (np.random.rand(11, 1)-0.5)*mag
    
    H_imu = meas.get_jacobian_IMU(x0)
    H_rb_a = meas.get_jacobian_rb(x0, x1, t).copy() # returns reference, therefore need new copy
    H_rb_b = meas.get_jacobian_rb(x2, x1, t).copy()
    H_rb2 = meas.get_jacobian_rb2(x0, x3, t)

    # Observability for imu update
    O1 = ct.obsv(F, H_imu)

    # Observabiliy for normal RB
    H2 = np.vstack([H_imu, H_rb_a])
    O2 = ct.obsv(F, H2)

    H3 = np.vstack([H_imu, H_rb_a, H_rb_b])
    O3 = ct.obsv(F, H3)

    # Observability for RB2 
    H4 = np.vstack([H_imu, H_rb2])
    O4 = ct.obsv(F, H4)

    rank_imu += np.linalg.matrix_rank(O1)
    rank_rb_a += np.linalg.matrix_rank(O2)
    rank_rb_b += np.linalg.matrix_rank(O3)
    rank_rb2 += np.linalg.matrix_rank(O4)

print(rank_imu/iter)
print(rank_rb_a/iter)
print(rank_rb_b/iter)
print(rank_rb2/iter)