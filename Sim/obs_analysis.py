import models_functions as mf
import numpy as np
import control as ct

# Observability analysis

dt = 0.01
mot = mf.MotionModel(dt=dt)
t = np.array([[0.1],[0.0]])
meas = mf.MeasModel(t=t)

# Perform Monte Carlo simulation
iter = 50000
np.random.seed(0)
rank_imu = 0
rank_rb_a = 0
rank_rb_b = 0
rank_rb2 = 0
mag = np.array([[np.pi], [100], [100], [20], [20], [0.03], [0.15], [0.15]])
mag2 = np.array([[2*np.pi], [100], [100]])
for i in range(iter):
    x0 = (np.random.rand(8, 1)-0.5)*mag
    u0 = (np.random.rand(3, 1)-0.5)*mag2
    x1 = (np.random.rand(8, 1)-0.5)*mag
    x2 = (np.random.rand(8, 1)-0.5)*mag
    x3 = (np.random.rand(8, 1)-0.5)*mag
    
    F = mot.get_A_B(x0=x0, u0=u0, dt=0.1)[0].copy()
    H_rb_a = meas.get_jacobian_rb(x0, x1, t).copy() # returns reference, therefore need new copy
    H_rb_b = meas.get_jacobian_rb(x0, x2, t).copy()
    H_rb2 = meas.get_jacobian_rb2(x0, x3, t).copy()

    # Observabiliy for normal RB
    H2 = np.vstack([H_rb_a])
    O2 = ct.obsv(F, H2)

    H3 = np.vstack([H_rb_a, H_rb_b])
    O3 = ct.obsv(F, H3)

    # Observability for RB2 
    H4 = np.vstack([H_rb2])
    O4 = ct.obsv(F, H4)

    rank_rb_a += np.linalg.matrix_rank(O2)
    rank_rb_b += np.linalg.matrix_rank(O3)
    rank_rb2 += np.linalg.matrix_rank(O4)

print(rank_rb_a/iter)
print(rank_rb_b/iter)
print(rank_rb2/iter)