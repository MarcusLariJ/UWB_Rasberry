from models_functions import *
import robot_sim as rsim
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import numpy as np


def setup_plot(x=[0, 100], y=[0,70], figsize=(10,7)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title("Robot trajectory")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim((x[0],x[1]))
    ax.set_ylim((y[0],y[1]))
    ax.grid()
    ax.tick_params(direction='in')
    ax.set_xticks(np.arange(x[0], x[1]+1, 10))
    ax.set_yticks(np.arange(y[0]+10, y[1]+1, 10))

    return fig, ax

def plot_variance_ellipse(ax: plt.Axes, P, x, color, linestyle):
    """
    Draw ellipse from covariance matrix.
    # Code from matplotlib.org
    Args:
        ax (plt.Axes): The ax of the main figure
        P: Covariance matrix between x and y
        x: Position x and y
    """
    pearson = P[0, 1]/np.sqrt(P[0, 0] * P[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = patch.Ellipse((0, 0), 
                            width=ell_radius_x * 2, 
                            height=ell_radius_y * 2,
                            facecolor='none',
                            edgecolor=color,
                            linestyle=linestyle,
                            alpha=0.7)

    scale_x = np.sqrt(P[0, 0]) * 3 # three standard deviations
    scale_y = np.sqrt(P[1, 1]) * 3

    transf = patch.transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(x[0], x[1])

    ellipse.set_transform(transf + ax.transData)

    ax.add_patch(ellipse)

def plot_position(ax: plt.Axes, x: np.ndarray, color = 'b', draw_arrow=True, marker=',', linestyle='-', label=None):
    """
    Args:
        ax (plt.Axes): The ax of the main figure
        x (np.ndarray): The state to plot (given as theta, px, py)
    """
    d = 2.0
    len = x.shape[1]
    theta = x[0,:]
    px = x[1,:]
    py = x[2,:]

    ax.plot(px, py, color=color, marker=marker, linestyle=linestyle, label=label)
    ax.scatter(px, py, c=color, s=1.5)
    if draw_arrow:
        for i in range(len):
            ax.arrow(px[i], 
                     py[i], 
                     d*np.cos(theta[i]), 
                     d*np.sin(theta[i]), 
                     color=color, 
                     head_width=0.5, 
                     length_includes_head=True, 
                     linestyle=linestyle)
        return 

def plot_position2(ax: plt.Axes, x: np.ndarray, P: np.ndarray, color = 'b', draw_arrow=True, marker=',', linestyle='-', label=None):
    """
    Extended version of position plotting. Includes covariance ellipse.
    The function handles extracting the relevant expectations and covariances for x, y.
    Args:
        ax (plt.Axes): The ax of the main figure
        x (np.ndarray): The state to plot 
        P (np.ndarray): The covariance to plot
    """
    pos = x[X_P, :]
    plot_position(ax, np.concatenate((x[X_THETA:X_THETA+1,:], 
                                      x[X_P,:]),axis=0), 
                                      color, 
                                      draw_arrow=draw_arrow, 
                                      marker=marker,
                                      linestyle=linestyle,
                                      label=label)
    if P is not None:
        Pxy = P[X_P, X_P, :]
        for i in range(P.shape[2]):
            plot_variance_ellipse(ax, Pxy[:,:,i], pos[:,i], color=color, linestyle=linestyle)

def plot_measurement(ax: plt.Axes, xi: np.ndarray, xj: np.ndarray):
    """
    Plot an arrow from robot i to robot j, to visualize a measurement made
    Args:
        ax (plt.Axes): The ax of the main figure
        xi (np.ndarray): The state of robot i recieving the measurement
        xj (np.ndarray): The state of the robot j sending the measurement
    """
    p1 = xi[X_P,0]
    p2 = xj[X_P,0]
    d = p2 - p1
    ax.arrow(p1[0], p1[1], d[0], d[1], linestyle=':',head_width=0.1, length_includes_head=True)

def plot_measurement2(ax: plt.Axes, x: np.ndarray, r: float, phi: float):
    """
    Plot an arrow from robot i to robot j, to visualize a measurement made
    TODO: Maybe finish this
    Args:
        ax (plt.Axes): The ax of the main figure
        x (np.ndarray): The position of the anchor
        r (float): Range of measurement
        phi (float): Angle of measurement (remember to add orientation) 
    """
    p = x[X_P,0]
    d1 = np.cos(phi)*r
    d2 = np.sin(phi)*r 
    ax.arrow(p[0], p[1], d1, d2, linestyle=':',head_width=0.1, length_includes_head=True)

def plot_innovation(ax: plt.Axes, inno: np.ndarray, var: float, dt=1, color='blue'):
    """
    Plots the innovation, and check if it is consistent, that is within +3 std bounds
    """
    sigma = np.sqrt(var)
    sigma_line = np.ones_like(inno)*3*sigma
    t = np.linspace(0, inno.shape[0]*dt, inno.shape[0])
    ax.plot(t, inno, color=color)
    ax.plot(t, sigma_line, color=color, linestyle='--')
    ax.plot(t, -sigma_line, color=color, linestyle='--')

def plot_NEES(ax: plt.Axes, 
              x_est: np.ndarray, 
              x_true: np.ndarray, 
              P: np.ndarray,
              rad_sel: np.ndarray, 
              prob=0.95, 
              dt=1, 
              color='blue'):
    """
    Plots the NEES (Normalized Estimation Error Squared), and plots it along with the confidence bounds
    For a consistent system, the results should be equal to the degrees of freedom of the system/length of state vector
    """
    nees, t, r1, r2 = rsim.NEES(x_est=x_est, x_true=x_true, P=P, rad_sel=rad_sel, prob=prob, dt=dt)
    r1_line = np.ones_like(nees)*r1
    r2_line = np.ones_like(nees)*r2
    # Plot anees
    ax.plot(t, nees, color=color)
    ax.plot(t, r1_line, color=color, linestyle='--')
    ax.plot(t, r2_line, color=color, linestyle='--')
    ax.set_ylim([0, r2+1])
    
def plot_ANEES(ax: plt.Axes, 
              x_est: np.ndarray, 
              x_true: np.ndarray, 
              P: np.ndarray,
              rad_sel: np.ndarray, 
              prob=0.95, 
              dt=1, 
              color='blue',
              thres_c='black',
              label='anees'):
    """
    Plots the NEES (Normalized Estimation Error Squared), and plots it along with the confidence bounds
    For a consistent system, the results should be equal to the degrees of freedom of the system/length of state vector
    """
    anees, t, r1, r2 = rsim.ANEES(x_est=x_est, x_true=x_true, P=P, rad_sel=rad_sel, prob=prob, dt=dt)
    r1_line = np.ones_like(anees)*r1
    r2_line = np.ones_like(anees)*r2
    # Plot anees
    ax.semilogy(t, anees, color=color, label=label)
    ax.semilogy(t, r1_line, color=thres_c, linestyle='--')
    ax.semilogy(t, r2_line, color=thres_c, linestyle='--')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("ANEES")
    ax.grid()
    ax.tick_params(axis='x', direction='in', which='both')
    ax.tick_params(axis='y', direction='in', which='both')
    ax.set_xlim([0,t[-1]])

def plot_ANIS(ax: plt.Axes,
              nis,
              df,
              dt=1,
              prob = 0.95,
              color='blue',
              thres_c='black',
              label='anis'):
    """
    Plots the ANIS (Average Normalized innovation squared)
    """
    anis, t, r1, r2 = rsim.ANIS(nis, df, dt, prob)
    r1_line = np.ones_like(anis)*r1
    r2_line = np.ones_like(anis)*r2
    # Plot anees
    ax.scatter(t, anis, color=color, label=label, s=5)
    ax.plot(t, r1_line, color=thres_c, linestyle='--')
    ax.plot(t, r2_line, color=thres_c, linestyle='--')
    ax.set_yscale('log')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("ANIS")
    ax.grid()
    ax.tick_params(axis='x', direction='in', which='both')
    ax.tick_params(axis='y', direction='in', which='both')
    ax.set_xlim([0,t[-1]])

def plot_ANIS_rb(ax: plt.Axes,
                 nis: np.ndarray,
                 df: float,
                 rb_ids: list,
                 pos_ids: list,
                 labels: list,
                 dt: int=1,
                 prob: float =0.95,
                 thres_c='black'):
    """
    Plots the ANIS for range/bearing. 
    It colorcodes the readings based on which robot we are making a reading to
    """
    anis, t, r1, r2 = rsim.ANIS(nis, df, dt, prob)
    r1_line = np.ones_like(anis)*r1
    r2_line = np.ones_like(anis)*r2
    # Then we sort through the possible ids, and give each a color:
    N = len(pos_ids)
    for i in range(N):
        # Get the indicies:
        indxs = np.where(rb_ids == pos_ids[i])[0]
        # Then do a scatterplot:
        ax.scatter(t[indxs], anis[indxs], label=labels[i], s=5)
    # Plot the rest
    ax.plot(t, r1_line, color=thres_c, linestyle='--')
    ax.plot(t, r2_line, color=thres_c, linestyle='--')
    ax.set_yscale('log')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("ANIS")
    ax.grid()
    ax.tick_params(axis='x', direction='in', which='both')
    ax.tick_params(axis='y', direction='in', which='both')
    ax.set_xlim([0,t[-1]])

def plot_abs_avg(ax, 
                 abs_e, 
                 dt=1, 
                 pos=True, 
                 colors = ['black', 'blue', 'red'],
                 labels = ['orientation', 'position']):
    """
    Plots the average absolute error of position or biases
    Assumes ax has a shape of at least 2!
    Set pos to false to plot bias erros
    Only use the output of the bp wrapper!
    """
    if ax.size < 2:
        raise ValueError("ax too short. Should be at least 2")
    
    x_num = abs_e.shape[1]
    N = abs_e.shape[2]
    # first, get the average error:
    avg_e = 1.0/N*np.sum(abs_e, axis=2)
    t = np.linspace(0, x_num*dt, x_num)
    
    if pos:
        # Plot the positioning error:
        pos_norm = np.linalg.norm(avg_e[1:3,:], axis=0)
        ax[0].plot(t, avg_e[0,:], color=colors[0], label=labels[0])
        ax[1].plot(t, pos_norm, color=colors[1], label=labels[1])
        ax[0].set_ylabel("Orientation error [rad]")
        ax[1].set_ylabel("Position error [m]")
    else:
        # Plot the bias error:
        bias_norm = np.linalg.norm(avg_e[4:6,:], axis=0)
        ax[0].plot(t, avg_e[3,:], color=colors[0], label=labels[0])
        ax[1].plot(t, bias_norm, color=colors[1], label=labels[1])
        ax[0].set_ylabel("Angular rate bias error [rad/s]")
        ax[1].set_ylabel("Acceleration bias error [m/s^2]")
    
    # Common plot settings
    for i in range(2):
        ax[i].tick_params(direction='in')
        ax[i].grid()
        ax[i].set_xlabel("Time [s]")
        ax[i].set_xlim([0, t[-1]])


def plot_RMSE(ax: plt.Axes,
              x_est: np.ndarray,
              x_true: np.ndarray,
              biases: np.ndarray = None,
              dt=1):
    """
    Plot the RMSE for position states and biases
    """
    if biases is None:
        state_indx = [0,2,3]
        x_true2 = x_true
        rad_sel = np.array([[True], [False], [False]])
    else:
        state_indx = [0,2,3,8,9,10]
        x_true2 = np.append(x_true, np.repeat(biases, x_true.shape[1], axis=1), axis=0)
        rad_sel = np.array([[True], [False], [False], [False], [False], [False]])
    x_est2 = x_est[state_indx]

    rmse = rsim.RMSE(x_true2, x_est2, rad_sel)
    x_num = rmse.shape[1]
    t = np.linspace(0, x_num*dt, x_num)

    # Plot states:
    ax.plot(t, rmse[0])
    ax.plot(t, np.linalg.norm(rmse[1:3], axis=0))
    if biases is None:
        ax.legend(['Theta', 'Position'])
    else: 
        ax.plot(t, rmse[3])
        ax.plot(t, rmse[4])
        ax.plot(t, rmse[5])
        ax.legend(['theta', 'Position', 'Angular rate bias', 'x acc bias', 'y acc bias'])

