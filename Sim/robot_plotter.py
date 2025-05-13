from models_functions import *
import robot_sim as rsim
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import numpy as np


def setup_plot(x=[0, 10], y=[0,7], figsize=(10,7)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title("Robot trajectory")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim((x[0],x[1]))
    ax.set_ylim((y[0],y[1]))
    ax.grid()
    ax.tick_params(direction='in')
    ax.set_xticks(np.arange(x[0], x[1]+1, 1))
    ax.set_yticks(np.arange(y[0]+1, y[1]+1, 1))

    return fig, ax

def plot_variance_ellipse(ax: plt.Axes, P, x, color):
    """
    Draw ellipse from covariance matrix.
    Args:
        ax (plt.Axes): The ax of the main figure
        P: Covariance matrix between x and y
        x: Position x and y
    """
    a = P[0,0]
    b = P[0,1]
    c = P[1,1]

    lam1 = (a + c)/2 + np.sqrt(((a-c)/2)**2 + b**2)
    lam2 = (a + c)/2 - np.sqrt(((a-c)/2)**2 + b**2)
    if (b == 0 and a >= c):
        theta = 0
    elif (b == 0 and a < c):
        theta = np.pi/2
    else:
        theta = np.arctan2(lam1-a, b)*(180/np.pi)
    
    ellipse = patch.Ellipse(xy=(x[0], x[1]), 
                            width=2*np.sqrt(lam1), 
                            height=2*np.sqrt(lam2), 
                            angle=theta,
                            facecolor='none',
                            edgecolor=color)

    ax.add_patch(ellipse)

def plot_position(ax: plt.Axes, x: np.ndarray, color = 'b'):
    """
    Args:
        ax (plt.Axes): The ax of the main figure
        x (np.ndarray): The state to plot (given as theta, px, py)
    """
    d = 0.25
    len = x.shape[1]
    theta = x[0,:]
    px = x[1,:]
    py = x[2,:]

    ax.plot(px, py, color=color)
    ax.scatter(px, py, c=color, s=1.5)
    for i in range(len):
        ax.arrow(px[i], py[i], d*np.cos(theta[i]), d*np.sin(theta[i]), color=color, head_width=0.05, length_includes_head=True)
    return 

def plot_position2(ax: plt.Axes, x: np.ndarray, P: np.ndarray, color = 'b'):
    """
    Extended version of position plotting. Includes covariance ellipse.
    The function handles extracting the relevant expectations and covariances for x, y.
    Args:
        ax (plt.Axes): The ax of the main figure
        x (np.ndarray): The state to plot 
        P (np.ndarray): The covariance to plot
    """
    pos = x[X_P, :]
    Pxy = P[X_P, X_P, :]
    plot_position(ax, np.concatenate((x[X_THETA:X_THETA+1,:], x[X_P,:]),axis=0), color)
    for i in range(P.shape[2]):
        plot_variance_ellipse(ax, Pxy[:,:,i], pos[:,i], color=color)

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
    
def plot_ANEES(ax: plt.Axes, 
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
    anees, t, r1, r2 = rsim.ANEES(x_est=x_est, x_true=x_true, P=P, rad_sel=rad_sel, prob=prob, dt=dt)
    r1_line = np.ones_like(anees)*r1
    r2_line = np.ones_like(anees)*r2
    # Plot anees
    ax.plot(t, anees, color=color)
    ax.plot(t, r1_line, color=color, linestyle='--')
    ax.plot(t, r2_line, color=color, linestyle='--')

def plot_RMSE(ax: plt.Axes,
              x_est: np.ndarray,
              x_true: np.ndarray,
              biases: np.ndarray = None,
              dt=1):
    """
    Plot the RMSE for position states and biases
    """
    if biases == None:
        state_indx = [0,2,3]
        x_true2 = x_true
        rad_sel = np.array([[True], [False], [False]])
    else:
        state_indx = [0,2,3,8,9,10]
        x_true2 = np.append(x_true, biases, axis=0)
        rad_sel = np.array([[True], [False], [False], [False], [False], [False]])
    x_est2 = x_est[state_indx]

    rmse = rsim.RMSE(x_true2, x_est2, rad_sel)
    x_num = rmse.shape[1]
    t = np.linspace(0, x_num*dt, x_num)

    # Plot states:
    ax.plot(t, rmse[0])
    ax.plot(t, np.linalg.norm(rmse[1:3], axis=0))
    if biases == None:
        ax.legend(['Theta', 'Position'])
    else: 
        ax.plot(t, rmse[3])
        ax.plot(t, rmse[4])
        ax.plot(t, rmse[5])
        ax.legend(['theta', 'x position', 'y position', 'Angular rate bias', 'x acc bias', 'y acc bias'])

