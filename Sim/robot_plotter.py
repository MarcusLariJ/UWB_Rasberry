from models_functions import *
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import numpy as np

def setup_plot() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title("Robot trajectory")
    ax.set_xlabel("x [no unit]")
    ax.set_ylabel("y [no unit]")
    ax.set_xlim((-15,15))
    ax.set_ylim((-15,15))
    ax.grid()

    return fig, ax

def plot_variance_ellipse(ax: plt.Axes, P, x):
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
                            edgecolor='red')

    ax.add_patch(ellipse)

def plot_position(ax: plt.Axes, x: np.ndarray, color = 'b'):
    """
    Args:
        ax (plt.Axes): The ax of the main figure
        x (np.ndarray): The state to plot 
    """
    d = 2
    p = x[X_P, 0]
    theta = x[X_THETA, 0]
    ax.arrow(p[0], p[1], d*np.cos(theta), d*np.sin(theta), color=color)
    ax.scatter(p[0], p[1], c=color)

def plot_position2(ax: plt.Axes, x: np.ndarray, P: np.ndarray, color = 'b'):
    """
    Extended version of position plotting. Includes covariance ellipse.
    The function handles extracting the relevant expectations and covariances for x, y.
    Args:
        ax (plt.Axes): The ax of the main figure
        x (np.ndarray): The state to plot 
        P (np.ndarray): The covariance to plot
    """
    pos = x[X_P, 0]
    Pxy = P[X_P, X_P]
    plot_position(ax, x, color)
    plot_variance_ellipse(ax, Pxy, pos)

def plot_measurement(ax: plt.Axes, xi: np.ndarray, xj: np.ndarray):
    """
    Plot an arrow from robot i to robot j, to visualize a measurement made
    Args:
        ax (plt.Axes): The ax of the main figure
        xi (np.ndarray): The state of robot i recieving the measurement
        xj (np.ndarray): The state of the robot j sending the measurement
    """
    p1 = xi[X_P]
    p2 = xj[X_P]
    d = p2 - p1
    ax.arrow(p1[0], p1[1], d[0], d[1])