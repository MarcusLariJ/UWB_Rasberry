from typing import List
import numpy as np

def RungeKutta(h: float, f: callable, x0: np.ndarray, u0: np.ndarray, *args: List):
    k1 = f(x0, u0, *args)
    k2 = f(x0+h*k1/2, u0, *args)
    k3 = f(x0+h*k2/2, u0, *args)
    k4 = f(x0+h*k3, u0, *args)
    return x0 + h/6*(k1 + 2*k2 + 2*k3 + k4)