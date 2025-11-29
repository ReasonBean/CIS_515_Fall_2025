import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def plotplane(w: np.ndarray, b: float, ll: np.ndarray, mm: np.ndarray, C1: tuple[float, float, float]):
    """
    This function plots a plane of equation
    z = (x, y)*w + b 
    ll(1) <= x <= mm(1)
    ll(2) <= y <= mm(2)
    
    C1 = color
    """
    u = np.array([ll[0], mm[0]])
    v = np.array([ll[1], mm[1]])
    U, V = np.meshgrid(u, v)
    m = U.shape[0]; n = U.shape[1]

    W = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            W[i,j] = w.T@np.block([[U[i,j]], [V[i,j]]]) + b
    Z = W

    ax = plt.gca()
    ax.plot_surface(U, V, Z, color=C1)
    plt.show() #TODO: Probably want ion() instead