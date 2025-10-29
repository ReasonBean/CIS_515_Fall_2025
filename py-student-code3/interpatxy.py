import numpy as np
import matplotlib.pyplot as plt
from bspline2b import bspline2b

def interpatxy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This version uses the natural end condition
    Uses Matlab \ to solve linear systems
    Input points: two column vectors of x and y coordinates of dim N+1
    
    This version uses x_0, x_1, ..., x_{N-1}, x_N to compute the Bezier
    points and the subdivision version of the de Casteljau algorithm
    to plot the Bezier segments (bspline2b and drawbezier_dc)
    
    This version outputs the x and y coordinates dx and dy of the de Boor control
    points d_{-1}, d_0, d_1, ..., d_{N+1} as column vectors
    and the x and y coordinates of the Bezier control polygons
    Bx and By
    """

    dx: np.ndarray = np.zeros(0)
    dy: np.ndarray = np.zeros(0)
    Bx: np.ndarray = np.zeros(0)
    By: np.ndarray = np.zeros(0)
    # === Compute dx, dy, Bx, By here === #
    raise Exception("Implement me")

    # Plots the spline
    Nx = len(dx)-1
    print(f"Nx = {Nx}")
    nn = 6 # Subdivision level
    drawb = True
    Bx, By = bspline2b(dx,dy,Nx,nn,drawb) # Copy your version of bspline2b into bspline2b.py
    plt.ion()
    plt.plot(x, y, 'b+')
    plt.ioff()

    return (dx, dy, Bx, By)
