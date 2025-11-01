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
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    N = len(x) - 1 # N is the number of segments
    
    dx: np.ndarray = np.zeros(0)
    dy: np.ndarray = np.zeros(0)
    Bx: np.ndarray = np.zeros(0)
    By: np.ndarray = np.zeros(0)
    
    # === Compute dx, dy, Bx, By here === #
    # d_prime = [d1, ..., d(N-1)]
    d_prime_x = np.zeros(N - 1)
    d_prime_y = np.zeros(N - 1)

    if N == 2:
        # For N=2 (3 points)
        # we get 4d1 = 6x1 - x0 - x2
        A = np.array([[4.0]])
        bx_rhs = np.array([6 * x[1] - x[0] - x[2]])
        by_rhs = np.array([6 * y[1] - y[0] - y[2]])
        d_prime_x = np.linalg.solve(A, bx_rhs)
        d_prime_y = np.linalg.solve(A, by_rhs)
    elif N == 3:
        # For N=3 (4 points)
        A = np.array([[4.0, 1.0], [1.0, 4.0]])
        bx_rhs = np.array([6 * x[1] - x[0], 6 * x[2] - x[3]])
        by_rhs = np.array([6 * y[1] - y[0], 6 * y[2] - y[3]])
        d_prime_x = np.linalg.solve(A, bx_rhs)
        d_prime_y = np.linalg.solve(A, by_rhs)
    else: # N >= 4
        A = np.zeros((N - 1, N - 1))
        np.fill_diagonal(A, 4)
        np.fill_diagonal(A[1:], 1)
        np.fill_diagonal(A[:, 1:], 1)
        
        bx_rhs = 6 * x[1:N]
        by_rhs = 6 * y[1:N]
        
        bx_rhs[0] -= x[0]
        by_rhs[0] -= y[0]
        bx_rhs[-1] -= x[N]
        by_rhs[-1] -= y[N]
        
        d_prime_x = np.linalg.solve(A, bx_rhs)
        d_prime_y = np.linalg.solve(A, by_rhs)

    # d0 and dN
    d0_x = (2/3) * x[0] + (1/3) * d_prime_x[0]
    d0_y = (2/3) * y[0] + (1/3) * d_prime_y[0]
    dN_x = (1/3) * d_prime_x[-1] + (2/3) * x[N]
    dN_y = (1/3) * d_prime_y[-1] + (2/3) * y[N]
    
    # Concat Boor points
    dx_plot = np.concatenate(([d0_x], d_prime_x, [dN_x]))
    dy_plot = np.concatenate(([d0_y], d_prime_y, [dN_y]))
    
    # Concat dx, dy for output (d-1 ... dN+1)
    dx = np.concatenate(([x[0]], dx_plot, [x[N]]))
    dy = np.concatenate(([y[0]], dy_plot, [y[N]]))
 
    #raise Exception("Implement me")

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
