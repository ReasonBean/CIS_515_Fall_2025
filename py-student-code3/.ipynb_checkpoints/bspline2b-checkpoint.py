import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable
from drawbezier_dc import drawbezier_dc

def bspline2b(dx: np.ndarray|list[float], dy: np.ndarray|list[float], N: int, nn: int, drawb: bool) -> tuple[np.ndarray, np.ndarray]:
    # Works if N >= 4
    if N < 4:
        raise ValueError("N must be 4 or greater.")

    dx = np.asarray(dx).flatten()
    dy = np.asarray(dy).flatten()

    Bx = np.zeros((N - 2, 4))
    By = np.zeros((N - 2, 4))
   
    # N=4 Special Case
    if N == 4:
        Bx[0, 0], By[0, 0] = dx[0], dy[0]
        Bx[0, 1], By[0, 1] = dx[1], dy[1]
        Bx[0, 2], By[0, 2] = 0.5 * dx[1] + 0.5 * dx[2], 0.5 * dy[1] + 0.5 * dy[2]
        
        junction_x = 0.25 * dx[1] + 0.5 * dx[2] + 0.25 * dx[3]
        junction_y = 0.25 * dy[1] + 0.5 * dy[2] + 0.25 * dy[3]
        Bx[0, 3], By[0, 3] = junction_x, junction_y
        Bx[1, 0], By[1, 0] = junction_x, junction_y
        
        Bx[1, 1], By[1, 1] = 0.5 * dx[2] + 0.5 * dx[3], 0.5 * dy[2] + 0.5 * dy[3]
        Bx[1, 2], By[1, 2] = dx[3], dy[3]
        Bx[1, 3], By[1, 3] = dx[4], dy[4]

    # N >= 5 General Cases
    else:
        # Segment C1 (applies to all N >= 5)
        Bx[0, 0], By[0, 0] = dx[0], dy[0]
        Bx[0, 1], By[0, 1] = dx[1], dy[1]
        Bx[0, 2], By[0, 2] = 0.5 * dx[1] + 0.5 * dx[2], 0.5 * dy[1] + 0.5 * dy[2]
        Bx[0, 3], By[0, 3] = (1/4)*dx[1] + (7/12)*dx[2] + (1/6)*dx[3], (1/4)*dy[1] + (7/12)*dy[2] + (1/6)*dy[3]

        if N == 5:
            # For N=5, C2 is the second-to-last segment, C*(N-3).
            # So we apply the C*(N-3) rules to it.
            r = 1 # Row index for C2
            Bx[r, 0], By[r, 0] = Bx[r-1, 3], By[r-1, 3]
            Bx[r, 1], By[r, 1] = (2/3)*dx[2] + (1/3)*dx[3], (2/3)*dy[2] + (1/3)*dy[3]
            Bx[r, 2], By[r, 2] = (1/3)*dx[2] + (2/3)*dx[3], (1/3)*dy[2] + (2/3)*dy[3]
            
            # Use the special formula for b_{N-3, 3}
            Bx[r, 3], By[r, 3] = (1/6)*dx[2] + (7/12)*dx[3] + (1/4)*dx[4], (1/6)*dy[2] + (7/12)*dy[3] + (1/4)*dy[4]

            # For N=5, C3 is the last segment, C(N-2).
            r = 2 # Row index for C3
            Bx[r, 0], By[r, 0] = Bx[r-1, 3], By[r-1, 3]
            Bx[r, 1], By[r, 1] = 0.5*dx[N-2] + 0.5*dx[N-1], 0.5*dy[N-2] + 0.5*dy[N-1]
            Bx[r, 2], By[r, 2] = dx[N-1], dy[N-1]
            Bx[r, 3], By[r, 3] = dx[N], dy[N]
            
        else: # N >= 6
            
            # Segment C2
            Bx[1, 0], By[1, 0] = Bx[0, 3], By[0, 3]
            Bx[1, 1], By[1, 1] = (2/3)*dx[2] + (1/3)*dx[3], (2/3)*dy[2] + (1/3)*dy[3]
            Bx[1, 2], By[1, 2] = (1/3)*dx[2] + (2/3)*dx[3], (1/3)*dy[2] + (2/3)*dy[3]
            Bx[1, 3], By[1, 3] = (1/6)*dx[2] + (4/6)*dx[3] + (1/6)*dx[4], (1/6)*dy[2] + (4/6)*dy[3] + (1/6)*dy[4]

            # Generic Segments Ci (i = 3 to N-4)
            for i in range(3, N - 3):
                r = i - 1
                Bx[r, 0], By[r, 0] = (1/6)*dx[i-1]+(4/6)*dx[i]+(1/6)*dx[i+1], (1/6)*dy[i-1]+(4/6)*dy[i]+(1/6)*dy[i+1]
                Bx[r, 1], By[r, 1] = (2/3)*dx[i] + (1/3)*dx[i+1], (2/3)*dy[i] + (1/3)*dy[i+1]
                Bx[r, 2], By[r, 2] = (1/3)*dx[i] + (2/3)*dx[i+1], (1/3)*dy[i] + (2/3)*dy[i+1]
                Bx[r, 3], By[r, 3] = (1/6)*dx[i] + (4/6)*dx[i+1] + (1/6)*dx[i+2], (1/6)*dy[i] + (4/6)*dy[i+1] + (1/6)*dy[i+2]
            
            # Segment C(N-3)
            r = N - 4
            i = r + 1
            Bx[r, 0], By[r, 0] = Bx[r-1, 3], By[r-1, 3]
            Bx[r, 1], By[r, 1] = (2/3)*dx[i] + (1/3)*dx[i+1], (2/3)*dy[i] + (1/3)*dy[i+1]
            Bx[r, 2], By[r, 2] = (1/3)*dx[i] + (2/3)*dx[i+1], (1/3)*dy[i] + (2/3)*dy[i+1]
            Bx[r, 3], By[r, 3] = (1/6)*dx[i] + (7/12)*dx[i+1] + (1/4)*dx[i+2], (1/6)*dy[i] + (7/12)*dy[i+1] + (1/4)*dy[i+2]

            # Segment C(N-2)
            r = N - 3
            Bx[r, 0], By[r, 0] = Bx[r-1, 3], By[r-1, 3]
            Bx[r, 1], By[r, 1] = 0.5*dx[N-2] + 0.5*dx[N-1], 0.5*dy[N-2] + 0.5*dy[N-1]
            Bx[r, 2], By[r, 2] = dx[N-1], dy[N-1]
            Bx[r, 3], By[r, 3] = dx[N], dy[N]

    plt.figure()
    dim_data = 2
    B = np.zeros((dim_data, 4))
    plt.plot(dx, dy, 'or-')
    for i in range(N - 2):
        B[0, :] = Bx[i,:]
        B[1, :] = By[i,:]
        drawbezier_dc(B, nn, drawb)
    return (Bx, By)