import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable
from drawbezier_dc import drawbezier_dc

"""
To display a cubic B-sline given by de Boor control points
d_0, ..., d_N  

Input points: left click for the d's then press enter (or return, or right click)  

Performs a loop from 1 to N - 2 to compute the Bezier
points using de Casteljau subdivision
nn is the subdivision level

This version also outputs the x-coodinates and the y-coordinates
of all the control points of the Bezier segments stored in
Bx(N-2,4) and By(N-2,4)
"""

def bspline2b(dx: np.ndarray|list[float], dy: np.ndarray|list[float], N: int, nn: int, drawb: bool) -> tuple[np.ndarray, np.ndarray]:
    # Works if N >= 4

    Bx = np.zeros(0)
    By = np.zeros(0)
    # === COMPUTE Bx AND By HERE ===
    raise Exception("Implement me")

    # nn is the subdivision level
    plt.figure()
    dim_data = 2
    B = np.zeros((dim_data, 4))
    plt.plot(dx, dy, 'or-')
    plt.ion()
    for i in range(N-2):
        B[0, :] = Bx[i,:]
        B[1, :] = By[i,:]
        drawbezier_dc(B, nn, drawb)
    plt.ioff()

    return (Bx, By)
