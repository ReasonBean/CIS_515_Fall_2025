import numpy as np
from bspline2b import bspline2b
import matplotlib.pyplot as plt

def show_bspline2b2(dx: np.ndarray, dy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    This is an auxilary function designed to output Bx and By
    for the version that uses dx and dy as input instead of
    screen input

    To plot a B-spline curve given by de Boor control points
    d_0, ..., d_N.
    Works if N >= 4
    This version uses the de Casteljau algorithm to plot the
    Bezier segments

     drawb = 1, shows Bezier control polygons

    nn = subdivision level for de Casteljau 
    """
    nn = 6

    N = len(dx)-1
    print(f"N = {N}\n")
    drawb = True
    # 
    Bx, By = bspline2b(dx, dy, N, nn, drawb)
    plt.ioff()
    # print(Bx)
    # print(By)

    return (Bx, By)