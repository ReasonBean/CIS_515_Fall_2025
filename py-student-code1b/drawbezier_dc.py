import numpy as np
import matplotlib.pyplot as plt

"""
function to draw a Bezier segment
using de Casteljau subdivision
nn = level of subdivision
used by bspline4_dc
also plots the Bezier control polygons if drawb = 1
"""

def drawbezier_dc(B: np.ndarray, nn: int, drawb: bool):
    # nn is the subdivision level
    
    # t parameter (form 0 to 1)
    t = np.linspace(0, 1, 2**nn + 1)
    
    # Cub Bezier function
    # P(t) = (1-t)^3*B0 + 3(1-t)^2*t*B1 + 3(1-t)*t^2*B2 + t^3*B3
    P = (1 - t)**3 * B[:, [0]] + \
        3 * (1 - t)**2 * t * B[:, [1]] + \
        3 * (1 - t) * t**2 * B[:, [2]] + \
        t**3 * B[:, [3]]
        
    plt.plot(P[0, :], P[1, :], '-')

    # drawb -> control point and polygon
    if drawb:
        
        plt.plot(B[0, :], B[1, :], 'r+--')
    else:
        # control point (red, +)
        plt.plot(B[0, :], B[1, :], 'r+')
    
    return