import numpy as np
import matplotlib.pyplot as plt

def showdata(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Displays data points
    """
    uv = np.block([u, v])
    uvp = uv.T

    # Finds the smallest x and y coordinates
    I = np.argmin(uvp, axis=0)
    xym = np.diag(uvp[I])
    ll = xym - 0.5
    # Finds the largest x and y coordinates
    J = np.argmax(uvp, axis=0)
    xyM = np.diag(uvp[J])
    mm = xyM + 0.5

    # close
    plt.figure()
    plt.ion()
    plt.plot(ll[0], ll[1])
    plt.plot(mm[0], mm[1])
    plt.plot(u[0,:], u[1,:],'b*')
    plt.plot(v[0,:], v[1,:], 'rx')

    return ll, mm