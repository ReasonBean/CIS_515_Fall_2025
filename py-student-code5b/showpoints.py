import numpy as np
import matplotlib.pyplot as plt

def showpoints(X: np.ndarray, y: np.ndarray, offset: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Displays the graph {(x_1,y_1),...,(x_m,y_m)} 
    """
    # Finds the smallest x and y coordinates
    xym= np.min(X, axis=0)
    ll = xym
    ll -= offset
    # Finds the largest x and y coordinates
    xyM = np.max(X, axis=0)
    mm = xyM
    mm += offset
    Y = X.T
    Z = np.block([[Y], [y.T]])

    x = Z[0,:]
    y = Z[1,:]
    z = Z[2,:]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z, marker='*', color="blue")
    plt.ion()

    return ll,mm