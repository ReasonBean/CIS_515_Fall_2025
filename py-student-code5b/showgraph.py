import numpy as np
import matplotlib.pyplot as plt

def showgraph(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Displays the graph {(x_1,y_1),...,(x_m,y_m)} 
    """
    Y = np.block([[X.T], [y.T]])

    # Find the smallest x and y coordinates
    xym = np.min(Y.T, axis=0)
    ll = xym
    ll -= 0.5
    # Finds the largest x and y coordinates
    xyM = np.max(Y.T, axis=0)
    mm = xyM
    mm += 0.5

    plt.figure()
    plt.ion()
    plt.plot(ll[0], ll[1])
    plt.plot(mm[0], mm[1])
    plt.plot(Y[0, :], Y[1, :], 'b*')    # Plots points

    return ll,mm