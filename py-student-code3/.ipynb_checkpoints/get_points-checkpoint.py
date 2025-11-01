import numpy as np
import matplotlib.pyplot as plt

"""
To input data points from a window on the screen
Left click on mouse to input points
middle, right or return to end input
"""
def get_points() -> tuple[np.ndarray, np.ndarray]:
    plt.cla()
    plt.ion()
    plt.axis([0, 10, 0, 10])
    points = plt.ginput(-1, show_clicks=True)
    plt.ioff()
    x, y = zip(*points)
    return np.array(x), np.array(y)