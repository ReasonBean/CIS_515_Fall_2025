import numpy as np
import os
from interpatxy import interpatxy
from typing import Iterable
import matplotlib.pyplot as plt
from get_points import get_points

# Project 2
# Python rewrite (SRM)

if not os.path.exists("output"):
    os.mkdir("output")
    os.mkdir(os.path.join("output", "part2i"))
    os.mkdir(os.path.join("output", "part2ii"))
    os.mkdir(os.path.join("output", "part2iii"))
    os.mkdir(os.path.join("output", "report"))
    os.mkdir(os.path.join("output", "code"))

def write_table(out: str, headers: list[str], dat: list[Iterable]):
    with open(out, "w") as outfile:
        outfile.writelines([",".join(headers) + "\n"])
        zipped = list(zip(*dat))
        outfile.writelines([",".join([str(x) for x in zipped[i]]) + "\n" for i in range(len(zipped))])

# Test points
x: list[np.ndarray] = [np.zeros(0) for _ in range(5)]
y: list[np.ndarray] = [np.zeros(0) for _ in range(5)]
x[0] = np.array([3.6942,1.3690,2.9865,5.8509,8.1929,8.2098,6.8281])
y[0] = np.array([1.2144,3.5925,7.3933,7.9217,6.9665,4.0396,1.5600])
x[1] = np.array([3.9806,2.2789,3.6942,6.8618,7.1820])
y[1] = np.array([2.1087,4.2429,7.0884,6.9461,4.3852])
x[2] = np.array([4.2334,1.0826,1.3016,4.9579,8.2435,4.8062])
y[2] = np.array([1.0315,3.6941,5.6250,7.9624,5.5640,5.8486])
x[3] = np.array([4.6040,2.2283,3.3741,2.1609,7.2494,6.8955,9.1702])
y[3] = np.array([1.3364,1.6616,3.5722,6.8242,8.6535,3.7957,2.8608])
x[4] = np.array([5.4297,5.2275,2.9865,1.4532,2.1778,3.2898,6.8113,9.0691,7.2999,7.2157,9.2713,7.4853,6.4575])
y[4] = np.array([4.6494,1.9055,1.7429,3.8974,8.0030,6.7429,9.1209,7.2917,6.4380,2.8404,2.7795,0.9502,1.3974])
# More test points

# Part 2i and 2ii
show_plot = True
for i in range(len(x)):
    dx, dy, Bx, By = interpatxy(x[i], y[i])
    # Part 2i tables
    fname = os.path.join("output", "part2i", f"{i+1}.txt")
    write_table(fname, ["dx", "dy"], [dx, dy])

    fname = os.path.join("output", "part2ii", f"{i+1}.txt")
    write_table(fname, [f"Bx_{j+1}" if j < Bx.shape[1] else f"By_{j+1-Bx.shape[1]}" for j in range(Bx.shape[1] + By.shape[1])], [Bx[:, j] if j < Bx.shape[1] else By[:, j-Bx.shape[1]] for j in range(Bx.shape[1] + By.shape[1])])

    try: # TODO: fix error handling
        imname = os.path.join("output", "part2ii", f"img{i+1}.png")
        plt.pause(0.5)
        plt.savefig(imname)
        plt.close('all')
    except:
        pass

# Part 2iii
for i in range(2):
    print('Please select >= 5 points for the spline (no requirements on how you pick them)')
    plt.figure()
    x2, y2 = get_points()
    dx, dy, Bx, By = interpatxy(x2, y2)
    imname = os.path.join("output", "part2iii", f"img{i+1}.png")
    plt.pause(0.5)
    plt.savefig(imname)
    plt.close('all')