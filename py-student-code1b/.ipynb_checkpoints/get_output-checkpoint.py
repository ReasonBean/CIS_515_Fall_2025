import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable
from bspline2b import bspline2b
from get_points import get_points

def write_table(out: str, headers: list[str], dat: list[Iterable]):
    with open(out, "w") as outfile:
        outfile.writelines([",".join(headers) + "\n"])
        zipped = list(zip(*dat))
        outfile.writelines([",".join([str(x) for x in zipped[i]]) + "\n" for i in range(len(zipped))])

# Project 1B
# Python rewrite (SRM)
if not os.path.exists("output"):
    os.mkdir("output")
    os.mkdir(os.path.join("output", "part1"))
    os.mkdir(os.path.join("output", "part2"))
    os.mkdir(os.path.join("output", "part3"))
    os.mkdir(os.path.join("output", "code"))

dx: list[np.ndarray] = [np.zeros(0) for _ in range(5)]
dy: list[np.ndarray] = [np.zeros(0) for _ in range(5)]
dx[0] = np.array([3.6942,1.3690,2.9865,5.8509,8.1929,8.2098,6.8281])
dy[0] = np.array([1.2144,3.5925,7.3933,7.9217,6.9665,4.0396,1.5600])
dx[1] = np.array([3.9806,2.2789,3.6942,6.8618,7.1820])
dy[1] = np.array([2.1087,4.2429,7.0884,6.9461,4.3852])
dx[2] = np.array([4.2334,1.0826,1.3016,4.9579,8.2435,4.8062])
dy[2] = np.array([1.0315,3.6941,5.6250,7.9624,5.5640,5.8486])
dx[3] = np.array([4.6040,2.2283,3.3741,2.1609,7.2494,6.8955,9.1702])
dy[3] = np.array([1.3364,1.6616,3.5722,6.8242,8.6535,3.7957,2.8608])
dx[4] = np.array([5.4297,5.2275,2.9865,1.4532,2.1778,3.2898,6.8113,9.0691,7.2999,7.2157,9.2713,7.4853,6.4575])
dy[4] = np.array([4.6494,1.9055,1.7429,3.8974,8.0030,6.7429,9.1209,7.2917,6.4380,2.8404,2.7795,0.9502,1.3974])

showb = True
nn = 6
# Part 1 and 2
for showb in [False, True]:
    for i in range(len(dx)):
        N = len(dx[i])-1
        Bx, By = bspline2b(dx[i], dy[i], N, nn, showb)
        fname = os.path.join("output", "part1", f"{i+1}.txt")
        write_table(fname, [f"Bx_{j+1}" if j < Bx.shape[1] else f"By_{j+1-Bx.shape[1]}" for j in range(Bx.shape[1] + By.shape[1])], [Bx[:, j] if j < Bx.shape[1] else By[:, j-Bx.shape[1]] for j in range(Bx.shape[1] + By.shape[1])])
        try: # TODO: fix error checking here
            imname = os.path.join("output", "part2", f"showb{int(showb)}_{i+1}.png")
            plt.pause(0.5)
            plt.savefig(imname)
            plt.close('all')
        except:
            print("Problem drawing the plots")

# Part 3
for i in range(2):
    print("Please select >= 5 points for the spline (no requirements on how you pick them)")
    plt.figure()
    dx2, dy2 = get_points()
    N = len(dx2)-1
    nn = 7
    print(f"N = {N}")
    bspline2b(dx2,dy2,N, nn, showb)
    plt.savefig(os.path.join("output", "part3", f"{i+1}.png"))
    _ = plt.ginput(1)
    plt.close('all')