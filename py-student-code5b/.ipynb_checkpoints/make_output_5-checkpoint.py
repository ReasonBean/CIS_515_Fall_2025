import numpy as np
import matplotlib.pyplot as plt
import os
from ridgeregb1 import ridgeregb1
from ridgeregv1 import ridgeregv1
from ridgeregv2 import ridgeregv2
from reglq import reglq
from showgraph import showgraph
from showpoints import showpoints
from makeline import makeline
from plotplane import plotplane
# Project 5
# Python rewrite (SRM)
# Note: 3d plotting in matplotlib is absolutely borked, so your images may not look like those of your Matlab-using peers

def read_matlab_dat(file: str, xt: list[np.ndarray], yt: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    with open(file, "r") as infile:
        lines = infile.readlines()
        arrays: list[np.ndarray] = [np.array([float(x) for x in lines[i].split()]).T for i in range(len(lines))]
        # Grab all of the data
        X13 = arrays[0]
        y13 = arrays[1]
        X8 = np.block([[arrays[2]], [arrays[3]]]).T
        y8 = arrays[4]
        X10 = np.block([[arrays[5]], [arrays[6]]]).T
        y10 = arrays[7]
        X20 = np.block([[arrays[i+8]] for i in range(30)]).T
        y20 = arrays[38]

        xt[1] = X13
        yt[1] = y13
        xt[3] = X8
        yt[3] = y8
        xt[4] = X10
        yt[4] = y10
        xt[5] = X20
        yt[5] = y20

    return xt, yt

def reg3(X: np.ndarray, y: np.ndarray, K: float) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, float, np.ndarray,
                                                          np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Calls four regression methods
    Ridge regression minimizing w, b, not penalizing b
    Ridge regression minimizing w, b, not penalizing b, using the KKT eqs
    Ridge regression minimizing w, b, penalizing b
    Least squares, penalizing b
    X is an m x n matrix, y a m x 1 column vector
    weight vector w, intercept b
    """
    m = y.shape[0]; n = X.shape[1] if len(X.shape) > 1 else 1
    w1, _, b1, xi1, _ = ridgeregv1(X, y, K)
    w2, b2, xi2, _, alpha2 = ridgeregb1(X, y, K)
    w3, _, b3, xi3, _ = ridgeregv2(X, y, K)
    w4, _, b4, xi4, _ = reglq(X, y)
    print(f"b1 = {b1}")
    print(f"b2 = {b2}")
    print(f"b3 = {b3}")
    print(f"b4 = {b4}")
    if n == 1:
        ll,mm = showgraph(X, y)
        plt.ioff()
        ll,mm = showgraph(X, y)
        ww1 = np.block([[w1], [np.array([-1])]]); ww3 = np.block([[w3], [np.array([-1])]])
        ww4 = np.block([[w4], [np.array([-1])]])
        n1 = np.sqrt(np.dot(ww1.T,ww1))[0,0]; n3 = np.sqrt(np.dot(ww3.T,ww3))[0,0]
        n4 = np.sqrt(np.dot(ww4.T, ww4))[0,0]
        l1 = makeline(ww1,-b1,ll,mm,n1)
        # best fit, ridge 1
        l2 = makeline(ww3,-b3,ll,mm,n3)
        # best fit,
        # ridge penalizing b
        l3 = makeline(ww4,-b4,ll,mm,n4)
        # best fit, least squares
        plt.plot(l1[0,:],l1[1,:],'-m',linewidth=1.2)  # magenta best
        plt.plot(l2[0,:],l2[1,:],'-r',linewidth=1.2)  # red
        plt.plot(l3[0,:],l3[1,:],'-b',linewidth=1.2)  # blue
        plt.ioff()
    elif n == 2:
        offset = 5
        ll,mm = showpoints(X, y, offset)
        ax = plt.gca()
        ax.set_box_aspect([1, 1, 1]) # FIXME:
        ax.set_xlim([ll[0], mm[0]])
        ax.set_ylim([ll[1], mm[1]])
        ax.view_init(elev=45, azim=225)
        ax.set_xlabel('X', fontsize=14); ax.set_ylabel('Y', fontsize=14)
        ax.set_zlabel('Z', fontsize=14)
        plt.ioff()
        ll, mm = showpoints(X, y, offset)
        C3 = (0, 0, 1)  # blue
        C1 = (1, 0, 1)  # magenta
        C2 = (1, 0, 0)  # red
        plotplane(w1,b1,ll,mm,C1)   # best fit, ridge 1, magenta
        plotplane(w3,b3,ll,mm,C2)   # best fit, ridge penalizinbg b, red
        plotplane(w4,b4,ll,mm,C3)   # best fit, least squares, blue
        ax = plt.gca()
        ax.set_box_aspect([1, 1, 1]) # FIXME:
        ax.set_xlim([ll[0], mm[0]])
        ax.set_ylim([ll[1], mm[1]])
        ax.view_init(elev=45, azim=225)
        ax.set_xlabel('X', fontsize=14); ax.set_ylabel('Y', fontsize=14)
        ax.set_zlabel('Z', fontsize=14)
        plt.ioff()

    return w1,b1,xi1,w2,b2,xi2,alpha2,w3,b3,xi3,w4,b4,xi4

def write_table2d(out: str, headers: list[str], dat: list[np.ndarray]):
    if len(headers) != len(dat):
        raise ValueError("Number of headers != number of arrays in write_table2d")
    
    with open(out, "w") as outfile:
        numcs = [x.shape[1] if len(x.shape) > 1 else 1 for x in dat]
        # Write headers
        header_ts = []
        for i in range(len(headers)):
            if numcs[i] == 1:
                header_ts.append(headers[i])
            else:
                header_ts += [f"{headers[i]}_{j+1}" for j in range(numcs[i])]
        outfile.writelines([",".join(header_ts) + "\n"])

        # Format data
        to_zip: list[np.ndarray] = []
        for i in range(len(dat)):
            if numcs[i] == 1:
                to_zip.append(dat[i].flatten())
            else:
                to_zip += [dat[i][:,j] for j in range(numcs[i])]
        zipped = list(zip(*to_zip))

        # Write
        outfile.writelines([",".join([str(x) for x in zipped[i]]) + "\n" for i in range(len(zipped))])

if not os.path.exists('output'):
      os.mkdir('output')
      os.mkdir(os.path.join('output', 'part1'))
      os.mkdir(os.path.join('output', 'part2'))
      os.mkdir(os.path.join('output', 'part3'))
      os.mkdir(os.path.join('output', 'part4'))
      os.mkdir(os.path.join('output', 'images'))

X_test: list[np.ndarray] = [np.zeros(0) for _ in range(6)]
y_test: list[np.ndarray] = [np.zeros(0) for _ in range(6)]
matlab_random = False

######################################## 
# This part is vital for the autograder so everyone gets the same results
# Comment out this part below if using this for the report
# Reads in random data from matlab rng so out results match matlab results
matlab_random = True
########################################

X3 = np.array([-10, -8, -6, -4, -2, 2, 4, 6, 8, 10, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9]).T
y3 = np.array([-3, 1, 0, 0, 1.5, 4, 6, 5, 1, 8, -2.5, 0.5, 1.5, -1, -0.5, 3.5, 5.5, 2.5, 4.5, 5]).T
X = np.array([[-10, 11], [-6, 5], [-2, 4], [0, 0], [1, 2], [2, -5], [6, -4], [10, -6]])
y = np.array([0, -2.5, 0.5, -2, 2.5, -4.2, 1, 4]).T

X_test[0] = X3
y_test[0] = y3
X_test[2] = X
y_test[2] = y


if matlab_random:
    X_test, y_test = read_matlab_dat("matlab.dat", X_test, y_test)
else:
    X13 = 15*np.random.randn(50,1)
    ww13 = 1
    y13 = X13*ww13 + 10*np.random.randn(50,1) + 20

    X8 = 10*np.random.randn(50,2)
    ww = np.array([1, 2]).T
    y8 = X8*ww + 10*np.random.randn(50,1) + 10

    X10 = 10*np.random.randn(100,2)
    ww2 = np.array([1, 2]).T
    y10 = X10*ww2 + 10*np.random.randn(100,1) + 15

    X20 = np.random.randn(50,30)
    ww20 = np.array([0, 2, 0, -3, 0, -4, 1, 0, 2, 0, 2, 3, 0, -5, 6, 0, 1, 2, 0, 10, 0, 0, 3, 4, 5, 0, 0, -6, -8, 0])
    y20 = X20*ww20 + np.random.randn(50,1)*0.1 + 5

    X_test[1] = X13
    y_test[1] = y13
    X_test[3] = X8
    y_test[3] = y8
    X_test[4] = X10
    y_test[4] = y10
    X_test[5] = X20
    y_test[5] = y20


# Autograded part
rho = 10
for i in range(len(X_test)):
    print(i)
    m = X_test[i].shape[0]
    n = X_test[i].shape[1] if len(X_test[i].shape) > 1 else 1
    w1list = np.zeros((6, n)); b1list = np.zeros((6, 1)); xi1list = np.zeros((6, m))
    w2list = np.zeros((6, n)); b2list = np.zeros((6, 1)); xi2list = np.zeros((6, m))
    alpha2list = np.zeros((6, m))
    w3list = np.zeros((6,n)); b3list = np.zeros((6,1)); xi3list = np.zeros((6,m))
    w4list = np.zeros((1,n)); b4list = np.zeros((1,1)); xi4list = np.zeros((1,m))
    for k in [-2, -1, 0, 1, 2, 3]:
        K = 10**k
        print(K)
        w1,b1,xi1,w2,b2,xi2,alpha2,w3,b3,xi3,w4,b4,xi4 = reg3(X_test[i], y_test[i], K)
        # For autograder
        # Part 1
        w1list[k+2,:] = w1.T
        b1list[k+2,0] = b1
        xi1list[k+2,:] = xi1.T
        # Part 2
        w2list[k+2,:] = w2.T
        b2list[k+2,0] = b2
        xi2list[k+2,:] = xi2.T
        alpha2list[k+2,:] = alpha2.T
        # Part 3
        w3list[k+2,:] = w3.T
        b3list[k+2,0] = b3
        xi3list[k+2,:] = xi3.T
        # Part 4
        if k == 0: # No regularization by K in part 4
            w4list[0,:] = w4.T
            b4list[0,0] = b4
            xi4list[0,:] = xi4.T
        # Save image
        try:
            imname = os.path.join('output', 'images', f'img_{k+2}_{i+1}.png')
            plt.pause(0.5)
            plt.savefig(imname)
            plt.close('all')
        except:
            print("error saving image")
    # Write data to text file
    fname = os.path.join('output', 'part1', f"{i+1}.txt")
    write_table2d(fname, ['w1list', 'b1list', 'xi1list'], [w1list, b1list, xi1list])
    fname = os.path.join('output', 'part2', f"{i+1}.txt")
    write_table2d(fname, ['w2list', 'b2list', 'xi2list', 'alpha2list'], [w2list, b2list, xi2list, alpha2list])
    fname = os.path.join('output', 'part3', f"{i+1}.txt")
    write_table2d(fname, ['w3list', 'b3list', 'xi3list'], [w3list, b3list, xi3list])
    fname = os.path.join('output', 'part4', f"{i+1}.txt")
    write_table2d(fname, ['w4list', 'b4list', 'xi4list'], [w4list, b4list, xi4list])