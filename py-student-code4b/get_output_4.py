import numpy as np
import matplotlib.pyplot as plt
import os
from SVMhard2 import SVMhard2
from typing import Iterable

# Project 4
# Python rewrite (SRM) - note: much slower than matlab version due to slow numpy matrix inversion

if not os.path.exists('output'):
    os.mkdir('output')
    os.mkdir(os.path.join('output', 'autograde'))
    os.mkdir(os.path.join('output', 'images'))
    os.mkdir(os.path.join('output', 'report'))
    os.mkdir(os.path.join('output', 'code'))

def write_b(out: str, b: np.ndarray):
    with open(out, "w") as outfile:
        outfile.writelines(["b\n"])
        outfile.writelines([f"{b[i]}\n" for i in range(len(b))])

def read_dat(file: str, u_list: list[np.ndarray], v_list: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    with open(file, "r") as infile:
        lines = infile.readlines()
        for i in range(3):
            u1 = np.array([float(x) for x in lines[4*i].split()])
            u2 = np.array([float(x) for x in lines[4*i+1].split()])
            v1 = np.array([float(x) for x in lines[4*i+2].split()])
            v2 = np.array([float(x) for x in lines[4*i+3].split()])
            u_list[i+1] = np.block([[u1], [u2]])
            v_list[i+1] = np.block([[v1], [v2]])
        
    return u_list, v_list

v: list[np.ndarray] = [np.zeros(0) for _ in range(4)]
u: list[np.ndarray] = [np.zeros(0) for _ in range(4)]

v[0] = np.array([[1, 2, 3, 1, 1, 3, -1, -3],
                 [-1, 0, -2, -0.5, -4, -3, -3,-3]])
u[0] = np.array([[-1, -1, 0, 1, -3, -4, 0.5, 3, 0.5], 
                 [0, 1, 2, 3, 0, -2, 2, 2.5, 2.5]])

matlab_random = False
####################################
# This part is vital for the autograder so everyone gets the same results
# Reads random input so our results match with the matlab results
# Comment out this part below if using this for the report
matlab_random = True
####################################
if matlab_random:
    u, v = read_dat("matlab.dat", u, v)
else:
    u[1] = 10.1*np.random.randn(2,20)+15
    v[1] = -10.1*np.random.randn(2,20)-15
    u[2] = 10.1*np.random.randn(2,20)+10
    v[2] = -10.1*np.random.randn(2,20)-10
    u[3] = 10.1*np.random.randn(2,50)+18
    v[3] = -10.1*np.random.randn(2,50)-18

# Autograded part
rho = 10
for i in range(len(u)):
    lamb, mu, w, b = SVMhard2(rho, u[i], v[i])
    # For autograder
    fname = os.path.join('output', 'autograde', f'b{i+1}.txt')
    try:
        len(b)
        write_b(fname, b)
    except:
        write_b(fname, np.array([b]))
    
    try:
        imname = os.path.join('output', 'images', f'img{i+1}.png')
        plt.pause(0.5)
        plt.savefig(imname)
        plt.close('all')
    except:
        pass