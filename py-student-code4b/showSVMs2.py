import numpy as np
import matplotlib.pyplot as plt
from makeline import makeline

def showSVMs2(w, b, eta, ll, mm, nw):
    """
    Function to display the result of running SVM
    on p blue points u_1, ..., u_p in u 
    and q red points v_1, ..., v_q in v
    """

    l = makeline(w,b,ll,mm,nw);        # makes separating line
    lm1 = makeline(w,b+eta,ll,mm,nw);  # makes blue margin line
    lm2 = makeline(w,b-eta,ll,mm,nw);  # makes red margin line

    plt.plot(l[0,:], l[1,:], '-m', linewidth=1.2)
    plt.plot(lm1[0,:], lm1[1,:], '-b', linewidth=1.2)
    plt.plot(lm2[0,:], lm2[1,:], '-r', linewidth=1.2)
    plt.ioff()
    return