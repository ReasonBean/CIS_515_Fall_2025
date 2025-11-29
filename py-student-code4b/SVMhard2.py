import numpy as np
from buildhardSVM2 import buildhardSVM2
from qsolve1 import qsolve1
import math
from showdata import showdata
from showSVMs2 import showSVMs2

def SVMhard2(rho: float,u: np.ndarray,v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs hard margin SVM version 2   

    p green vectors u_1, ..., u_p in n x p array u
    q red   vectors v_1, ..., v_q in n x q array v

    First builds the matrices for the dual program
    """
    p = u.shape[1]; q = v.shape[1]; n = u.shape[0]
    A, c, X, Pa, qa = buildhardSVM2(u,v)
    # Runs quadratic solver
    tolr = 1e-10; tols = 1e-10; iternum = 120000
    lam,U,nr,ns,kk = qsolve1(Pa, qa, A, c, rho, tolr, tols, iternum)
    if kk > iternum:
        print("** qsolve did not converge. Problem not solvable **")
    lamb = lam[:p]
    mu = lam[p:p+q]

    w = np.zeros(0)
    b = np.zeros(0)
    numsvl1 = 0; numsvm1 = 0
    #####
    #  Solve for w and b here, as well as numsvl1 and numsvm1
    #  numsvl1 is the count for nonzero lambda and numsvm1 is the number of nonzero mu
    #####
    raise Exception("Implement me")

    #Some additional error checking
    nw = math.sqrt(w.T@w) # norm of w
    print(f"nw = {nw}")
    delta = 1/nw
    print(f"delta = {delta}")
    if delta < 1e-9:
        print('** Warning, delta too small, program does not converge **')
    
    if n == 2:
        ll,mm = showdata(u,v)
        if numsvl1 > 0 and numsvm1 > 0:
            showSVMs2(w,b,1,ll,mm,nw)
    
    return lamb, mu, w, b
