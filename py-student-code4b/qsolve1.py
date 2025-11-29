import numpy as np
import numpy.linalg as linalg
import math

def qsolve1(P: np.ndarray, q: np.ndarray, A: np.ndarray, 
            b: np.ndarray, rho: float, tolr: float, tols: float, iternum: int) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    """
    Solve a quadratic programming problem
    min (1/2) x^T P x + x^T q + r
    subject to Ax = b, x >= 0  using ADMM
    P n x n,   q, r, in R^n, A m x n,  b in R^m
    A of rank m
    """

    m = A.shape[0]; print(f"m = {m}")
    n = P.shape[0]; print(f"n = {n}")
    u = np.ones(n); u[0] = 0 # To initialize u
    z = np.ones(n) # to initialize z
    # iternum = maximum number of iterations; 
    # iternum = 80000 works well 
    k = 0; nr = 1; ns = 1
    # typically tolr = 10^(-10); tols = 10^(-10);
    # Convergence is controlled by the norm nr of the primal residual r
    # and the norm ns of the dual residual s

    while k <= iternum and (ns > tols or nr > tolr):
        z0 = z
        k += 1
        # Make KKT matrix
        KK = np.block([[P + rho*np.eye(n,n), A.T], [A, np.zeros((m, m))]])
        # Makes right hand side of KKT equation
        bb = np.concatenate([-q + rho*(z-u), b])
        # Solves KKT equation
        xx = linalg.inv(KK)@bb
        # Update x, z, u (ADMM update steps)
        x = xx[:n]
        z = np.fmax(np.zeros(len(x)), x+u)
        u += x-z
        # test stopping criteria
        r = x-z                     # primal residual
        nr = math.sqrt(np.dot(r,r)) # norm of primal residual
        s = rho*(z-z0)              # dual residual
        ns = math.sqrt(np.dot(s,s)) # norm of dual residual

    return (x, u, nr, ns, k)