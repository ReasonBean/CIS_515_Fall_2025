import numpy as np

def buildhardSVM2(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    builds the matrix of constraints A for
    hard SVM h2, and the right hand side
    Aso builds X and Xa = X'*X, and the vector q = -1_{p+q}
    for the linear part of the quadratic function 
    The right-hand side is c = 0 (Ax = 0).
    """
    p = u.shape[1]; q = v.shape[1]
    A = np.block([np.ones((1,p)), -np.ones((1,q))])
    c = np.zeros(1)
    X = np.block([-u, v])
    Xa = X.T@X
    q2 = -np.ones(p+q)
    return (A, c, X, Xa, q2)