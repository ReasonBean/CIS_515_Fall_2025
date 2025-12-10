import numpy as np

def ridgeregb1(X: np.ndarray, y: np.ndarray, K: float) -> tuple[np.ndarray, float, np.ndarray, float, np.ndarray]:
    """
    Ridge regression 
    b is not penalized
    Uses the KKT equations
    X is an m x n matrix, y a m x 1 colum vector
    weight vector w, intercept b
    Solution in terms of the dual variables
    This version does not display the solution

    Warning: in Python, handling both n=1 and n>1 together takes some trickery - you may need to use np.newaxis to expand dims
    """
    # Basic setup
    if X.ndim == 1:
        X = X[:, np.newaxis]
    if y.ndim == 1:
        y = y[:, np.newaxis]

    m,n=X.shape

    # KKT (left)
    
    # top left
    TL=X@X.T+K*np.eye(m)

    # top right
    TR=np.ones((m,1))

    # bottom left
    BL=np.ones((1, m))

    # bottom right
    BR=np.zeros((1,1))

    KKT_matrix = np.block([
        [TL, TR],
        [BL, BR]
    ])

    # KKT (right)
    r_vector = np.vstack([y, [[0]]])

    # Solve linear system
    sol = np.linalg.solve(KKT_matrix, r_vector)

    # extract solution
    alpha = sol[:m]
    mu=sol[m].item()

    # recover param
    w=X.T@alpha
    b=mu
    xi=K*alpha
    nxi=np.linalg.norm(xi)
    
    #raise Exception("Implement me")
    #return np.zeros(0), 0, np.zeros(0), 0, np.zeros(0)
    return w, b, xi, nxi, alpha